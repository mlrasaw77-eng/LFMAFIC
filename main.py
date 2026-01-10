from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import json
import re
import os

# Matikan warning oneDNN (opsional, agar log bersih)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

# --- KONFIGURASI MODEL ---
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"

print("--- SYSTEM CHECK ---")
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU NVIDIA Terdeteksi: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠️ GPU tidak terdeteksi. Menggunakan CPU.")

print(f"--- SEDANG MEMUAT MODEL: {MODEL_ID} ---")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # SETTING PENTING UNTUK RTX 3050 (6GB)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # Gunakan float16 agar hemat VRAM (wajib untuk GPU 6GB)
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Pindahkan ke GPU
    model.to(device)
    
    print("✅ Model Berhasil Dimuat ke GPU!")

except Exception as e:
    print(f"❌ Gagal memuat model: {str(e)}")
    exit(1)

# Struktur Data Request
class AiRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024

def clean_json_output(text):
    """Membersihkan output agar menjadi JSON murni"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    return text

@app.get("/")
def read_root():
    return {"status": "Online", "device": device, "model": MODEL_ID}

@app.post("/generate")
def generate_text(request: AiRequest):
    try:
        # PERBAIKAN 1: System Prompt yang LEBIH GALAK soal bahasa
        messages = [
            {
                "role": "system", 
                "content": "You are a professional Career Consultant for Indonesian users. You MUST speak in standard, formal Indonesian (Bahasa Indonesia Baku). Do NOT use Malay, Spanish, or English words. Output ONLY valid JSON."
            },
            {"role": "user", "content": request.prompt}
        ]
        
        # Suppress warning attention mask
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                # PERBAIKAN 2: Tuning Parameter agar bahasa lebih baku
                temperature=0.3,        # Turunkan dari 0.7 ke 0.3 (biar tidak ngarang kata)
                top_p=0.85,             # Sedikit diperketat
                top_k=40,
                repetition_penalty=1.15, # Cegah pengulangan tapi jangan terlalu tinggi biar kalimat wajar
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=input_ids.ne(tokenizer.pad_token_id) # Fix warning attention mask
            )

        response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        print(f"--- RAW OUTPUT ---\n{response_text}\n------------------")

        cleaned_json = clean_json_output(response_text)
        
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            # Fallback darurat: Jika JSON rusak, return error biar tidak crash di Laravel
            return {
                "summary": "Maaf, terjadi kesalahan format. Silakan coba lagi.",
                "courses": [],
                "skillGaps": [],
                "recommendations": [],
                "industryTrends": [],
                "nextSteps": ["Coba generate ulang."],
                "learningPath": "",
                "tips": []
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ambil PORT dari Environment Variable Railway, default ke 8001 jika di local
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)