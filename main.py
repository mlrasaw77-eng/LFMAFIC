from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import json
import re
import os

app = FastAPI()

# --- KONFIGURASI MODEL (GGUF) ---
# Menggunakan Qwen 2.5 1.5B (Sangat cerdas, sangat ringan untuk CPU)
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

print("--- SYSTEM CHECK ---")
print(f"--- SEDANG MEMUAT MODEL GGUF: {REPO_ID} ---")

try:
    # n_ctx=2048: Context window (bisa dinaikkan jika RAM cukup, misal 4096)
    # n_threads=2: Sesuaikan dengan jumlah Core CPU VPS kamu
    llm = Llama.from_pretrained(
        repo_id=REPO_ID,
        filename=FILENAME,
        n_ctx=2048,
        n_threads=2, 
        verbose=False
    )
    print("✅ Model Berhasil Dimuat (Mode Hemat RAM)!")

except Exception as e:
    print(f"❌ Gagal memuat model: {str(e)}")
    exit(1)

# Struktur Data Request
class AiRequest(BaseModel):
    prompt: str
    max_tokens: int = 512 # Jangan terlalu besar di CPU

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
    return {"status": "Online", "device": "cpu-optimized", "model": REPO_ID}

@app.post("/generate")
def generate_text(request: AiRequest):
    try:
        # System Prompt
        messages = [
            {
                "role": "system", 
                "content": "You are a professional Career Consultant for Indonesian users. You MUST speak in standard, formal Indonesian (Bahasa Indonesia Baku). Output ONLY valid JSON."
            },
            {"role": "user", "content": request.prompt}
        ]

        # Generate menggunakan llama-cpp (Jauh lebih cepat dari transformers di CPU)
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=0.3,
            top_p=0.85,
            response_format={
                "type": "json_object" # Fitur native GGUF untuk memaksa output JSON
            }
        )

        response_text = output['choices'][0]['message']['content']
        
        print(f"--- RAW OUTPUT ---\n{response_text}\n------------------")

        cleaned_json = clean_json_output(response_text)
        
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            return {
                "summary": "Maaf, format error.",
                "recommendations": [],
                "nextSteps": ["Coba lagi."]
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)