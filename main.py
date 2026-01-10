import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

app = FastAPI()

# --- KONFIGURASI GGUF ---
# Repo khusus yang menyediakan versi GGUF (Hemat RAM)
# Kita pakai LFM 2.5 versi 1.2B yang sudah di-quantize ke 4-bit (Q4_K_M)
REPO_ID = "LiquidAI/LFM-1.2B-Instruct-GGUF" # Atau repo valid lainnya jika ini belum tersedia publik, bisa pakai 'bartowski/LFM-1.2B-Instruct-GGUF'
# Karena LFM 2.5 sangat baru, jika GGUF belum stabil, kita bisa fallback ke Qwen (lihat catatan bawah).
# Mari asumsikan kita pakai repo GGUF umum untuk testing, atau Qwen yang pasti jalan.
# SEMENTARA: Saya sarankan pakai Qwen2.5-0.5B GGUF dulu yang PASTI JALAN di 512MB RAM.
# Jika LFM GGUF sudah ada, ganti REPO_ID di bawah.

# Opsi 1: LFM (Jika repo GGUF-nya sudah valid di HF)
# MODEL_REPO = "LiquidAI/LFM-1.2B-Instruct-GGUF"
# MODEL_FILE = "lfm-1.2b-instruct-q4_k_m.gguf"

# Opsi 2: Qwen 0.5B (Rekomendasi Paling Aman & Cepat untuk Railway Free Tier)
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

print(f"--- MENYIAPKAN AI HEMAT RAM ({MODEL_REPO}) ---")

try:
    # 1. Download Model GGUF (Caching otomatis)
    print("Mendownload model GGUF (hanya sekali)...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )
    
    # 2. Load ke RAM (Hanya butuh ~500MB RAM!)
    print("Memuat ke RAM...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,       # Context window
        n_threads=2,      # Gunakan 2 core CPU
        verbose=False     # Matikan log berisik
    )
    print("✅ AI Siap!")

except Exception as e:
    print(f"❌ Error Init: {e}")
    # Jangan exit, biarkan server jalan biar bisa debug log
    llm = None

class AiRequest(BaseModel):
    prompt: str
    max_tokens: int = 512

@app.get("/")
def read_root():
    return {"status": "Online", "model": MODEL_FILE}

@app.post("/generate")
def generate_text(request: AiRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="Model gagal dimuat saat startup.")

    try:
        # Format Prompt ChatML (Qwen/LFM support format ini)
        # <|im_start|>system...
        prompt_formatted = f"""<|im_start|>system
You are a helpful assistant speaking valid JSON Indonesian.<|im_end|>
<|im_start|>user
{request.prompt}<|im_end|>
<|im_start|>assistant
"""

        output = llm(
            prompt_formatted,
            max_tokens=request.max_tokens,
            temperature=0.3,
            stop=["<|im_end|>"], # Stop token agar tidak ngawur
            echo=False
        )

        # Ambil text hasil
        return {"response": output['choices'][0]['text']}

    except Exception as e:
        print(f"Error Generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)