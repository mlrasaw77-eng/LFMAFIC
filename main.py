from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import json
import re
import os

app = FastAPI()

# --- KONFIGURASI MODEL ---
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

print("--- SYSTEM CHECK ---")
print(f"--- SEDANG MEMUAT MODEL: {REPO_ID} ---")

try:
    # UPDATED: n_ctx dinaikkan ke 4096 agar kuat menampung prompt panjang dari Laravel
    llm = Llama.from_pretrained(
        repo_id=REPO_ID,
        filename=FILENAME,
        n_ctx=4096, 
        n_threads=2, 
        verbose=False
    )
    print("✅ Model Berhasil Dimuat (Setting: Context 4096, Threads 2)!")

except Exception as e:
    print(f"❌ Gagal memuat model: {str(e)}")
    exit(1)

class AiRequest(BaseModel):
    prompt: str
    max_tokens: int = 800 # Default dinaikkan sedikit

def clean_json_output(text):
    """Membersihkan markdown block ```json ... ``` agar menjadi raw JSON"""
    # Hapus ```json di awal
    text = re.sub(r'```json\s*', '', text)
    # Hapus ``` di akhir
    text = re.sub(r'```', '', text)
    # Ambil hanya yang ada di dalam kurung kurawal pertama dan terakhir
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

@app.get("/")
def read_root():
    return {"status": "Online", "model": REPO_ID}

@app.post("/generate")
def generate_text(request: AiRequest):
    try:
        # System Prompt yang lebih tegas
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful AI Career Assistant. You output ONLY valid JSON. Do not output any conversational text outside the JSON block."
            },
            {"role": "user", "content": request.prompt}
        ]

        print(f"--- PROCESSING REQUEST (Length: {len(request.prompt)}) ---")

        # Generate output
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            # TUNING PENTING DISINI:
            temperature=0.6,       # 0.6 - 0.7: Seimbang antara kreatif & patuh aturan
            top_p=0.9,             # Variasi kata lebih luas
            repeat_penalty=1.1,    # Mencegah AI mengulang-ulang prompt user
            response_format={
                "type": "json_object" # Memaksa output JSON (native feature)
            }
        )

        response_text = output['choices'][0]['message']['content']
        
        # Debugging: Print output mentah ke terminal Docker
        print(f"--- RAW OUTPUT FROM AI ---\n{response_text[:200]}...\n------------------")

        cleaned_json = clean_json_output(response_text)
        
        try:
            # Coba parse ke Python Dict
            data = json.loads(cleaned_json)
            return data
        except json.JSONDecodeError:
            print("❌ JSON Decode Error. Raw text was invalid.")
            # Fallback jika gagal parse, agar Laravel tidak crash total
            return {
                "summary": "Terjadi kesalahan format data dari AI.",
                "skillGaps": [],
                "recommendations": [],
                "industryTrends": [],
                "nextSteps": ["Silakan coba request ulang."]
            }

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)