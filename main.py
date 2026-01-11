from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import json
import re
import os

app = FastAPI()

# --- LOAD MODEL (Sama seperti sebelumnya) ---
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
# ... (Kode load model tetap sama) ...
try:
    llm = Llama.from_pretrained(
        repo_id=REPO_ID, filename=FILENAME, n_ctx=4096, n_threads=2, verbose=False
    )
except Exception as e:
    exit(1)

class AiRequest(BaseModel):
    prompt: str
    max_tokens: int = 800

def clean_json_output(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match: return match.group(1)
    return text.strip()

# --- FUNGSI LOGIKA AHP / RANKING ---
def calculate_ahp_ranking(recommendations):
    """
    Menghitung skor akhir berdasarkan bobot kriteria.
    Asumsi Bobot AHP (Bisa dinamis tergantung user preference):
    - Relevansi: 0.5 (50%)
    - Waktu (Singkat): 0.3 (30%)
    - Harga (Murah): 0.2 (20%)
    """
    ranked_results = []
    
    for item in recommendations:
        # Normalisasi nilai (Skala 1-10 yang diberikan AI)
        # Relevansi: Semakin tinggi semakin baik
        score_relevansi = item.get('relevance_score', 5)
        
        # Waktu: Semakin kecil (cepat) semakin baik. Kita balik nilainya.
        # Jika AI kasih score 10 (sangat lama), maka nilainya jadi kecil.
        # Rumus simplifikasi: 11 - score
        raw_time = item.get('time_efficiency_score', 5)
        score_waktu = 11 - raw_time 

        # Harga: Semakin murah (score affordability tinggi) semakin baik
        score_harga = item.get('affordability_score', 5)

        # RUMUS AHP (Weighted Sum)
        final_score = (score_relevansi * 0.5) + (score_waktu * 0.3) + (score_harga * 0.2)
        
        item['ahp_score'] = round(final_score, 2)
        ranked_results.append(item)

    # Sort dari skor tertinggi ke terendah
    return sorted(ranked_results, key=lambda x: x['ahp_score'], reverse=True)

@app.post("/generate-ahp")
def generate_ahp(request: AiRequest):
    try:
        # Prompt khusus yang meminta AI memberikan SKOR ANGKA (1-10)
        # Ini penting agar bisa dihitung matematis
        system_prompt = """You are an AI Analyst. 
        Your task is to recommend courses and assign NUMERICAL SCORES (1-10) for AHP calculation.
        
        Criteria Scoring Guide (1-10):
        - relevance_score: 1 (Not relevant) to 10 (Perfect match).
        - time_efficiency_score: 1 (Very long duration) to 10 (Very short/Fast).
        - affordability_score: 1 (Expensive) to 10 (Free/Cheap).
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt}
        ]

        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=0.6,
            response_format={"type": "json_object"}
        )

        cleaned_json = clean_json_output(output['choices'][0]['message']['content'])
        data = json.loads(cleaned_json)

        # --- PROSES AHP DI PYTHON ---
        # Jika AI mengembalikan list rekomendasi, kita hitung rankingnya
        if 'recommendations' in data and isinstance(data['recommendations'], list):
            # Update list dengan urutan baru hasil hitungan matematika
            data['recommendations'] = calculate_ahp_ranking(data['recommendations'])
            data['note'] = "Diurutkan menggunakan Metode Hybrid AHP (AI + Python Logic)"

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)