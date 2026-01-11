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
print(f"--- MEMUAT MODEL AHP: {REPO_ID} ---")

try:
    llm = Llama.from_pretrained(
        repo_id=REPO_ID,
        filename=FILENAME,
        n_ctx=4096, 
        n_threads=2, 
        verbose=False
    )
    print("✅ Model Berhasil Dimuat dengan Dukungan AHP!")
except Exception as e:
    print(f"❌ Gagal: {str(e)}")
    exit(1)

class AiRequest(BaseModel):
    prompt: str
    max_tokens: int = 1000

def clean_json_output(text):
    """Mengekstrak JSON murni dari output AI"""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1]
    return text.strip()

def calculate_ahp_ranking(recommendations):
    """
    LOGIKA AHP BERDASARKAN FILE EXCEL TUGAS AKHIR
    """
    # Bobot Kriteria Utama (Eigen Vector Kriteria)
    weights = {
        'C1': 0.5148,  # Harga
        'C2': 0.2393,  # Rating
        'C3': 0.1327,  # Peminat
        'C4': 0.0737,  # Durasi Belajar
        'C5': 0.0395   # Tingkat Kesulitan
    }

    # Bobot Sub-Kriteria (Eigen Vector Normalisasi Sub-Kriteria)
    rating_map = {
        'Sangat Baik': 0.5148,
        'Baik': 0.2393,
        'Cukup': 0.1327,
        'Kurang': 0.0737,
        'Sangat Kurang': 0.0394
    }

    ranked_results = []
    for item in recommendations:
        # Ambil label dari AI, jika tidak ada/salah ketik default ke 'Cukup'
        s_c1 = rating_map.get(item.get('harga_rating', 'Cukup'), 0.1327)
        s_c2 = rating_map.get(item.get('rating_rating', 'Cukup'), 0.1327)
        s_c3 = rating_map.get(item.get('peminat_rating', 'Cukup'), 0.1327)
        s_c4 = rating_map.get(item.get('durasi_rating', 'Cukup'), 0.1327)
        s_c5 = rating_map.get(item.get('kesulitan_rating', 'Cukup'), 0.1327)

        # RUMUS: Total Skor = Σ (Bobot Kriteria * Bobot Rating)
        total_score = (
            (weights['C1'] * s_c1) +
            (weights['C2'] * s_c2) +
            (weights['C3'] * s_c3) +
            (weights['C4'] * s_c4) +
            (weights['C5'] * s_c5)
        )
        
        item['ahp_score'] = round(total_score, 4)
        # Tambahkan label deskriptif untuk frontend
        item['priority_rank'] = "High" if total_score > 0.3 else ("Medium" if total_score > 0.15 else "Low")
        ranked_results.append(item)

    # Urutkan berdasarkan skor tertinggi (Alternatif Terbaik)
    return sorted(ranked_results, key=lambda x: x['ahp_score'], reverse=True)

@app.get("/")
def read_root():
    return {"status": "Online", "method": "AHP-Integrated", "model": REPO_ID}

@app.post("/generate-ahp")
def generate_ahp(request: AiRequest):
    try:
        messages = [
            {
                "role": "system", 
                "content": "You are a Decision Support System (DSS) using AHP method. Output ONLY valid JSON in Bahasa Indonesia."
            },
            {"role": "user", "content": request.prompt}
        ]

        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=0.7,
            repeat_penalty=1.1,
            response_format={"type": "json_object"}
        )

        response_text = output['choices'][0]['message']['content']
        cleaned_json = clean_json_output(response_text)
        data = json.loads(cleaned_json)

        # Proses Ranking AHP jika ada list rekomendasi
        if 'recommendations' in data and isinstance(data['recommendations'], list):
            data['recommendations'] = calculate_ahp_ranking(data['recommendations'])
            data['ahp_metadata'] = {
                "status": "Sorted by AHP Eigen Weights",
                "top_priority": data['recommendations'][0]['title'] if data['recommendations'] else None
            }

        return data

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)