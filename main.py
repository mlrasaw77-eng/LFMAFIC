from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import json
import re
import os

app = FastAPI()

# --- LOAD MODEL ---
REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

try:
    llm = Llama.from_pretrained(
        repo_id=REPO_ID, filename=FILENAME, n_ctx=4096, n_threads=2, verbose=False
    )
    print("✅ Server AI Siap (Mode: Standard + AHP)")
except Exception as e:
    exit(1)

class AiRequest(BaseModel):
    prompt: str
    max_tokens: int = 1000

def clean_json_output(text):
    start = text.find('{')
    end = text.rfind('}')
    return text[start:end+1] if start != -1 and end != -1 else text.strip()

# --- FUNGSI HITUNG AHP (Bisa untuk Kursus atau Skill) ---
def calculate_ahp_ranking(recommendations):
    # Bobot Eigen Vector dari Tugas Akhir Anda
    weights = {'C1': 0.5148, 'C2': 0.2393, 'C3': 0.1327, 'C4': 0.0737, 'C5': 0.0395}
    rating_map = {'Sangat Baik': 0.5148, 'Baik': 0.2393, 'Cukup': 0.1327, 'Kurang': 0.0737, 'Sangat Kurang': 0.0394}

    ranked = []
    for item in recommendations:
        # Mengambil rating (Misal: harga_rating atau skill_market_demand_rating)
        # Gunakan get dengan default label 'Cukup' agar aman
        s1 = rating_map.get(item.get('harga_rating', item.get('c1_rating', 'Cukup')), 0.1327)
        s2 = rating_map.get(item.get('rating_rating', item.get('c2_rating', 'Cukup')), 0.1327)
        s3 = rating_map.get(item.get('peminat_rating', item.get('c3_rating', 'Cukup')), 0.1327)
        s4 = rating_map.get(item.get('durasi_rating', item.get('c4_rating', 'Cukup')), 0.1327)
        s5 = rating_map.get(item.get('kesulitan_rating', item.get('c5_rating', 'Cukup')), 0.1327)

        score = (weights['C1']*s1) + (weights['C2']*s2) + (weights['C3']*s3) + (weights['C4']*s4) + (weights['C5']*s5)
        item['ahp_score'] = round(score, 4)
        ranked.append(item)
    return sorted(ranked, key=lambda x: x['ahp_score'], reverse=True)

@app.get("/")
def home():
    return {"status": "Online", "features": ["Skill Gap Analysis", "AHP Course Ranking"]}

# ENDPOINT 1: Untuk Analisis Skill (Tekstual)
@app.post("/generate")
def generate_standard(request: AiRequest):
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": request.prompt}],
        max_tokens=request.max_tokens,
        response_format={"type": "json_object"}
    )
    return json.loads(clean_json_output(output['choices'][0]['message']['content']))

# ENDPOINT 2: Untuk Rekomendasi Kursus (Matematis AHP)
@app.post("/generate-ahp")
def generate_ahp(request: AiRequest):
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": request.prompt}],
        max_tokens=request.max_tokens,
        response_format={"type": "json_object"}
    )
    data = json.loads(clean_json_output(output['choices'][0]['message']['content']))
    if 'recommendations' in data:
        data['recommendations'] = calculate_ahp_ranking(data['recommendations'])
    return data

if __name__ == "__main__":
    # Mengambil port dari environment variable, default ke 8001 jika tidak ada
    port = int(os.getenv("PORT", 8001)) 
    uvicorn.run(app, host="0.0.0.0", port=port)