# 1. Gunakan Python versi ringan
FROM python:3.10-slim

# 2. Set folder kerja di dalam container
WORKDIR /app
#ini update baru
# 3. Install tool dasar linux
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. PENTING: Install PyTorch versi CPU-Only (Biar hemat storage & RAM)
# Jika install versi biasa, ukurannya 4GB++. Versi ini cuma 200MB.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install sisa library (FastAPI, Transformers, dll)
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy kode main.py ke dalam container
COPY . .

# 8. Perintah untuk menjalankan aplikasi
# Kita gunakan variable $PORT yang disediakan Railway
CMD uvicorn main:app --host 0.0.0.0 --port $PORT