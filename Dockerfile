# Gunakan Python Slim
FROM python:3.10-slim

WORKDIR /app

# 1. Install System Dependencies (Wajib untuk compile llama.cpp)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Requirements
COPY requirements.txt .

# 3. Upgrade PIP & Install Library
# Kita install llama-cpp-python pre-built atau compile manual
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Code
COPY . .

# 5. Jalankan
CMD uvicorn main:app --host 0.0.0.0 --port $PORT