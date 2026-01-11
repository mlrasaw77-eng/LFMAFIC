FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (Wajib untuk llama-cpp)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]