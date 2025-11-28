# Python base image
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tüm dosyaları kopyala
COPY . .

# Requirements yükle
RUN pip install --no-cache-dir -r requirements.txt

# Port
EXPOSE 8080

# Uygulamayı başlat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]