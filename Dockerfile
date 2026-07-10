FROM python:3.10-slim

WORKDIR /app

# Copy requirement dulu agar cache docker lebih efektif
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh folder backend ke dalam container
COPY backend/ .

# Jalankan aplikasi
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
