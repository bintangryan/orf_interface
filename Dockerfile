# Gunakan image Python yang ringan
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Copy requirements dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file backend lu (termasuk model dan dataset)
COPY . .

# Jalankan server FastAPI menggunakan port standar Cloud Run (8080)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
