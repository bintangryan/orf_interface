FROM python:3.11-slim

# Mencegah Python membuat file .pyc dan membuffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh project
COPY . .

# Cloud Run menggunakan port 8080
ENV PORT=8080
EXPOSE 8080

# Jalankan FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]