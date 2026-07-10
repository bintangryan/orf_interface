# Pake image python 3.10 slim biar cepet
FROM python:3.10-slim

# Set working directory ke /app
WORKDIR /app

# Copy requirements.txt dari folder backend
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh isi folder backend ke /app di dalam container
COPY backend/ .

# Expose port yang dipake FastAPI
EXPOSE 8080

# Jalankan uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
