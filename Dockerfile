# Pake image python 3.10 slim biar cepet
FROM python:3.10-slim

# Set working directory ke /app
WORKDIR /app

# Copy requirements.txt dari folder backend
COPY backend/requirements.txt .

# Install dependencies, DITAMBAH gdown untuk mendownload model
RUN pip install --no-cache-dir -r requirements.txt gdown

# Copy seluruh isi folder backend ke /app di dalam container
COPY backend/ .

# --- DOWNLOAD MODEL DARI GOOGLE DRIVE ---
# 1. Hapus file pointer Git LFS yang berukuran kecil/rusak
RUN rm -f model/model_IndoBERT_Benchmark_S1_20_TEXT_ONLY.pth

# 2. Download model asli pakai gdown. 
# PENTING: GANTI tulisan ID_FILE_KAMU_DISINI dengan ID dari link Google Drive milikmu!
RUN gdown --id 1c-LWVuSXTHvtIqmVuBxirp83fggR55j0 -O model/model_IndoBERT_Benchmark_S1_20_TEXT_ONLY.pth

# Expose port yang dipake FastAPI
EXPOSE 8080

# Jalankan uvicorn menggunakan variabel PORT bawaan Cloud Run
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
