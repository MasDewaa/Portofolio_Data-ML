# Gunakan image dasar Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependensi sistem yang diperlukan untuk PyTorch dan Pillow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt ke dalam container
COPY requirements.txt /app/

# Install dependensi Python yang terdaftar di requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . /app

# Expose port untuk aplikasi FastAPI
EXPOSE 8000

# Jalankan aplikasi menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
