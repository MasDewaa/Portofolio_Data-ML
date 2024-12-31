import os
from dotenv import load_dotenv

# Memuat file .env
load_dotenv()

# Mengambil API key dari file .env
API_KEY = os.getenv("API_KEY")
