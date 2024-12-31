from app.fetch_weather import fetch_weather_data
from app.process_data import calculate_average_temperature
from app.save_data import save_to_json
from app.config import API_KEY
"""
    The main function fetches weather data for a specified city, saves it to a JSON file, calculates the
    average temperature for the last 7 days, and displays the result.
"""
def main():
    
    city = input("Masukkan nama kota: ")
    output_file = "weather_data.json"

    try:
        if not API_KEY:
            raise ValueError("API key tidak ditemukan. Pastikan file .env sudah benar.")

        print(f"Mengambil data cuaca untuk kota {city}...")
        weather_data = fetch_weather_data(city)

        print(f"Menyimpan data ke dalam file {output_file}...")
        save_to_json(weather_data, output_file)

        print("Menghitung rata-rata suhu untuk 7 hari terakhir...")
        average_temp = calculate_average_temperature(weather_data)

        print(f"Rata-rata suhu di {city} untuk 7 hari terakhir adalah {average_temp:.2f} °C.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
