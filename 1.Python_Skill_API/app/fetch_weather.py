import requests
from app.config import API_KEY

def fetch_weather_data(city):
    """
    Mengambil data cuaca untuk kota tertentu dari OpenWeatherMap API.

    :param city: Nama kota untuk mengambil data cuaca.
    :return: Data cuaca dalam format JSON.
    """
    geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
    params_geo = {
        "q": city,
        "appid": API_KEY
    }

    response_geo = requests.get(geocoding_url, params=params_geo)
    response_geo.raise_for_status()
    location_data = response_geo.json()

    if not location_data:
        raise ValueError("Kota tidak ditemukan.")

    lat = location_data[0]["lat"]
    lon = location_data[0]["lon"]

    weather_url = "http://api.openweathermap.org/data/3.0/onecall"
    params_weather = {
        "lat": lat,
        "lon": lon,
        "exclude": "current,minutely,hourly,alerts",
        "appid": API_KEY
    }
    response_weather = requests.get(weather_url, params=params_weather)
    response_weather.raise_for_status()

    return response_weather.json()
