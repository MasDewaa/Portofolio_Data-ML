def calculate_average_temperature(data):
    """
    Menghitung rata-rata suhu harian selama 7 hari terakhir.

    :param data: Data cuaca dalam format JSON.
    :return: Rata-rata suhu dalam Celsius.
    """
    daily_temperatures = [(day["temp"]["day"] - 273.15) for day in data["daily"][:7]]
    return sum(daily_temperatures) / len(daily_temperatures)
