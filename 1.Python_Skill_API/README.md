# Weather Data Analysis Project

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [How It Works](#how-it-works)

## Introduction
This project demonstrates how to integrate with the OpenWeatherMap API to fetch weather data, save it into a JSON file, and perform simple data analysis. Specifically, the program calculates the average daily temperature for the next 7 days based on the fetched data.

## Features
- Fetch weather data for any city using the OpenWeatherMap API.
- Save weather data in a structured JSON format.
- Calculate and display the 7-day average daily temperature.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/weather-data-analysis.git
   cd weather-data-analysis
   ```

2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Obtain an API key from [OpenWeatherMap](https://openweathermap.org/api) and replace the placeholder `api_key` in the code with your key.

## Usage

1. Open the `main.py` file and specify:
   - `api_key`: Your OpenWeatherMap API key.
   - `city`: The city for which you want to fetch weather data.
   - `output_file`: The name of the JSON file to save the data.

2. Run the script:
   ```bash
   python main.py
   ```

3. The program will:
   - Fetch weather data for the specified city.
   - Save the data to the specified JSON file.
   - Calculate and display the average temperature for the next 7 days.

## Project Structure
```
weather-data-analysis/
├── main.py             # Main script to fetch and analyze weather data
├── requirements.txt    # Python dependencies
├── weather_data.json   # Output JSON file (generated dynamically)
└── README.md           # Project documentation
```

## Dependencies
- Python 3.7 or higher
- `requests` library (for HTTP requests)
- `json` (built-in for JSON file handling)

Install dependencies using the following command:
```bash
pip install requests
```

## How It Works
1. **Fetch Weather Data**:
   - Uses the OpenWeatherMap Geocoding API to convert city names to coordinates (latitude and longitude).
   - Fetches 7-day weather forecast data using the One Call API with the retrieved coordinates.

2. **Save to JSON**:
   - The fetched data is saved to a JSON file for offline analysis or debugging purposes.

3. **Analyze Weather Data**:
   - Extracts daily temperatures for the next 7 days from the JSON data.
   - Calculates the average daily temperature over the 7-day period and displays it in the console.
