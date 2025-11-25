"""Module to load weather_data data from an Excel file."""
import pandas as pd

DATA_PATH_WEATHER = r"/mnt/c/Users/KaySmitsInfront/PycharmProjects/thesis_KaySmits_Wärtsilä/sea_trial_GPS_weather.xlsx"

def load_speed_trials(data_path):
    """Load and clean speed trials data from Excel file."""
    weather_data = pd.read_excel(data_path)
    print("Weather data loaded with shape:", weather_data.shape, )
    return weather_data

WEATHER_DATA = load_speed_trials(DATA_PATH_WEATHER)
