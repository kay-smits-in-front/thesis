"""Cleaning of weather data for analysis."""
from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from load_data.weather_data import WEATHER_DATA


def clean_weather_data(verbose=True):
    """Match and interpolate weather data to speed trials."""
    speed_trials_weather = SPEED_TRIALS_REGULAR.copy()

    if verbose:
        print(f"Starting with cleaned speed trials: {speed_trials_weather.shape}")

    # Create datetime from Date and Time
    speed_trials_weather['datetime_temp'] = pd.to_datetime(
        speed_trials_weather['Date'].astype(str) + ' ' + speed_trials_weather['Time'].astype(str),
        errors='coerce'
    )

    weather_data = WEATHER_DATA.copy()
    weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'], utc=True)
    speed_trials_weather['datetime_temp'] = speed_trials_weather['datetime_temp'].dt.tz_localize('UTC')
    weather_data = weather_data.sort_values('timestamp').reset_index(drop=True)

    weather_numeric_cols = ['mean_wave_direction', 'mean_wave_period', 'significant_wave_height',
                            'wind_u_component_10m', 'wind_v_component_10m', 'air_density',
                            'wind_speed_10m', 'wind_direction_10m']

    for col in weather_numeric_cols:
        weather_data[col] = pd.to_numeric(weather_data[col], errors='coerce')

    weather_columns = [col for col in weather_data.columns if col not in ['latitude', 'longitude', 'timestamp']]

    for col in weather_columns:
        speed_trials_weather[col] = np.nan

    if verbose:
        print(f"Matching weather data to speed trials...")

    for idx, row in speed_trials_weather.iterrows():
        st_time = row['datetime_temp']
        if pd.isna(st_time):
            continue
        time_diff = (weather_data['timestamp'] - st_time).abs()
        match_idx = time_diff.idxmin()

        if time_diff[match_idx] <= pd.Timedelta(hours=1):
            for col in weather_columns:
                speed_trials_weather.at[idx, col] = weather_data.at[match_idx, col]

    # Create elapsed_seconds for interpolation
    speed_trials_weather['elapsed_seconds'] = (
            speed_trials_weather['datetime_temp'] - speed_trials_weather['datetime_temp'].min()
    ).dt.total_seconds()

    speed_trials_weather = speed_trials_weather.drop(columns=['datetime_temp'])

    if verbose:
        weather_missing = speed_trials_weather[weather_columns].isna().all(axis=1).sum()
        print(f"Rows without any weather data: {weather_missing}")

    # Interpolate missing weather data
    if verbose:
        print(f"Interpolating weather data...")

    for col in weather_numeric_cols:
        if col not in speed_trials_weather.columns:
            continue

        x = speed_trials_weather['elapsed_seconds'].values
        y = speed_trials_weather[col].values
        valid_idx = ~np.isnan(y)

        if valid_idx.sum() < 2:
            continue

        speed_trials_weather[col] = speed_trials_weather[col].interpolate(method='linear', limit_direction='both')

    if verbose:
        print(f"Final shape: {speed_trials_weather.shape}")
        for col in weather_columns:
            if col in speed_trials_weather.columns:
                missing = speed_trials_weather[col].isna().sum()
                pct = (missing / len(speed_trials_weather)) * 100
                print(f"  {col}: {missing} missing ({pct:.1f}%)")

    return speed_trials_weather


SPEED_TRIALS_WEATHER_CLEAN = clean_weather_data(verbose=False)

# Save to CSV for verification
import os
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)
SPEED_TRIALS_WEATHER_CLEAN.to_csv(os.path.join(OUTPUT_PATH, "SPEED_TRIALS_WEATHER_CLEAN.csv"), index=False)
print(f"SPEED_TRIALS_WEATHER_CLEAN saved to: {os.path.join(OUTPUT_PATH, 'SPEED_TRIALS_WEATHER_CLEAN.csv')}")


if __name__ == "__main__":
    print(f"SPEED_TRIALS_WEATHER_CLEAN shape: {SPEED_TRIALS_WEATHER_CLEAN.shape}")