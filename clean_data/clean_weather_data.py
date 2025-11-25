"""Cleaning of weather data for analysis."""

from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from load_data.weather_data import WEATHER_DATA


def clean_weather_data(verbose=True):
	speed_trials_weather = SPEED_TRIALS_REGULAR.copy()

	if verbose:
		print(f"Starting with cleaned speed trials: {speed_trials_weather.shape}")

	base_time = pd.Timestamp('2000-01-01', tz='UTC')
	speed_trials_weather['datetime_temp'] = base_time + pd.to_timedelta(speed_trials_weather['elapsed_seconds'], unit='s')

	weather_data = WEATHER_DATA.copy()
	weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'], utc=True)
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
		time_diff = (weather_data['timestamp'] - st_time).abs()
		match_idx = time_diff.idxmin()

		if time_diff[match_idx] <= pd.Timedelta(minutes=2):
			for col in weather_columns:
				speed_trials_weather.at[idx, col] = weather_data.at[match_idx, col]

	speed_trials_weather = speed_trials_weather.drop(columns=['datetime_temp'])

	# NO THRESHOLD REMOVAL - keep all rows with weather data
	if verbose:
		weather_missing = speed_trials_weather[weather_columns].isna().all(axis=1).sum()
		print(f"Rows without any weather data: {weather_missing}")

	def interpolate_weather_data_by_dataset(df, time_col='elapsed_seconds'):
		weather_cols = ['mean_wave_direction', 'mean_wave_period', 'significant_wave_height',
		                'wind_u_component_10m', 'wind_v_component_10m', 'air_density',
		                'wind_speed_10m', 'wind_direction_10m']

		df_result = df.copy()

		for dataset_id in df_result['dataset_id'].unique():
			mask = df_result['dataset_id'] == dataset_id
			dataset_subset = df_result[mask]

			x = dataset_subset[time_col].values

			for col in weather_cols:
				if col not in df_result.columns:
					continue

				y = dataset_subset[col].values
				valid_idx = ~np.isnan(y)

				if valid_idx.sum() < 2:
					continue

				x_valid = x[valid_idx]
				y_valid = y[valid_idx]

				sort_order = np.argsort(x_valid)
				x_valid = x_valid[sort_order]
				y_valid = y_valid[sort_order]

				unique_idx = np.concatenate(([True], np.diff(x_valid) > 0))
				x_valid = x_valid[unique_idx]
				y_valid = y_valid[unique_idx]

				if len(x_valid) < 2:
					continue

				cs = CubicSpline(x_valid, y_valid, extrapolate=True)
				df_result.loc[mask, col] = cs(x)

		return df_result

	if verbose:
		print(f"Interpolating weather data...")

	speed_trials_weather = interpolate_weather_data_by_dataset(speed_trials_weather)

	if verbose:
		print(f"Final shape: {speed_trials_weather.shape}")
		for col in weather_columns:
			missing = speed_trials_weather[col].isna().sum()
			pct = (missing / len(speed_trials_weather)) * 100
			print(f"  {col}: {missing} missing ({pct:.1f}%)")

	return speed_trials_weather


SPEED_TRIALS_WEATHER_CLEAN = clean_weather_data(verbose=False)