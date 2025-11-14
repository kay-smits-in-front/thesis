"""Cleaning of weather data for analysis."""

from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
from load_data.speed_trials import SPEED_TRIALS
from load_data.weather_data import WEATHER_DATA


def clean_weather_data(verbose=True):
	speed_trials_weather = SPEED_TRIALS.copy()

	header_row = speed_trials_weather.iloc[0, 0].split(';')
	header_row = header_row[:len(speed_trials_weather.columns)]
	speed_trials_weather = speed_trials_weather.iloc[2:].reset_index(drop=True)
	speed_trials_weather.columns = header_row

	def parse_datetime(val):
		if pd.isna(val) or not isinstance(val, str):
			return None
		try:
			parts = val.split(';')
			if len(parts) == 2:
				datetime_str = f"{parts[0]} {parts[1]}"
				return pd.to_datetime(datetime_str, format='%m/%d/%Y %H:%M:%S')
		except:
			return None
		return None

	datetime_col = speed_trials_weather.iloc[:, 0].apply(parse_datetime)
	speed_trials_weather['datetime_parsed'] = pd.to_datetime(datetime_col, utc=True)

	time_diff = speed_trials_weather['datetime_parsed'].diff()
	backward_jumps = time_diff[time_diff < pd.Timedelta(0)]

	if verbose and len(backward_jumps) > 0:
		print(f"\nFound {len(backward_jumps)} backward time jumps (dataset boundaries):")
		for idx in backward_jumps.index[:10]:
			print(f"Row {idx}: {speed_trials_weather['datetime_parsed'].iloc[idx-1]} -> {speed_trials_weather['datetime_parsed'].iloc[idx]}")

	speed_trials_weather = speed_trials_weather.sort_values('datetime_parsed').reset_index(drop=True)
	datetime_col = speed_trials_weather['datetime_parsed']

	speed_trials_weather['elapsed_seconds'] = (datetime_col - datetime_col.min()).dt.total_seconds()
	speed_trials_weather['hour'] = datetime_col.dt.hour
	speed_trials_weather['minute'] = datetime_col.dt.minute
	speed_trials_weather['second'] = datetime_col.dt.second

	weather_data = WEATHER_DATA.copy()
	weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'], utc=True)
	weather_data = weather_data.sort_values('timestamp').reset_index(drop=True)
	weather_numeric_cols = ['mean_wave_direction', 'mean_wave_period', 'significant_wave_height',
	                        'wind_u_component_10m', 'wind_v_component_10m', 'air_density',
	                        'wind_speed_10m', 'wind_direction_10m']
	for col in weather_numeric_cols:
		weather_data[col] = pd.to_numeric(weather_data[col], errors='coerce')

	weather_data = weather_data.sort_values('timestamp').reset_index(drop=True)

	weather_columns = [col for col in weather_data.columns if col not in ['latitude', 'longitude', 'timestamp']]
	for col in weather_columns:
		speed_trials_weather[col] = None

	for idx, row in speed_trials_weather.iterrows():
		st_time = row['datetime_parsed']
		time_diff = (weather_data['timestamp'] - st_time).abs()
		match_idx = time_diff.idxmin()
		if time_diff[match_idx] <= pd.Timedelta(minutes=2):
			for col in weather_columns:
				speed_trials_weather.at[idx, col] = weather_data.at[match_idx, col]

	for col in weather_columns:
		speed_trials_weather[col] = pd.to_numeric(speed_trials_weather[col], errors='coerce')

	speed_trials_weather = speed_trials_weather.drop(columns=[speed_trials_weather.columns[0],
	                                                          speed_trials_weather.columns[1],
	                                                          'datetime_parsed'])

	def convert_value(val):
		if pd.isna(val):
			return None
		if isinstance(val, str):
			val = val.strip()
			if val == '':
				return None
			if ';' in val:
				parts = val.split(';')
				try:
					integer_part = float(parts[0])
					decimal_part = parts[1].strip()
					if decimal_part == '':
						return integer_part
					if decimal_part.startswith('-'):
						decimal_value = -float(decimal_part[1:]) / 10
					else:
						decimal_value = float(decimal_part) / 10
					return integer_part + decimal_value
				except:
					return None
		return val

	original_speed_columns = [col for col in speed_trials_weather.columns
	                          if col not in weather_columns + ['elapsed_seconds', 'hour', 'minute', 'second']]
	for col in original_speed_columns:
		speed_trials_weather[col] = speed_trials_weather[col].map(convert_value)
		speed_trials_weather[col] = pd.to_numeric(speed_trials_weather[col], errors='coerce')

	threshold = 0.5 * len(original_speed_columns)
	missing_counts = speed_trials_weather[original_speed_columns].isna().sum(axis=1)
	speed_trials_weather = speed_trials_weather[missing_counts <= threshold].reset_index(drop=True)

	if verbose:
		print(f"\nFinal shape: {speed_trials_weather.shape}")

	def interpolate_weather_data(df, time_col='elapsed_seconds'):
		"""Cubic spline interpolation for weather columns."""
		weather_cols = ['mean_wave_direction', 'mean_wave_period', 'significant_wave_height',
		                'wind_u_component_10m', 'wind_v_component_10m', 'air_density',
		                'wind_speed_10m', 'wind_direction_10m']

		df_result = df.copy()
		x = df_result[time_col].values

		for col in weather_cols:
			if col not in df_result.columns:
				continue

			y = df_result[col].values
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

			cs = CubicSpline(x_valid, y_valid, extrapolate='linear')
			df_result[col] = cs(x)

		return df_result

	speed_trials_weather = interpolate_weather_data(speed_trials_weather)

	return speed_trials_weather


SPEED_TRIALS_WEATHER_CLEAN = clean_weather_data(verbose=False)