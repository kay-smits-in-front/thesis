"""Cleaning of weather data for analysis."""

from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
from load_data.speed_trials import SPEED_TRIALS
from load_data.weather_data import WEATHER_DATA


SPEED_TRIALS_WEATHER_CLEAN = SPEED_TRIALS


header_row = SPEED_TRIALS_WEATHER_CLEAN.iloc[0, 0].split(';')
header_row = header_row[:len(SPEED_TRIALS_WEATHER_CLEAN.columns)]
SPEED_TRIALS_WEATHER_CLEAN = SPEED_TRIALS_WEATHER_CLEAN.iloc[2:].reset_index(drop=True)
SPEED_TRIALS_WEATHER_CLEAN.columns = header_row

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


print("Parsing datetime column...")
datetime_col = SPEED_TRIALS_WEATHER_CLEAN.iloc[:, 0].apply(parse_datetime)
SPEED_TRIALS_WEATHER_CLEAN['datetime_parsed'] = pd.to_datetime(datetime_col, utc=True)

print("Checking for backward jumps in time...")
time_diff = SPEED_TRIALS_WEATHER_CLEAN['datetime_parsed'].diff()
backward_jumps = time_diff[time_diff < pd.Timedelta(0)]
if len(backward_jumps) > 0:
	print(f"\nFound {len(backward_jumps)} backward time jumps (dataset boundaries):")
	for idx in backward_jumps.index[:10]:
		print(f"Row {idx}: {SPEED_TRIALS_WEATHER_CLEAN['datetime_parsed'].iloc[idx-1]} -> {SPEED_TRIALS_WEATHER_CLEAN['datetime_parsed'].iloc[idx]}")
else:
	print("No backward jumps found")

print("\nSorting entire dataset by datetime...")
SPEED_TRIALS_WEATHER_CLEAN = SPEED_TRIALS_WEATHER_CLEAN.sort_values('datetime_parsed').reset_index(drop=True)
datetime_col = SPEED_TRIALS_WEATHER_CLEAN['datetime_parsed']

print("Creating time features...")
SPEED_TRIALS_WEATHER_CLEAN['elapsed_seconds'] = (datetime_col - datetime_col.min()).dt.total_seconds()
SPEED_TRIALS_WEATHER_CLEAN['hour'] = datetime_col.dt.hour
SPEED_TRIALS_WEATHER_CLEAN['minute'] = datetime_col.dt.minute
SPEED_TRIALS_WEATHER_CLEAN['second'] = datetime_col.dt.second

print("Loading weather data...")
weather_data = WEATHER_DATA
weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'], utc=True)
weather_data = weather_data.sort_values('timestamp').reset_index(drop=True)
weather_numeric_cols = ['mean_wave_direction', 'mean_wave_period', 'significant_wave_height', 'wind_u_component_10m', 'wind_v_component_10m', 'air_density', 'wind_speed_10m', 'wind_direction_10m']
for col in weather_numeric_cols:
	weather_data[col] = pd.to_numeric(weather_data[col], errors='coerce')

weather_data = weather_data.sort_values('timestamp').reset_index(drop=True)

print("Merging weather data within 2 minute tolerance...")
weather_columns = [col for col in weather_data.columns if col not in ['latitude', 'longitude', 'timestamp']]
for col in weather_columns:
	SPEED_TRIALS_WEATHER_CLEAN[col] = None

for idx, row in SPEED_TRIALS_WEATHER_CLEAN.iterrows():
	st_time = row['datetime_parsed']
	time_diff = (weather_data['timestamp'] - st_time).abs()
	match_idx = time_diff.idxmin()
	if time_diff[match_idx] <= pd.Timedelta(minutes=2):
		for col in weather_columns:
			SPEED_TRIALS_WEATHER_CLEAN.at[idx, col] = weather_data.at[match_idx, col]

# Convert weather columns to numeric
for col in weather_columns:
	SPEED_TRIALS_WEATHER_CLEAN[col] = pd.to_numeric(SPEED_TRIALS_WEATHER_CLEAN[col], errors='coerce')

print("Dropping original datetime columns...")
SPEED_TRIALS_WEATHER_CLEAN = SPEED_TRIALS_WEATHER_CLEAN.drop(columns=[SPEED_TRIALS_WEATHER_CLEAN.columns[0], SPEED_TRIALS_WEATHER_CLEAN.columns[1], 'datetime_parsed'])

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

print("Converting remaining columns...")
original_speed_columns = [col for col in SPEED_TRIALS_WEATHER_CLEAN.columns if col not in weather_columns + ['elapsed_seconds', 'hour', 'minute', 'second']]
for col in original_speed_columns:
	SPEED_TRIALS_WEATHER_CLEAN[col] = SPEED_TRIALS_WEATHER_CLEAN[col].map(convert_value)
	SPEED_TRIALS_WEATHER_CLEAN[col] = pd.to_numeric(SPEED_TRIALS_WEATHER_CLEAN[col], errors='coerce')

print("Removing rows with >50% missing data in speed_trials columns...")
threshold = 0.5 * len(original_speed_columns)
missing_counts = SPEED_TRIALS_WEATHER_CLEAN[original_speed_columns].isna().sum(axis=1)
SPEED_TRIALS_WEATHER_CLEAN = SPEED_TRIALS_WEATHER_CLEAN[missing_counts <= threshold].reset_index(drop=True)

print(f"\nFinal shape: {SPEED_TRIALS_WEATHER_CLEAN.shape}")
print("\nVerifying chronological order (should be True):")
print((SPEED_TRIALS_WEATHER_CLEAN['elapsed_seconds'].diff().dropna() >= 0).all())
print("\nSample of elapsed_seconds from different parts:")
print(SPEED_TRIALS_WEATHER_CLEAN['elapsed_seconds'].iloc[[0, 5000, 10000, 15000, 20000, -1]])
print("\nWeather data sample:")
print(SPEED_TRIALS_WEATHER_CLEAN[weather_columns].head(10))

def interpolate_weather_data(df, time_col='elapsed_seconds'):
	"""Cubic spline interpolatie voor alle weerkolommen."""
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

SPEED_TRIALS_WEATHER_CLEAN = interpolate_weather_data(SPEED_TRIALS_WEATHER_CLEAN)
print(SPEED_TRIALS_WEATHER_CLEAN.head())
