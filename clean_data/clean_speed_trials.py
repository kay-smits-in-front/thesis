"""Script to clean and preprocess the speed_trials dataset."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from load_data.speed_trials import SPEED_TRIALS

speed_trials = SPEED_TRIALS


print("Initial shape:", speed_trials.shape)
header_row = speed_trials.iloc[0, 0].split(';')
header_row = header_row[:len(speed_trials.columns)]
speed_trials = speed_trials.iloc[2:].reset_index(drop=True)
speed_trials.columns = header_row

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
datetime_col = speed_trials.iloc[:, 0].apply(parse_datetime)
speed_trials['datetime_parsed'] = datetime_col

print("Checking for backward jumps in time...")
time_diff = speed_trials['datetime_parsed'].diff()
backward_jumps = time_diff[time_diff < pd.Timedelta(0)]

if len(backward_jumps) > 0:
    print(f"\nFound {len(backward_jumps)} backward time jumps (dataset boundaries):")
    for idx in backward_jumps.index[:10]:
        print(f"Row {idx}: {speed_trials['datetime_parsed'].iloc[idx-1]} -> {speed_trials['datetime_parsed'].iloc[idx]}")
else:
    print("No backward jumps found")

print("\nSorting entire dataset by datetime...")
speed_trials = speed_trials.sort_values('datetime_parsed').reset_index(drop=True)
datetime_col = speed_trials['datetime_parsed']

print("Creating time features...")
speed_trials['elapsed_seconds'] = (datetime_col - datetime_col.min()).dt.total_seconds()
speed_trials['hour'] = datetime_col.dt.hour
speed_trials['minute'] = datetime_col.dt.minute
speed_trials['second'] = datetime_col.dt.second

print("Dropping original datetime columns...")
speed_trials = speed_trials.drop(columns=[speed_trials.columns[0], speed_trials.columns[1], 'datetime_parsed'])

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
for col in speed_trials.columns[:-4]:
    speed_trials[col] = speed_trials[col].map(convert_value)
    speed_trials[col] = pd.to_numeric(speed_trials[col], errors='coerce')

print("Removing rows with >50% missing data...")
threshold = 0.5 * len(speed_trials.columns)
SPEED_TRIALS_REGULAR = speed_trials.dropna(thresh=threshold).reset_index(drop=True)

print(f"\nFinal shape: {SPEED_TRIALS_REGULAR.shape}")
print("\nVerifying chronological order (should be True):")
print((SPEED_TRIALS_REGULAR['elapsed_seconds'].diff().dropna() >= 0).all())
print("\nSample of elapsed_seconds from different parts:")
print(SPEED_TRIALS_REGULAR['elapsed_seconds'].iloc[[0, 5000, 10000, 15000, 20000, -1]])
