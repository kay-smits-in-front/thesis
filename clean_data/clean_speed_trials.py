"""Script to clean and preprocess the speed_trials dataset."""
import pandas as pd
import sys
import io
from load_data.speed_trials import SPEED_TRIALS


def clean_speed_trials(verbose=True):
    speed_trials = SPEED_TRIALS.copy()

    if verbose:
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

    datetime_col = speed_trials.iloc[:, 0].apply(parse_datetime)
    speed_trials['datetime_parsed'] = datetime_col

    time_diff = speed_trials['datetime_parsed'].diff()
    backward_jumps = time_diff[time_diff < pd.Timedelta(0)]

    if verbose and len(backward_jumps) > 0:
        print(f"\nFound {len(backward_jumps)} backward time jumps (dataset boundaries):")
        for idx in backward_jumps.index[:10]:
            print(f"Row {idx}: {speed_trials['datetime_parsed'].iloc[idx-1]} -> {speed_trials['datetime_parsed'].iloc[idx]}")

    speed_trials = speed_trials.sort_values('datetime_parsed').reset_index(drop=True)
    datetime_col = speed_trials['datetime_parsed']

    speed_trials['elapsed_seconds'] = (datetime_col - datetime_col.min()).dt.total_seconds()
    speed_trials['hour'] = datetime_col.dt.hour
    speed_trials['minute'] = datetime_col.dt.minute
    speed_trials['second'] = datetime_col.dt.second

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

    for col in speed_trials.columns[:-4]:
        speed_trials[col] = speed_trials[col].map(convert_value)
        speed_trials[col] = pd.to_numeric(speed_trials[col], errors='coerce')

    threshold = 0.5 * len(speed_trials.columns)
    speed_trials_clean = speed_trials.dropna(thresh=threshold).reset_index(drop=True)

    if verbose:
        print(f"\nFinal shape: {speed_trials_clean.shape}")

    return speed_trials_clean


SPEED_TRIALS_REGULAR = clean_speed_trials(verbose=False)