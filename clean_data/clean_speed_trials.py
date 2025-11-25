"""Script to clean and preprocess the speed_trials dataset."""
import os

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from load_data.speed_trials import SPEED_TRIALS


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

        all_rows.append(processed_row)

def interpolate_missing_column(df, col_name, time_col='elapsed_seconds', verbose=True):
    """Interpolate missing values in a column using cubic spline per dataset."""

    if col_name not in df.columns:
        if verbose:
            print(f"Column {col_name} not found, skipping interpolation")
        return df

    df_result = df.copy()
    missing_count = df_result[col_name].isna().sum()

    if missing_count == 0:
        if verbose:
            print(f"No missing values in {col_name}, skipping interpolation")
        return df_result

    if verbose:
        print(f"\nInterpolating {missing_count} missing values in {col_name}")

    for dataset_id in df_result['dataset_id'].unique():
        mask = df_result['dataset_id'] == dataset_id
        dataset_subset = df_result[mask]

        x = dataset_subset[time_col].values
        y = dataset_subset[col_name].values

        valid_idx = ~np.isnan(y)

        if valid_idx.sum() < 2:
            if verbose:
                print(f"  Dataset {dataset_id}: too few valid values, skipping")
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
        df_result.loc[mask, col_name] = cs(x)

        if verbose:
            interpolated = mask.sum() - valid_idx[mask].sum()
            print(f"  Dataset {dataset_id}: interpolated {interpolated} values")

    return df_result


def add_unit_conversions(df, verbose=True):
    """Add unit conversions for physics-informed neural network."""

    if verbose:
        print("\n" + "="*70)
        print("ADDING UNIT CONVERSIONS FOR PINN")
        print("="*70)

    df_converted = df.copy()

    df_converted['u_ms'] = df['OPC_07_WATER_SPEED'] * 0.514444
    df_converted['v_ms'] = 0.0
    df_converted['r_rads'] = df['GPS_HDG_HEADING_ROT_S'] * 0.0174533
    df_converted['nP_revs'] = df['OPC_40_PROP_RPM_FB'] / 60.0

    df_converted['v_ms'] = 0.0

    if verbose:
        print("\nConversions applied:")
        print("✓ OPC_07_WATER_SPEED (kts) -> u_ms (m/s)")
        print("✓ v_ms = 0.0 (m/s) - sway velocity not available")
        print("✓ GPS_HDG_HEADING_ROT_S (deg/s) -> r_rads (rad/s)")
        print("✓ OPC_40_PROP_RPM_FB (RPM) -> nP_revs (rev/s)")

        print("\nConverted value ranges:")
        print(f"u (m/s): min={df_converted['u_ms'].min():.2f}, max={df_converted['u_ms'].max():.2f}, mean={df_converted['u_ms'].mean():.2f}")
        print(f"v (m/s): {df_converted['v_ms'].iloc[0]:.2f} (constant)")
        print(f"r (rad/s): min={df_converted['r_rads'].min():.6f}, max={df_converted['r_rads'].max():.6f}, mean={df_converted['r_rads'].mean():.6f}")
        print(f"nP (rev/s): min={df_converted['nP_revs'].min():.2f}, max={df_converted['nP_revs'].max():.2f}, mean={df_converted['nP_revs'].mean():.2f}")

    return df_converted


def preprocess_with_dataset_boundaries(speed_trials, verbose=True):
    """Preprocess data and identify dataset boundaries."""

    datetime_col = speed_trials.iloc[:, 0].apply(parse_datetime)
    speed_trials['datetime_parsed'] = datetime_col

    speed_trials = speed_trials.sort_values('datetime_parsed').reset_index(drop=True)
    datetime_col = speed_trials['datetime_parsed']

    time_diff = datetime_col.diff()

    backward_jumps = time_diff[time_diff < pd.Timedelta(0)]
    large_gaps = time_diff[time_diff > pd.Timedelta(seconds=5)]

    boundaries = sorted(set(backward_jumps.index.tolist() + large_gaps.index.tolist()))

    if verbose:
        print(f"\nFound {len(boundaries)} dataset boundaries:")
        print(f"  - Backward jumps: {len(backward_jumps)}")
        print(f"  - Large gaps (>5 sec): {len(large_gaps)}")

    speed_trials['dataset_id'] = 0
    current_id = 0
    for boundary in boundaries:
        speed_trials.loc[boundary:, 'dataset_id'] = current_id + 1
        current_id += 1

    if verbose:
        print(f"\nCreated {speed_trials['dataset_id'].nunique()} separate datasets")
        for dataset_id in speed_trials['dataset_id'].unique():
            count = (speed_trials['dataset_id'] == dataset_id).sum()
            print(f"  Dataset {dataset_id}: {count} rows")

    speed_trials['elapsed_seconds'] = (datetime_col - datetime_col.min()).dt.total_seconds()
    speed_trials['hour'] = datetime_col.dt.hour
    speed_trials['minute'] = datetime_col.dt.minute
    speed_trials['second'] = datetime_col.dt.second

    speed_trials = speed_trials.drop(columns=[speed_trials.columns[0], speed_trials.columns[1], 'datetime_parsed'])

    return speed_trials


def create_lag_features_with_boundaries(df, target_col, n_lags=60, forecast_horizon=10):
    """Create lag features respecting dataset boundaries."""

    drop_cols = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED"]
    cols_to_lag = [col for col in df.columns if col not in drop_cols + ['dataset_id'] and col != target_col]

    result_dfs = []

    for dataset_id in df['dataset_id'].unique():
        dataset_mask = df['dataset_id'] == dataset_id
        df_subset = df[dataset_mask].copy()

        lag_dfs = [df_subset.copy()]

        for col in cols_to_lag:
            for lag in range(1, n_lags + 1):
                lag_dfs.append(df_subset[col].shift(lag).rename(f'{col}_lag_{lag}'))

        df_lagged = pd.concat(lag_dfs, axis=1)

        target_shifted = df_subset[target_col].shift(-forecast_horizon).rename(f'{target_col}_target')
        df_lagged = pd.concat([df_lagged, target_shifted], axis=1)

        result_dfs.append(df_lagged)

    df_combined = pd.concat(result_dfs, axis=0).reset_index(drop=True)
    df_combined = df_combined.dropna()

    X = df_combined.drop(columns=drop_cols + [target_col, f'{target_col}_target', 'dataset_id'], errors='ignore')
    y = df_combined[f'{target_col}_target']

    return X, y


def clean_speed_trials(verbose=True):
    speed_trials = SPEED_TRIALS.copy()

    if verbose:
        print("Initial shape:", speed_trials.shape)

    header_row = speed_trials.iloc[0, 0].split(';')
    header_row = header_row[:len(speed_trials.columns)]
    speed_trials = speed_trials.iloc[2:].reset_index(drop=True)
    speed_trials.columns = header_row

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

    for col in speed_trials.columns[2:]:
        speed_trials[col] = speed_trials[col].map(convert_value)
        speed_trials[col] = pd.to_numeric(speed_trials[col], errors='coerce')

    threshold = 0.5 * len(speed_trials.columns)
    speed_trials_clean = speed_trials.dropna(thresh=threshold).reset_index(drop=True)

    speed_trials_clean = preprocess_with_dataset_boundaries(speed_trials_clean, verbose=verbose)

    # Interpolate last column if it has missing values
    last_col = speed_trials_clean.columns[-1]
    speed_trials_clean = interpolate_missing_column(speed_trials_clean, last_col, verbose=verbose)

    # Add unit conversions for PINN
    speed_trials_clean = add_unit_conversions(speed_trials_clean, verbose=verbose)

    if verbose:
        print(f"\nFinal shape: {speed_trials_clean.shape}")

    return speed_trials_clean

SPEED_TRIALS_REGULAR = clean_speed_trials(verbose=False)
print(SPEED_TRIALS_REGULAR.shape)