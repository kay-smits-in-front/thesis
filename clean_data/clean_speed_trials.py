"""Script to clean and preprocess the speed_trials dataset."""
import os

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from load_data.speed_trials import SPEED_TRIALS


def recombine_columns_correctly(speed_trials_raw, verbose=True):
    """Recombine columns that were split by Excel."""
    if verbose:
        print("Recombining columns correctly...")

    header_line = str(speed_trials_raw.iloc[0, 0])
    header_parts = header_line.split(';')

    if verbose:
        print(f"Total header parts: {len(header_parts)}")
        print(f"Header parts: {header_parts[:5]}...")

    all_rows = []
    for idx in range(2, len(speed_trials_raw)):
        datetime_val = speed_trials_raw.iloc[idx, 0]

        if pd.isna(datetime_val):
            continue

        processed_row = [datetime_val]

        col_idx = 1
        while col_idx < len(speed_trials_raw.columns) - 1:
            integer_part = speed_trials_raw.iloc[idx, col_idx]
            decimal_part = speed_trials_raw.iloc[idx, col_idx + 1]

            if pd.isna(integer_part) or pd.isna(decimal_part):
                combined = None
            else:
                try:
                    int_val = float(integer_part)
                    dec_val = abs(float(decimal_part))
                    if int_val < 0:
                        combined = int_val - dec_val / 100
                    else:
                        combined = int_val + dec_val / 100
                except:
                    combined = None

            processed_row.append(combined)
            col_idx += 2

        all_rows.append(processed_row)

    num_data_cols = len(all_rows[0]) - 1 if len(all_rows) > 0 else 0
    columns = [header_parts[0]] + header_parts[1:1+num_data_cols]

    speed_trials = pd.DataFrame(all_rows, columns=columns)

    if verbose:
        print(f"After recombining: {speed_trials.shape}")
        print(f"Columns: {list(speed_trials.columns)[:10]}")
        print(f"Sample row:\n{speed_trials.iloc[0]}")

    return speed_trials


def detect_dataset_boundaries(df, datetime_col, verbose=True):
    """Detect dataset boundaries based on datetime."""
    df['datetime_parsed'] = pd.to_datetime(df[datetime_col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    valid_count = df['datetime_parsed'].notna().sum()
    if verbose:
        print(f"\nValid datetimes: {valid_count} out of {len(df)}")

    if valid_count > 0:
        time_diff = df['datetime_parsed'].diff()
        backward_jumps = time_diff[time_diff < pd.Timedelta(0)]
        large_gaps = time_diff[time_diff > pd.Timedelta(seconds=10)]

        boundaries = sorted(set(backward_jumps.index.tolist() + large_gaps.index.tolist()))

        if verbose:
            print(f"Found {len(boundaries)} dataset boundaries")
    else:
        boundaries = []

    df['dataset_id'] = 0
    current_id = 0
    for boundary in boundaries:
        df.loc[boundary:, 'dataset_id'] = current_id + 1
        current_id += 1

    if verbose:
        print(f"Created {df['dataset_id'].nunique()} separate datasets")
        for dataset_id in df['dataset_id'].unique():
            count = (df['dataset_id'] == dataset_id).sum()
            print(f"  Dataset {dataset_id}: {count} rows")

    return df


def add_unit_conversions(df, verbose=True):
    """Add unit conversions for physics-informed neural network."""

    if verbose:
        print("\n" + "="*70)
        print("ADDING UNIT CONVERSIONS FOR PINN")
        print("="*70)

    df_converted = df.copy()

    if 'OPC_07_WATER_SPEED' in df.columns:
        df_converted['u_ms'] = df['OPC_07_WATER_SPEED'] * 0.514444
    if 'GPS_HDG_HEADING_ROT_S' in df.columns:
        df_converted['r_rads'] = df['GPS_HDG_HEADING_ROT_S'] * 0.0174533
    if 'OPC_40_PROP_RPM_FB' in df.columns:
        df_converted['nP_revs'] = df['OPC_40_PROP_RPM_FB'] / 60.0

    df_converted['v_ms'] = 0.0

    if verbose:
        print("Conversions applied based on available columns")

    return df_converted


def clean_speed_trials(verbose=True):
    """Clean speed trials data with correct column recombination."""
    speed_trials_raw = SPEED_TRIALS.copy()

    if verbose:
        print("Initial shape:", speed_trials_raw.shape)

    speed_trials = recombine_columns_correctly(speed_trials_raw, verbose=verbose)

    if verbose:
        print(f"Non-null counts per column:\n{speed_trials.count()}")

    datetime_col = speed_trials.columns[0]
    speed_trials = detect_dataset_boundaries(speed_trials, datetime_col, verbose=verbose)

    speed_trials = add_unit_conversions(speed_trials, verbose=verbose)

    if verbose:
        print(f"\nFinal shape: {speed_trials.shape}")

    return speed_trials

def save_to_excel(df, filename='speed_trials_clean.xlsx', verbose=True):
    """Save cleaned data to Excel file."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, filename)

    df.to_excel(output_path, index=False)

    if verbose:
        print(f"\nSaved cleaned data to: {output_path}")

    return output_path


SPEED_TRIALS_REGULAR = clean_speed_trials(verbose=True)
save_to_excel(SPEED_TRIALS_REGULAR)