"""Data preprocessing: Add features, filter OPC_04/05, remove anomalies."""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_engineered_features(data):
	"""Add TRIM and DRAFT_AVG features."""
	data = data.copy()

	# Find all DRAFT columns
	draft_cols = [col for col in data.columns if 'DRAFT' in col.upper()]
	print(f"  DRAFT columns found: {draft_cols}")

	# Add DRAFT_AVG
	if len(draft_cols) > 0:
		data['DRAFT_AVG'] = data[draft_cols].mean(axis=1)
		print(f"  Added DRAFT_AVG column")

	# Add TRIM (Forward - Aft)
	if 'OPC_14_VES_DRAFT_FWD' in data.columns and 'OPC_15_VES_DRAFT_AFT' in data.columns:
		data['TRIM'] = data['OPC_14_VES_DRAFT_FWD'] - data['OPC_15_VES_DRAFT_AFT']
		print(f"  Added TRIM column (FWD - AFT)")
		print(f"    TRIM mean: {data['TRIM'].mean():.4f}")
		print(f"    TRIM min: {data['TRIM'].min():.4f}")
		print(f"    TRIM max: {data['TRIM'].max():.4f}")

	return data


def apply_opc_filtering(data):
	"""Apply OPC_4 and OPC_5 filtering."""
	opc4_cols = [col for col in data.columns if col.startswith('OPC_04')]
	opc5_cols = [col for col in data.columns if col.startswith('OPC_05')]

	if not opc4_cols and not opc5_cols:
		return data

	print(f"\nStep 2: Applying OPC_4/5 filtering (values must be in [-3, 3])...")
	original_rows = len(data)
	mask = pd.Series(True, index=data.index)

	for col in opc4_cols + opc5_cols:
		col_mask = (data[col] >= -3) & (data[col] <= 3)
		mask = mask & col_mask

	data_filtered = data[mask].reset_index(drop=True)
	removed = original_rows - len(data_filtered)
	print(f"  Removed {removed} rows ({removed/original_rows*100:.2f}%)")

	return data_filtered


def remove_anomalies(data, target_col, contamination=0.05):
	"""Remove overlapping anomalies detected by both IF and LOF."""
	print(f"\nStep 3: Detecting anomalies...")

	# Exclude columns
	exclude_cols = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	                "elapsed_seconds", "hour", "minute", "second", "dataset_id",
	                "GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	                "Unnamed: 23", "GPS_HDG_HEADING_ROT_S", "OPC_07_WATER_SPEED", "OPC_40_PROP_RPM_FB",
	                target_col]

	numeric_data = data.select_dtypes(include=[np.number])
	feature_data = numeric_data.drop(columns=exclude_cols, errors='ignore')
	data_clean = feature_data.dropna()

	print(f"  Using {len(feature_data.columns)} features")
	print(f"  Analyzing {len(data_clean)} rows")

	# Scale data for anomaly detection
	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(data_clean)

	# Isolation Forest
	iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
	iso_predictions = iso_forest.fit_predict(data_scaled)
	iso_anomalies = iso_predictions == -1

	# LOF
	lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
	lof_predictions = lof.fit_predict(data_scaled)
	lof_anomalies = lof_predictions == -1

	# Find overlap
	both_anomalies = iso_anomalies & lof_anomalies

	print(f"  Isolation Forest: {iso_anomalies.sum()} anomalies")
	print(f"  LOF: {lof_anomalies.sum()} anomalies")
	print(f"  Both methods: {both_anomalies.sum()} anomalies")

	# Create mask for original data
	anomaly_mask = pd.Series(False, index=data.index)
	anomaly_mask.loc[data_clean.index[both_anomalies]] = True

	# Remove anomalies
	data_no_anomalies = data[~anomaly_mask].reset_index(drop=True)
	print(f"  Removed {anomaly_mask.sum()} overlapping anomalies")

	return data_no_anomalies


def main():
	"""Prepare data for baseline models."""
	from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
	from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN

	# Define target column
	TARGET_COL = 'OPC_12_CPP_ENGINE_POWER'

	print("\n" + "="*70)
	print("DATA PREPROCESSING PIPELINE")
	print("="*70)
	print(f"Target column: {TARGET_COL}")

	# ========================================================================
	# PROCESS SPEED_TRIALS_REGULAR
	# ========================================================================
	print("\n" + "="*70)
	print("PROCESSING: SPEED_TRIALS_REGULAR")
	print("="*70)

	print(f"\nOriginal data: {SPEED_TRIALS_REGULAR.shape}")

	# Check if target exists
	if TARGET_COL not in SPEED_TRIALS_REGULAR.columns:
		print(f"\nERROR: Target column '{TARGET_COL}' not found in data!")
		print(f"Available columns: {list(SPEED_TRIALS_REGULAR.columns)}")
		return

	# Step 1: Add engineered features BEFORE filtering
	print("\nStep 1: Adding engineered features...")
	data_st = add_engineered_features(SPEED_TRIALS_REGULAR)
	print(f"  Shape after adding features: {data_st.shape}")

	# Check for NaN values after adding features
	nan_counts = data_st.isna().sum()
	if nan_counts.sum() > 0:
		print(f"  NaN values found in {(nan_counts > 0).sum()} columns")
		print(f"  Total rows with any NaN: {data_st.isna().any(axis=1).sum()}")
		print(f"  Dropping rows with NaN values...")
		data_st = data_st.dropna().reset_index(drop=True)
		print(f"  Shape after dropping NaN: {data_st.shape}")

	# Step 2: Apply OPC filtering
	data_st = apply_opc_filtering(data_st)
	print(f"  Shape after OPC filtering: {data_st.shape}")

	# Step 3: Remove anomalies
	data_st_clean = remove_anomalies(data_st, TARGET_COL, contamination=0.05)
	print(f"  Shape after removing anomalies: {data_st_clean.shape}")

	# Step 4: Filter out low-power startup data
	print(f"\nStep 4: Filtering operational data (power > 1000 kW)...")
	power_threshold = 1000
	operational_mask = data_st_clean[TARGET_COL] > power_threshold
	removed_startup = (~operational_mask).sum()
	print(f"  Removed {removed_startup} low-power rows ({removed_startup/len(data_st_clean)*100:.2f}%)")

	data_st_clean = data_st_clean[operational_mask].reset_index(drop=True)
	print(f"  Shape after power filtering: {data_st_clean.shape}")
	print(f"  Final power range: {data_st_clean[TARGET_COL].min():.2f} - {data_st_clean[TARGET_COL].max():.2f} kW")

	# Save FINAL cleaned data - THIS IS THE SINGLE SOURCE OF TRUTH
	output_path_st = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
	data_st_clean.to_csv(output_path_st, index=False)
	print(f"\n✓ SAVED FINAL: {output_path_st}")

	# ========================================================================
	# PROCESS SPEED_TRIALS_WEATHER_CLEAN
	# ========================================================================
	print("\n" + "="*70)
	print("PROCESSING: SPEED_TRIALS_WEATHER_CLEAN")
	print("="*70)

	print(f"\nOriginal data: {SPEED_TRIALS_WEATHER_CLEAN.shape}")

	# Check if target exists
	if TARGET_COL not in SPEED_TRIALS_WEATHER_CLEAN.columns:
		print(f"\nERROR: Target column '{TARGET_COL}' not found in weather data!")
		return

	# Step 1: Add engineered features BEFORE filtering
	print("\nStep 1: Adding engineered features...")
	data_weather = add_engineered_features(SPEED_TRIALS_WEATHER_CLEAN)
	print(f"  Shape after adding features: {data_weather.shape}")

	# Check for NaN values after adding features
	nan_counts = data_weather.isna().sum()
	if nan_counts.sum() > 0:
		print(f"  NaN values found in {(nan_counts > 0).sum()} columns")
		print(f"  Total rows with any NaN: {data_weather.isna().any(axis=1).sum()}")
		print(f"  Dropping rows with NaN values...")
		data_weather = data_weather.dropna().reset_index(drop=True)
		print(f"  Shape after dropping NaN: {data_weather.shape}")

	# Step 2: Apply OPC filtering
	data_weather = apply_opc_filtering(data_weather)
	print(f"  Shape after OPC filtering: {data_weather.shape}")

	# Step 3: Remove anomalies
	data_weather_clean = remove_anomalies(data_weather, TARGET_COL, contamination=0.05)
	print(f"  Shape after removing anomalies: {data_weather_clean.shape}")

	# Step 4: Filter out low-power startup data
	print(f"\nStep 4: Filtering operational data (power > 1000 kW)...")
	power_threshold = 1000
	operational_mask = data_weather_clean[TARGET_COL] > power_threshold
	removed_startup = (~operational_mask).sum()
	print(f"  Removed {removed_startup} low-power rows ({removed_startup/len(data_weather_clean)*100:.2f}%)")

	data_weather_clean = data_weather_clean[operational_mask].reset_index(drop=True)
	print(f"  Shape after power filtering: {data_weather_clean.shape}")
	print(f"  Final power range: {data_weather_clean[TARGET_COL].min():.2f} - {data_weather_clean[TARGET_COL].max():.2f} kW")

	# Save FINAL cleaned data - THIS IS THE SINGLE SOURCE OF TRUTH
	output_path_weather = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")
	data_weather_clean.to_csv(output_path_weather, index=False)
	print(f"\n✓ SAVED FINAL: {output_path_weather}")

	# ========================================================================
	# SUMMARY
	# ========================================================================
	print(f"\n{'='*70}")
	print("PREPROCESSING COMPLETE")
	print(f"{'='*70}")
	print(f"\nFINAL datasets created (single source of truth for all models):")
	print(f"  - SPEED_TRIALS_REGULAR_FINAL.csv ({len(data_st_clean)} rows, {len(data_st_clean.columns)} columns)")
	print(f"  - SPEED_TRIALS_WEATHER_FINAL.csv ({len(data_weather_clean)} rows, {len(data_weather_clean.columns)} columns)")
	print(f"\nPreprocessing pipeline:")
	print(f"  1. Added TRIM and DRAFT_AVG features")
	print(f"  2. Dropped rows with NaN values")
	print(f"  3. Applied OPC_04/05 filtering (values in [-3, 3])")
	print(f"  4. Removed anomalies (contamination=0.05)")
	print(f"  5. Filtered operational data (engine power > 1000 kW)")
	print(f"\nTo create visualizations, run: python3 create_visualizations.py")
	print(f"\nAll models should now load from these FINAL files!")


if __name__ == "__main__":
	main()