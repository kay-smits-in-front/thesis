"""Create visualizations from preprocessed data."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_target_over_time(data, target_col, suffix=''):
	"""Plot target variable over time."""
	print(f"  Plotting target over time...")

	if target_col not in data.columns:
		print(f"    ERROR: Target column '{target_col}' not found")
		return

	target_data = data[target_col].dropna()

	plt.figure(figsize=(16, 6))
	plt.plot(target_data.index, target_data.values, linewidth=0.5, alpha=0.7, color='steelblue')
	plt.xlabel('Index')
	plt.ylabel(target_col)
	plt.title(f'{target_col} Over Time')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	filename = f'target_over_time{suffix}.png'
	plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
	plt.close()

	print(f"    Saved: {filename}")


def plot_feature_importance(data, target_col, suffix=''):
	"""Plot feature importance based on correlation with target."""
	print(f"  Calculating feature importance...")

	if target_col not in data.columns:
		print(f"    ERROR: Target column '{target_col}' not found")
		return

	exclude_cols = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	                "elapsed_seconds", "hour", "minute", "second", "dataset_id",
	                "GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	                "OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"]

	numeric_data = data.select_dtypes(include=[np.number])
	feature_data = numeric_data.drop(columns=exclude_cols, errors='ignore')

	correlations = feature_data.corrwith(feature_data[target_col]).abs().drop(target_col).sort_values(ascending=False)
	top_10 = correlations.head(10)

	print(f"    Top 10 features:")
	for feat, corr in top_10.items():
		print(f"      {feat}: {corr:.3f}")

	plt.figure(figsize=(12, 8))
	plt.barh(range(len(top_10)), top_10.values, color='steelblue')
	plt.yticks(range(len(top_10)), top_10.index)
	plt.xlabel('Absolute Correlation with Target')
	plt.title('Feature Importance (Top 10)')
	plt.gca().invert_yaxis()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	filename = f'feature_importance{suffix}.png'
	plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
	plt.close()

	print(f"    Saved: {filename}")


def plot_correlation_matrix(data, target_col, suffix=''):
	"""Plot correlation matrix for top features."""
	print(f"  Creating correlation matrix...")

	if target_col not in data.columns:
		print(f"    ERROR: Target column '{target_col}' not found")
		return

	exclude_cols = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	                "elapsed_seconds", "hour", "minute", "second", "dataset_id",
	                "GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	                "OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"]

	numeric_data = data.select_dtypes(include=[np.number])
	feature_data = numeric_data.drop(columns=exclude_cols, errors='ignore')
	data_clean = feature_data.dropna()

	correlations = data_clean.corrwith(data_clean[target_col]).abs().drop(target_col).sort_values(ascending=False)
	top_features = correlations.head(15).index.tolist()

	selected_cols = top_features + [target_col]
	corr_matrix = data_clean[selected_cols].corr()

	fig, axes = plt.subplots(1, 2, figsize=(20, 8))

	# Full correlation matrix
	ax = axes[0]
	sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
	            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
	ax.set_title('Correlation Matrix: Top 20 Features + Target')

	# Target correlation
	ax = axes[1]
	target_corr = corr_matrix[target_col].drop(target_col).sort_values()
	colors = ['red' if x < 0 else 'steelblue' for x in target_corr.values]
	ax.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
	ax.set_yticks(range(len(target_corr)))
	ax.set_yticklabels(target_corr.index)
	ax.set_xlabel('Correlation with Target')
	ax.set_title(f'Correlation with {target_col}')
	ax.axvline(0, color='black', linewidth=0.5)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	filename = f'correlation_matrix{suffix}.png'
	plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
	plt.close()

	print(f"    Saved: {filename}")


def plot_weather_correlations(data, target_col):
	"""Plot correlation of weather features with target."""
	print(f"  Creating weather correlations...")

	weather_cols = ['mean_wave_direction', 'mean_wave_period', 'significant_wave_height',
	                'wind_u_component_10m', 'wind_v_component_10m', 'air_density',
	                'wind_speed_10m', 'wind_direction_10m']

	available_weather = [col for col in weather_cols if col in data.columns]

	if not available_weather:
		print("    No weather features found in dataset")
		return

	if target_col not in data.columns:
		print(f"    ERROR: Target column '{target_col}' not found")
		return

	analysis_cols = available_weather + [target_col]
	weather_data = data[analysis_cols].dropna()

	if len(weather_data) < 10:
		print(f"    Not enough data with weather features: {len(weather_data)} rows")
		return

	corr_matrix = weather_data.corr()

	fig, axes = plt.subplots(1, 2, figsize=(16, 6))

	# Full weather correlation matrix
	ax = axes[0]
	sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
	            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
	ax.set_title('Weather Features Correlation Matrix')

	# Target correlation only
	ax = axes[1]
	target_corr = corr_matrix[target_col].drop(target_col).sort_values()
	colors = ['red' if x < 0 else 'steelblue' for x in target_corr.values]
	ax.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
	ax.set_yticks(range(len(target_corr)))
	ax.set_yticklabels(target_corr.index)
	ax.set_xlabel('Correlation with Target')
	ax.set_title(f'Weather Correlation with {target_col}')
	ax.axvline(0, color='black', linewidth=0.5)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, 'weather_correlations.png'), dpi=300, bbox_inches='tight')
	plt.close()

	print(f"    Saved: weather_correlations.png")
	print(f"    Analyzed {len(weather_data)} rows with complete weather data")


def plot_anomaly_isolation_forest(data, target_col, contamination=0.05, suffix=''):
	"""Detect and plot anomalies using Isolation Forest."""
	print(f"  Running Isolation Forest anomaly detection...")

	if target_col not in data.columns:
		print(f"    ERROR: Target column '{target_col}' not found")
		return

	exclude_cols = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	                "elapsed_seconds", "hour", "minute", "second", "dataset_id",
	                "GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	                "OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"]

	numeric_data = data.select_dtypes(include=[np.number])
	feature_data = numeric_data.drop(columns=exclude_cols, errors='ignore')
	data_clean = feature_data.dropna()

	if len(data_clean) < 100:
		print(f"    Not enough data: {len(data_clean)} rows")
		return

	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(data_clean)

	iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
	predictions = iso_forest.fit_predict(data_scaled)
	anomaly_scores = iso_forest.decision_function(data_scaled)

	anomalies = predictions == -1
	anomaly_count = anomalies.sum()

	print(f"    Anomalies detected: {anomaly_count} ({(anomaly_count/len(data_clean))*100:.2f}%)")

	pca = PCA(n_components=2)
	data_pca = pca.fit_transform(data_scaled)

	fig, axes = plt.subplots(1, 2, figsize=(16, 6))

	ax = axes[0]
	ax.scatter(data_pca[~anomalies, 0], data_pca[~anomalies, 1], c='blue', alpha=0.5, s=20, label='Normal')
	ax.scatter(data_pca[anomalies, 0], data_pca[anomalies, 1], c='red', alpha=0.8, s=50, marker='x', label='Anomaly')
	ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
	ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
	ax.set_title(f'Isolation Forest: {anomaly_count} anomalies')
	ax.legend()
	ax.grid(True, alpha=0.3)

	ax = axes[1]
	ax.hist(anomaly_scores, bins=50, edgecolor='black', alpha=0.7)
	ax.axvline(np.percentile(anomaly_scores, contamination*100), color='red', linestyle='--',
	           linewidth=2, label=f'Threshold (bottom {contamination*100:.0f}%)')
	ax.set_xlabel('Anomaly Score')
	ax.set_ylabel('Frequency')
	ax.set_title('Anomaly Score Distribution')
	ax.legend()
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	filename = f'anomaly_isolation_forest{suffix}.png'
	plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
	plt.close()

	print(f"    Saved: {filename}")
	return anomalies


def plot_anomaly_lof(data, target_col, n_neighbors=20, contamination=0.05, suffix=''):
	"""Detect and plot anomalies using Local Outlier Factor."""
	print(f"  Running LOF anomaly detection...")

	if target_col not in data.columns:
		print(f"    ERROR: Target column '{target_col}' not found")
		return

	exclude_cols = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	                "elapsed_seconds", "hour", "minute", "second", "dataset_id",
	                "GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	                "OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"]

	numeric_data = data.select_dtypes(include=[np.number])
	feature_data = numeric_data.drop(columns=exclude_cols, errors='ignore')
	data_clean = feature_data.dropna()

	if len(data_clean) < 100:
		print(f"    Not enough data: {len(data_clean)} rows")
		return

	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(data_clean)

	lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
	predictions = lof.fit_predict(data_scaled)
	lof_scores = -lof.negative_outlier_factor_

	anomalies = predictions == -1
	anomaly_count = anomalies.sum()

	print(f"    Anomalies detected: {anomaly_count} ({(anomaly_count/len(data_clean))*100:.2f}%)")

	pca = PCA(n_components=2)
	data_pca = pca.fit_transform(data_scaled)

	fig, axes = plt.subplots(1, 2, figsize=(16, 6))

	ax = axes[0]
	ax.scatter(data_pca[~anomalies, 0], data_pca[~anomalies, 1], c='blue', alpha=0.5, s=20, label='Normal')
	ax.scatter(data_pca[anomalies, 0], data_pca[anomalies, 1], c='red', alpha=0.8, s=50, marker='x', label='Anomaly')
	ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
	ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
	ax.set_title(f'LOF: {anomaly_count} anomalies')
	ax.legend()
	ax.grid(True, alpha=0.3)

	ax = axes[1]
	ax.hist(lof_scores, bins=50, edgecolor='black', alpha=0.7)
	ax.axvline(np.percentile(lof_scores, (1-contamination)*100), color='red', linestyle='--',
	           linewidth=2, label=f'Threshold (top {contamination*100:.0f}%)')
	ax.set_xlabel('LOF Score')
	ax.set_ylabel('Frequency')
	ax.set_title('LOF Score Distribution')
	ax.legend()
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	filename = f'anomaly_lof{suffix}.png'
	plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
	plt.close()

	print(f"    Saved: {filename}")
	return anomalies


def plot_anomaly_comparison(data, iso_anomalies, lof_anomalies, target_col, suffix=''):
	"""Compare anomalies detected by both methods."""
	print(f"  Comparing anomaly detection methods...")

	if iso_anomalies is None or lof_anomalies is None:
		print(f"    ERROR: Missing anomaly data")
		return

	if target_col not in data.columns:
		print(f"    ERROR: Target column '{target_col}' not found")
		return

	exclude_cols = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	                "elapsed_seconds", "hour", "minute", "second", "dataset_id",
	                "GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	                "OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"]

	numeric_data = data.select_dtypes(include=[np.number])
	feature_data = numeric_data.drop(columns=exclude_cols, errors='ignore')
	data_clean = feature_data.dropna()

	both = iso_anomalies & lof_anomalies
	only_iso = iso_anomalies & ~lof_anomalies
	only_lof = ~iso_anomalies & lof_anomalies
	neither = ~iso_anomalies & ~lof_anomalies

	both_count = both.sum()
	only_iso_count = only_iso.sum()
	only_lof_count = only_lof.sum()
	neither_count = neither.sum()

	print(f"    Both methods: {both_count}")
	print(f"    Only Isolation Forest: {only_iso_count}")
	print(f"    Only LOF: {only_lof_count}")
	print(f"    Neither (normal): {neither_count}")

	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(data_clean)
	pca = PCA(n_components=2)
	data_pca = pca.fit_transform(data_scaled)

	fig, axes = plt.subplots(1, 2, figsize=(16, 6))

	ax = axes[0]
	ax.scatter(data_pca[neither, 0], data_pca[neither, 1], c='blue', alpha=0.3, s=20, label=f'Normal ({neither_count})')
	ax.scatter(data_pca[only_iso, 0], data_pca[only_iso, 1], c='orange', alpha=0.7, s=40, label=f'Only IF ({only_iso_count})')
	ax.scatter(data_pca[only_lof, 0], data_pca[only_lof, 1], c='green', alpha=0.7, s=40, label=f'Only LOF ({only_lof_count})')
	ax.scatter(data_pca[both, 0], data_pca[both, 1], c='red', alpha=0.9, s=60, marker='x', label=f'Both ({both_count})')
	ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
	ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
	ax.set_title('Anomaly Comparison')
	ax.legend()
	ax.grid(True, alpha=0.3)

	ax = axes[1]
	categories = ['Both Methods', 'Only IF', 'Only LOF', 'Normal']
	counts = [both_count, only_iso_count, only_lof_count, neither_count]
	colors = ['red', 'orange', 'green', 'blue']

	bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
	ax.set_ylabel('Count')
	ax.set_title('Anomaly Detection Summary')
	ax.grid(True, alpha=0.3, axis='y')

	for bar, count in zip(bars, counts):
		ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
		        f'{count}\n({count/len(data_clean)*100:.1f}%)',
		        ha='center', va='bottom', fontsize=10)

	plt.tight_layout()
	filename = f'anomaly_comparison{suffix}.png'
	plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
	plt.close()

	print(f"    Saved: {filename}")


def main():
	"""Create visualizations from preprocessed data."""
	TARGET_COL = 'OPC_12_CPP_ENGINE_POWER'

	print("\n" + "="*70)
	print("CREATING VISUALIZATIONS FROM PREPROCESSED DATA")
	print("="*70)

	# Load preprocessed data
	regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
	weather_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")

	if not os.path.exists(regular_path):
		print(f"\nERROR: {regular_path} not found!")
		print("Run pre_process.py first to create the preprocessed data.")
		return

	if not os.path.exists(weather_path):
		print(f"\nERROR: {weather_path} not found!")
		print("Run pre_process.py first to create the preprocessed data.")
		return

	print(f"\nLoading preprocessed data...")
	data_regular = pd.read_csv(regular_path)
	data_weather = pd.read_csv(weather_path)

	print(f"  SPEED_TRIALS_REGULAR_FINAL: {data_regular.shape}")
	print(f"  SPEED_TRIALS_WEATHER_FINAL: {data_weather.shape}")

	# ========================================================================
	# VISUALIZATIONS FOR SPEED_TRIALS_REGULAR
	# ========================================================================
	print("\n" + "="*70)
	print("CREATING VISUALIZATIONS: SPEED_TRIALS_REGULAR")
	print("="*70)

	plot_target_over_time(data_regular, TARGET_COL, suffix='_regular')
	plot_feature_importance(data_regular, TARGET_COL, suffix='_regular')
	plot_correlation_matrix(data_regular, TARGET_COL, suffix='_regular')

	iso_anomalies_regular = plot_anomaly_isolation_forest(data_regular, TARGET_COL, suffix='_regular')
	lof_anomalies_regular = plot_anomaly_lof(data_regular, TARGET_COL, suffix='_regular')
	plot_anomaly_comparison(data_regular, iso_anomalies_regular, lof_anomalies_regular, TARGET_COL, suffix='_regular')

	# ========================================================================
	# VISUALIZATIONS FOR SPEED_TRIALS_WEATHER
	# ========================================================================
	print("\n" + "="*70)
	print("CREATING VISUALIZATIONS: SPEED_TRIALS_WEATHER")
	print("="*70)

	plot_target_over_time(data_weather, TARGET_COL, suffix='_weather')
	plot_feature_importance(data_weather, TARGET_COL, suffix='_weather')
	plot_correlation_matrix(data_weather, TARGET_COL, suffix='_weather')
	plot_weather_correlations(data_weather, TARGET_COL)

	iso_anomalies_weather = plot_anomaly_isolation_forest(data_weather, TARGET_COL, suffix='_weather')
	lof_anomalies_weather = plot_anomaly_lof(data_weather, TARGET_COL, suffix='_weather')
	plot_anomaly_comparison(data_weather, iso_anomalies_weather, lof_anomalies_weather, TARGET_COL, suffix='_weather')

	# ========================================================================
	# SUMMARY
	# ========================================================================
	print(f"\n{'='*70}")
	print("VISUALIZATIONS COMPLETE")
	print(f"{'='*70}")
	print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
	print(f"\nSpeed Trials Regular:")
	print(f"  - target_over_time_regular.png")
	print(f"  - feature_importance_regular.png")
	print(f"  - correlation_matrix_regular.png")
	print(f"  - anomaly_isolation_forest_regular.png")
	print(f"  - anomaly_lof_regular.png")
	print(f"  - anomaly_comparison_regular.png")
	print(f"\nSpeed Trials Weather:")
	print(f"  - target_over_time_weather.png")
	print(f"  - feature_importance_weather.png")
	print(f"  - correlation_matrix_weather.png")
	print(f"  - weather_correlations.png")
	print(f"  - anomaly_isolation_forest_weather.png")
	print(f"  - anomaly_lof_weather.png")
	print(f"  - anomaly_comparison_weather.png")


if __name__ == "__main__":
	main()