import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.src.layers import Dropout
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from carbontracker.tracker import CarbonTracker

os.makedirs('model_performance', exist_ok=True)


def create_multivariate_lag_features(df, target_col, n_lags, forecast_horizon):
	# Exclude columns to prevent feature leakage and redundancy
	exclude_cols = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	                "elapsed_seconds", "hour", "minute", "second", "dataset_id",
	                "GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	                "OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"]

	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in exclude_cols]

	if target_col not in numeric_cols:
		raise ValueError(f"{target_col} not found in numeric columns")

	print(f"  Total numeric columns: {len(numeric_cols)}")
	print(f"  Excluded columns: {len(exclude_cols)}")
	print(f"  Feature columns used: {len(feature_cols)}")

	X_list = []
	y_list = []

	for i in range(n_lags, len(df) - forecast_horizon):
		features = []
		for col in feature_cols:
			lag_values = df[col].iloc[i - n_lags:i].values
			features.extend(lag_values)

		target = df[target_col].iloc[i + forecast_horizon]

		X_list.append(features)
		y_list.append(target)

	X = np.array(X_list)
	y = np.array(y_list)

	return X, y


def create_mlp_model(input_dim: int) -> keras.Model:
	from tensorflow.keras import regularizers

	# L2 regularization to prevent overfitting
	l2_reg = 0.0001

	model = keras.Sequential([
		layers.Dense(32, activation='relu',  # Reduced from 32
		             kernel_regularizer=regularizers.l2(l2_reg)),
		layers.Dense(1)
	])
	optimizer = keras.optimizers.Adam(learning_rate=0.01)
	model.compile(optimizer=optimizer, loss='mse')
	return model


def normalize_data(X_train, X_val, X_test):
	scaler = StandardScaler()
	scaler.fit(X_train)

	X_train_scaled = scaler.transform(X_train)
	X_val_scaled = scaler.transform(X_val)
	X_test_scaled = scaler.transform(X_test)

	return X_train_scaled, X_val_scaled, X_test_scaled


def plot_actual_vs_predicted(y_true, y_pred, split_name: str, model_name: str):
	plt.figure(figsize=(15, 5))
	plt.plot(y_true[:1000], label='Actual', alpha=0.7)
	plt.plot(y_pred[:1000], label='Predicted', alpha=0.7)
	plt.xlabel('Time Step')
	plt.ylabel('Engine Power')
	plt.title(f'Actual vs Predicted - {split_name}')
	plt.legend()
	plt.grid(True)
	plt.savefig(f'model_performance/{model_name}_{split_name}_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
	plt.close()


def plot_prediction_errors(y_true, y_pred, split_name: str, model_name: str):
	errors = y_true - y_pred.flatten()

	plt.figure(figsize=(15, 5))
	plt.plot(errors[:1000])
	plt.xlabel('Time Step')
	plt.ylabel('Prediction Error')
	plt.title(f'Prediction Errors - {split_name}')
	plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
	plt.grid(True)
	plt.savefig(f'model_performance/{model_name}_{split_name}_prediction_errors.png', dpi=300, bbox_inches='tight')
	plt.close()


def plot_error_variance(y_true, y_pred, split_name: str, model_name: str):
	errors = y_true - y_pred.flatten()

	plt.figure(figsize=(10, 5))
	plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
	plt.xlabel('Prediction Error')
	plt.ylabel('Frequency')
	plt.title(f'Error Distribution - {split_name}')
	plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
	plt.grid(True)
	plt.savefig(f'model_performance/{model_name}_{split_name}_error_distribution.png', dpi=300, bbox_inches='tight')
	plt.close()


def plot_training_loss(history, model_name: str):
	plt.figure(figsize=(10, 5))
	plt.plot(history.history['loss'], label='Train Loss')
	plt.plot(history.history['val_loss'], label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss (MSE)')
	plt.title('Training and Validation Loss')
	plt.legend()
	plt.grid(True)
	plt.savefig(f'model_performance/{model_name}_training_loss.png', dpi=300, bbox_inches='tight')
	plt.close()


def split_data(X, y, train_ratio=0.6, val_ratio=0.2):
	train_idx = int(len(X) * train_ratio)
	val_idx = int(len(X) * (train_ratio + val_ratio))

	X_train = X[:train_idx]
	X_val = X[train_idx:val_idx]
	X_test = X[val_idx:]

	y_train = y[:train_idx]
	y_val = y[train_idx:val_idx]
	y_test = y[val_idx:]

	return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate(dataset, dataset_name: str, n_lags: int):
	target_col = "OPC_12_CPP_ENGINE_POWER"

	print(f"\n{'='*60}")
	print(f"Processing {dataset_name}")
	print(f"{'='*60}")
	print(f"Dataset shape: {dataset.shape}")

	print(f"\nCreating multivariate lag features with n_lags={n_lags}...")

	X, y = create_multivariate_lag_features(dataset, target_col, n_lags=n_lags, forecast_horizon=15)

	print(f"  Feature matrix shape: {X.shape}")
	print(f"  Number of samples: {len(y)}")

	X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

	X_train_scaled, X_val_scaled, X_test_scaled = normalize_data(X_train, X_val, X_test)

	model = create_mlp_model(input_dim=X_train_scaled.shape[1])



	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()

	history = model.fit(X_train_scaled, y_train, epochs=60, batch_size=32, verbose=1,
	                    validation_data=(X_val_scaled, y_val))

	tracker.epoch_end()

	model_name = f"MLP_lags{n_lags}_{dataset_name.replace(' ', '_')}"

	plot_training_loss(history, model_name)

	y_train_pred = model.predict(X_train_scaled, verbose=1)
	r2_train = r2_score(y_train, y_train_pred)
	mse_train = mean_squared_error(y_train, y_train_pred)
	mae_train = mean_absolute_error(y_train, y_train_pred)

	plot_actual_vs_predicted(y_train, y_train_pred, 'train', model_name)
	plot_prediction_errors(y_train, y_train_pred, 'train', model_name)
	plot_error_variance(y_train, y_train_pred, 'train', model_name)

	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()

	y_val_pred = model.predict(X_val_scaled, verbose=1)
	r2_val = r2_score(y_val, y_val_pred)
	mse_val = mean_squared_error(y_val, y_val_pred)
	mae_val = mean_absolute_error(y_val, y_val_pred)

	plot_actual_vs_predicted(y_val, y_val_pred, 'val', model_name)
	plot_prediction_errors(y_val, y_val_pred, 'val', model_name)
	plot_error_variance(y_val, y_val_pred, 'val', model_name)

	tracker.epoch_end()

	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()

	y_test_pred = model.predict(X_test_scaled, verbose=1)
	r2_test = r2_score(y_test, y_test_pred)
	mse_test = mean_squared_error(y_test, y_test_pred)
	mae_test = mean_absolute_error(y_test, y_test_pred)

	plot_actual_vs_predicted(y_test, y_test_pred, 'test', model_name)
	plot_prediction_errors(y_test, y_test_pred, 'test', model_name)
	plot_error_variance(y_test, y_test_pred, 'test', model_name)

	tracker.epoch_end()

	print(f"\nMLP with n_lags={n_lags} - {dataset_name}:")
	print(f"Train  - R²: {r2_train:.4f}, MSE: {mse_train:.4f}, MAE: {mae_train:.4f}")
	print(f"Val    - R²: {r2_val:.4f}, MSE: {mse_val:.4f}, MAE: {mae_val:.4f}")
	print(f"Test   - R²: {r2_test:.4f}, MSE: {mse_test:.4f}, MAE: {mae_test:.4f}")


print("Loading preprocessed data...")
OUTPUT_DIR = "output"

regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
weather_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")

if not os.path.exists(regular_path):
	print(f"\nERROR: {regular_path} not found!")
	print("Run pre_process.py first to create the preprocessed data.")
	exit(1)

if not os.path.exists(weather_path):
	print(f"\nERROR: {weather_path} not found!")
	print("Run pre_process.py first to create the preprocessed data.")
	exit(1)

speed_trials_regular = pd.read_csv(regular_path)
speed_trials_weather = pd.read_csv(weather_path)

print(f"Loaded SPEED_TRIALS_REGULAR_FINAL: {speed_trials_regular.shape}")
print(f"Loaded SPEED_TRIALS_WEATHER_FINAL: {speed_trials_weather.shape}")

train_and_evaluate(speed_trials_regular, "Regular", n_lags=30)
train_and_evaluate(speed_trials_weather, "Weather", n_lags=30)