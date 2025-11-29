import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from carbontracker.tracker import CarbonTracker

os.makedirs('model_performance', exist_ok=True)

EXCLUDE_COLS = [
	"OPC_12_CPP_ENGINE_POWER",
	"OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	"elapsed_seconds", "hour", "minute", "second", "dataset_id",
	"GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	"OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"
]


def create_sequences(df, target_col, n_lags, forecast_horizon):
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in EXCLUDE_COLS]

	print(f"  Feature columns: {len(feature_cols)}")

	X_list, y_list = [], []
	for i in range(n_lags, len(df) - forecast_horizon):
		X_list.append(df[feature_cols].iloc[i - n_lags:i].values)
		y_list.append(df[target_col].iloc[i + forecast_horizon])

	return np.array(X_list), np.array(y_list), feature_cols


def split_data(X, y, train_ratio=0.6, val_ratio=0.2):
	train_idx = int(len(X) * train_ratio)
	val_idx = int(len(X) * (train_ratio + val_ratio))
	return (X[:train_idx], X[train_idx:val_idx], X[val_idx:],
	        y[:train_idx], y[train_idx:val_idx], y[val_idx:])


def normalize_sequences(X_train, X_val, X_test):
	scaler = StandardScaler()
	X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
	scaler.fit(X_train_reshaped)

	X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
	X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
	X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

	return X_train_scaled, X_val_scaled, X_test_scaled


def create_lstm_model(input_shape):
	model = keras.Sequential([
		layers.LSTM(64, return_sequences=True, input_shape=input_shape),
		layers.Dropout(0.2),
		layers.LSTM(32, return_sequences=False),
		layers.Dropout(0.2),
		layers.Dense(1)
	])
	model.compile(optimizer='adam', loss='mse', metrics=['mae'])
	return model


def plot_actual_vs_predicted(y_true, y_pred, split_name, model_name):
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


def plot_prediction_errors(y_true, y_pred, split_name, model_name):
	errors = y_true - y_pred
	plt.figure(figsize=(15, 5))
	plt.plot(errors[:1000])
	plt.xlabel('Time Step')
	plt.ylabel('Prediction Error')
	plt.title(f'Prediction Errors - {split_name}')
	plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
	plt.grid(True)
	plt.savefig(f'model_performance/{model_name}_{split_name}_prediction_errors.png', dpi=300, bbox_inches='tight')
	plt.close()


def plot_error_variance(y_true, y_pred, split_name, model_name):
	errors = y_true - y_pred
	plt.figure(figsize=(10, 5))
	plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
	plt.xlabel('Prediction Error')
	plt.ylabel('Frequency')
	plt.title(f'Error Distribution - {split_name}')
	plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
	plt.grid(True)
	plt.savefig(f'model_performance/{model_name}_{split_name}_error_distribution.png', dpi=300, bbox_inches='tight')
	plt.close()


def plot_training_loss(history, model_name):
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


def train_and_evaluate(dataset, dataset_name, n_lags=60):
	target_col = "OPC_12_CPP_ENGINE_POWER"

	print(f"\n{'='*60}")
	print(f"LSTM - {dataset_name}")
	print(f"{'='*60}")
	print(f"Dataset shape: {dataset.shape}")

	print(f"\nCreating sequences with n_lags={n_lags}...")
	X, y, feature_cols = create_sequences(dataset, target_col, n_lags, 10)
	print(f"  Samples: {len(y)}, Sequence shape: {X.shape}")

	X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
	X_train_scaled, X_val_scaled, X_test_scaled = normalize_sequences(X_train, X_val, X_test)

	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()

	model = create_lstm_model((X_train_scaled.shape[1], X_train_scaled.shape[2]))
	history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32,
	                    validation_data=(X_val_scaled, y_val), verbose=1)

	tracker.epoch_end()

	model_name = f"LSTM_lags{n_lags}_{dataset_name.replace(' ', '_')}"
	plot_training_loss(history, model_name)

	# Train evaluation
	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()
	y_train_pred = model.predict(X_train_scaled, verbose=1).flatten()
	r2_train = r2_score(y_train, y_train_pred)
	mse_train = mean_squared_error(y_train, y_train_pred)
	mae_train = mean_absolute_error(y_train, y_train_pred)
	plot_actual_vs_predicted(y_train, y_train_pred, 'train', model_name)
	plot_prediction_errors(y_train, y_train_pred, 'train', model_name)
	plot_error_variance(y_train, y_train_pred, 'train', model_name)
	tracker.epoch_end()

	# Validation evaluation
	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()
	y_val_pred = model.predict(X_val_scaled, verbose=1).flatten()
	r2_val = r2_score(y_val, y_val_pred)
	mse_val = mean_squared_error(y_val, y_val_pred)
	mae_val = mean_absolute_error(y_val, y_val_pred)
	plot_actual_vs_predicted(y_val, y_val_pred, 'val', model_name)
	plot_prediction_errors(y_val, y_val_pred, 'val', model_name)
	plot_error_variance(y_val, y_val_pred, 'val', model_name)
	tracker.epoch_end()

	# Test evaluation
	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()
	y_test_pred = model.predict(X_test_scaled, verbose=1).flatten()
	r2_test = r2_score(y_test, y_test_pred)
	mse_test = mean_squared_error(y_test, y_test_pred)
	mae_test = mean_absolute_error(y_test, y_test_pred)
	plot_actual_vs_predicted(y_test, y_test_pred, 'test', model_name)
	plot_prediction_errors(y_test, y_test_pred, 'test', model_name)
	plot_error_variance(y_test, y_test_pred, 'test', model_name)
	tracker.epoch_end()

	print(f"\nLSTM with n_lags={n_lags} - {dataset_name}:")
	print(f"Train - R²: {r2_train:.4f}, MSE: {mse_train:.4f}, MAE: {mae_train:.4f}")
	print(f"Val   - R²: {r2_val:.4f}, MSE: {mse_val:.4f}, MAE: {mae_val:.4f}")
	print(f"Test  - R²: {r2_test:.4f}, MSE: {mse_test:.4f}, MAE: {mae_test:.4f}")


OUTPUT_DIR = "output"
regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
weather_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")

if os.path.exists(regular_path) and os.path.exists(weather_path):
	print("Loading preprocessed data...")
	speed_trials_regular = pd.read_csv(regular_path)
	speed_trials_weather = pd.read_csv(weather_path)

	print(f"Loaded SPEED_TRIALS_REGULAR_FINAL: {speed_trials_regular.shape}")
	print(f"Loaded SPEED_TRIALS_WEATHER_FINAL: {speed_trials_weather.shape}")

	train_and_evaluate(speed_trials_regular, "Regular", n_lags=60)
	train_and_evaluate(speed_trials_weather, "Weather", n_lags=60)
else:
	print("ERROR: Run pre_process.py first!")