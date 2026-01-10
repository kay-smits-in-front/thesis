"""
RNN Optimization - CLEAN VERSION
Only tests physics_weight=0.0 (data-only model)
Only uses regular dataset for faster optimization
Tests multiple architectures and batch sizes
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


CONFIG = {
	'output_dir': 'model_performance',
	'architectures': [[64], [64, 32], [128, 64, 32]],
	'batch_sizes': [16, 32],
	'physics_weights': [0.0],  # ONLY data-only model (no physics)
	'timesteps': 15,
	'epochs': 15,
	'patience': 7,
	'learning_rate': 0.001,
	'dropout_rate': 0.2
}

EXCLUDE_COLS = [
	"OPC_12_CPP_ENGINE_POWER",
	"OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	"elapsed_seconds", "hour", "minute", "second", "dataset_id",
	"GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	"OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"
]

os.makedirs(CONFIG['output_dir'], exist_ok=True)


def create_sequences(X, y, timesteps):
	"""Create sequences for RNN input"""
	X_seq, y_seq = [], []
	for i in range(timesteps, len(X)):
		X_seq.append(X[i-timesteps:i])
		y_seq.append(y[i])
	return np.array(X_seq), np.array(y_seq)


class RNNModel(keras.Model):
	"""Simple RNN model for data-only training (no physics)"""
	def __init__(self, architecture, dropout_rate):
		super().__init__()
		self.rnn_layers = []
		self.dropout_layers = []

		for i, units in enumerate(architecture):
			return_sequences = (i < len(architecture) - 1)
			self.rnn_layers.append(SimpleRNN(units, return_sequences=return_sequences))
			self.dropout_layers.append(Dropout(dropout_rate))

		self.output_layer = Dense(1)

	def call(self, inputs):
		x = inputs
		for rnn, dropout in zip(self.rnn_layers, self.dropout_layers):
			x = rnn(x)
			x = dropout(x)
		return self.output_layer(x)


def prepare_data(data, target_col, timesteps):
	"""Prepare data with proper splitting to prevent leakage"""
	all_exclude = EXCLUDE_COLS + [target_col]
	numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in all_exclude]

	X = data[feature_cols]
	y = data[target_col]

	# Remove NaN values
	valid_mask = ~(X.isna().any(axis=1) | y.isna())
	X = X[valid_mask]
	y = y[valid_mask]

	# Split BEFORE scaling (60/20/20)
	train_size = int(len(X) * 0.6)
	val_size = int(len(X) * 0.2)

	X_train_raw = X.iloc[:train_size]
	y_train_raw = y.iloc[:train_size]
	X_val_raw = X.iloc[train_size:train_size+val_size]
	y_val_raw = y.iloc[train_size:train_size+val_size]
	X_test_raw = X.iloc[train_size+val_size:]
	y_test_raw = y.iloc[train_size+val_size:]

	# Fit scalers ONLY on training data
	scaler_X = StandardScaler()
	scaler_y = StandardScaler()
	scaler_X.fit(X_train_raw)
	scaler_y.fit(y_train_raw.values.reshape(-1, 1))

	# Transform each split separately
	X_train_scaled = scaler_X.transform(X_train_raw)
	y_train_scaled = scaler_y.transform(y_train_raw.values.reshape(-1, 1)).flatten()
	X_val_scaled = scaler_X.transform(X_val_raw)
	y_val_scaled = scaler_y.transform(y_val_raw.values.reshape(-1, 1)).flatten()
	X_test_scaled = scaler_X.transform(X_test_raw)
	y_test_scaled = scaler_y.transform(y_test_raw.values.reshape(-1, 1)).flatten()

	# Create sequences AFTER scaling (prevents leakage)
	X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, timesteps)
	X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, timesteps)
	X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, timesteps)

	splits = {
		'X_train': tf.convert_to_tensor(X_train_seq, dtype=tf.float32),
		'y_train': tf.convert_to_tensor(y_train_seq, dtype=tf.float32),
		'X_val': tf.convert_to_tensor(X_val_seq, dtype=tf.float32),
		'y_val': tf.convert_to_tensor(y_val_seq, dtype=tf.float32),
		'X_test': tf.convert_to_tensor(X_test_seq, dtype=tf.float32),
		'y_test': tf.convert_to_tensor(y_test_seq, dtype=tf.float32)
	}

	return splits, scaler_y


def train_single_model(splits, scaler_y, architecture, batch_size, config):
	"""Train a single model configuration"""
	model = RNNModel(architecture, config['dropout_rate'])
	model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='mse')

	# Early stopping
	early_stop = keras.callbacks.EarlyStopping(
		monitor='val_loss', patience=config['patience'], restore_best_weights=True
	)

	# Train
	model.fit(
		splits['X_train'], splits['y_train'],
		validation_data=(splits['X_val'], splits['y_val']),
		epochs=config['epochs'],
		batch_size=batch_size,
		callbacks=[early_stop],
		verbose=0
	)

	# Evaluate on all splits
	def evaluate(X, y):
		y_pred = model.predict(X, verbose=0).flatten()
		y_original = scaler_y.inverse_transform(y.numpy().reshape(-1, 1)).flatten()
		y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
		return {
			'r2': float(r2_score(y_original, y_pred_original)),
			'mse': float(mean_squared_error(y_original, y_pred_original)),
			'mae': float(mean_absolute_error(y_original, y_pred_original))
		}

	return {
		'architecture': architecture,
		'batch_size': batch_size,
		'train': evaluate(splits['X_train'], splits['y_train']),
		'val': evaluate(splits['X_val'], splits['y_val']),
		'test': evaluate(splits['X_test'], splits['y_test'])
	}


def run_optimization(data, target_col, config):
	"""Run optimization over all architecture and batch size combinations"""
	print(f"\n{'='*70}")
	print(f"RNN OPTIMIZATION - DATA-ONLY MODEL (physics_weight=0.0)")
	print(f"Dataset: REGULAR (no weather features)")
	print(f"{'='*70}")

	splits, scaler_y = prepare_data(data, target_col, config['timesteps'])

	print(f"Train samples: {len(splits['X_train'])}")
	print(f"Val samples: {len(splits['X_val'])}")
	print(f"Test samples: {len(splits['X_test'])}")

	results = []
	total_configs = len(config['architectures']) * len(config['batch_sizes'])
	current = 0

	for arch in config['architectures']:
		for batch_size in config['batch_sizes']:
			current += 1
			print(f"\n[{current}/{total_configs}] Training: arch={arch}, batch={batch_size}")

			result = train_single_model(splits, scaler_y, arch, batch_size, config)
			results.append(result)

			print(f"  Test R²: {result['test']['r2']:.4f}, MSE: {result['test']['mse']:.2f}, MAE: {result['test']['mae']:.2f}")

	# Find best configuration
	best_result = max(results, key=lambda x: x['test']['r2'])
	print(f"\n{'='*70}")
	print(f"BEST CONFIGURATION:")
	print(f"  Architecture: {best_result['architecture']}")
	print(f"  Batch size: {best_result['batch_size']}")
	print(f"  Test R²: {best_result['test']['r2']:.4f}")
	print(f"  Test MSE: {best_result['test']['mse']:.2f}")
	print(f"  Test MAE: {best_result['test']['mae']:.2f}")
	print(f"{'='*70}")

	return results


if __name__ == "__main__":
	OUTPUT_DIR = "output"
	regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")

	if os.path.exists(regular_path):
		speed_trials_regular = pd.read_csv(regular_path)
		target_col = 'OPC_12_CPP_ENGINE_POWER'

		results = run_optimization(speed_trials_regular, target_col, CONFIG)

		# Save results
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_file = f'{CONFIG["output_dir"]}/rnn_optimization_clean_{timestamp}.json'
		with open(output_file, 'w') as f:
			json.dump(results, f, indent=2)
		print(f"\nResults saved to: {output_file}")
	else:
		print("ERROR: Run pre_process.py first!")
