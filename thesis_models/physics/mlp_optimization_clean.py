"""
MLP Optimization with Simple Early Pruning
Tests architectures and batch sizes efficiently (NO physics - data-only)
Prunes poorly performing architectures early to save compute time
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


CONFIG = {
	'output_dir': 'model_performance',
	'architectures': [[128, 64, 32], [64, 32], [64]],
	'batch_sizes': [16, 32],
	'n_lags': 15,
	'epochs': 20,
	'patience': 7,
	'learning_rate': 0.001,
	'dropout_rate': 0.2,
	'prune_threshold': 0.5  # Prune if validation R² < 0.5
}

EXCLUDE_COLS = [
	"OPC_12_CPP_ENGINE_POWER",
	"OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	"elapsed_seconds", "hour", "minute", "second", "dataset_id",
	"GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	"OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"
]

os.makedirs(CONFIG['output_dir'], exist_ok=True)


def create_multivariate_lag_features(df, target_col, n_lags):
	"""Create lag features for all numeric columns"""
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in EXCLUDE_COLS]

	X_list, y_list = [], []
	for i in range(n_lags, len(df)):
		features = []
		for col in feature_cols:
			features.extend(df[col].iloc[i - n_lags:i].values)
		X_list.append(features)
		y_list.append(df[target_col].iloc[i])

	return np.array(X_list), np.array(y_list), feature_cols


class MLPModel(keras.Model):
	"""Simple MLP model for data-only training"""
	def __init__(self, architecture, dropout_rate):
		super().__init__()
		self.dense_layers = []
		self.dropout_layers = []
		for units in architecture:
			self.dense_layers.append(Dense(units, activation='relu'))
			self.dropout_layers.append(Dropout(dropout_rate))
		self.output_layer = Dense(1)

	def call(self, inputs):
		x = inputs
		for dense, dropout in zip(self.dense_layers, self.dropout_layers):
			x = dense(x)
			x = dropout(x)
		return self.output_layer(x)


def prepare_data(df, target_col, n_lags):
	"""Prepare data with proper splitting to prevent leakage"""
	X, y, feature_cols = create_multivariate_lag_features(df, target_col, n_lags)

	# Split BEFORE scaling (60/20/20)
	train_size = int(len(X) * 0.6)
	val_size = int(len(X) * 0.2)

	X_train_raw = X[:train_size]
	y_train_raw = y[:train_size]
	X_val_raw = X[train_size:train_size+val_size]
	y_val_raw = y[train_size:train_size+val_size]
	X_test_raw = X[train_size+val_size:]
	y_test_raw = y[train_size+val_size:]

	# Fit scalers ONLY on training data
	scaler_X = StandardScaler()
	scaler_y = StandardScaler()
	scaler_X.fit(X_train_raw)
	scaler_y.fit(y_train_raw.reshape(-1, 1))

	# Transform each split separately
	X_train_scaled = scaler_X.transform(X_train_raw)
	y_train_scaled = scaler_y.transform(y_train_raw.reshape(-1, 1)).flatten()
	X_val_scaled = scaler_X.transform(X_val_raw)
	y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
	X_test_scaled = scaler_X.transform(X_test_raw)
	y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

	splits = {
		'X_train': tf.convert_to_tensor(X_train_scaled, dtype=tf.float32),
		'y_train': tf.convert_to_tensor(y_train_scaled, dtype=tf.float32),
		'X_val': tf.convert_to_tensor(X_val_scaled, dtype=tf.float32),
		'y_val': tf.convert_to_tensor(y_val_scaled, dtype=tf.float32),
		'X_test': tf.convert_to_tensor(X_test_scaled, dtype=tf.float32),
		'y_test': tf.convert_to_tensor(y_test_scaled, dtype=tf.float32)
	}

	return splits, scaler_y


def train_single_model(splits, scaler_y, architecture, batch_size, config):
	"""Train a single model configuration"""
	model = MLPModel(architecture, config['dropout_rate'])
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
	"""Run optimization with simple architecture pruning"""
	print(f"\n{'='*80}")
	print(f"MLP OPTIMIZATION WITH EARLY PRUNING (DATA-ONLY)")
	print(f"Dataset: REGULAR | Prune threshold: Val R² < {config['prune_threshold']}")
	print(f"{'='*80}")

	splits, scaler_y = prepare_data(data, target_col, config['n_lags'])

	print(f"\nTrain: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
	print(f"\nTesting {len(config['architectures'])} architectures × {len(config['batch_sizes'])} batch sizes")

	results = []
	tested = 0
	skipped = 0

	for arch in config['architectures']:
		# Test with smallest batch size first
		first_batch = config['batch_sizes'][0]

		tested += 1
		print(f"\n[{tested}] Architecture: {arch}, Batch: {first_batch}")

		result = train_single_model(splits, scaler_y, arch, first_batch, config)
		results.append(result)

		val_r2 = result['val']['r2']
		test_r2 = result['test']['r2']
		print(f"  Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}, MSE: {result['test']['mse']:.2f}")

		# Early pruning: if first batch performs poorly, skip other batches
		if val_r2 < config['prune_threshold']:
			skipped_batches = len(config['batch_sizes']) - 1
			skipped += skipped_batches
			print(f"  ⚠️  PRUNED: Skipping {skipped_batches} remaining batch sizes (Val R² < {config['prune_threshold']})")
			continue

		# Test remaining batch sizes
		for batch_size in config['batch_sizes'][1:]:
			tested += 1
			print(f"\n[{tested}] Architecture: {arch}, Batch: {batch_size}")

			result = train_single_model(splits, scaler_y, arch, batch_size, config)
			results.append(result)

			print(f"  Val R²: {result['val']['r2']:.4f}, Test R²: {result['test']['r2']:.4f}, MSE: {result['test']['mse']:.2f}")

	# Summary
	print(f"\n{'='*80}")
	print("OPTIMIZATION COMPLETE")
	print(f"{'='*80}")
	print(f"Tested: {tested} configurations")
	print(f"Skipped: {skipped} configurations (early pruning)")
	print(f"Total results: {len(results)}")

	# Find best configuration
	best_result = max(results, key=lambda x: x['test']['r2'])
	print(f"\n{'='*80}")
	print("BEST CONFIGURATION:")
	print(f"{'='*80}")
	print(f"  Architecture: {best_result['architecture']}")
	print(f"  Batch size: {best_result['batch_size']}")
	print(f"  Test R²: {best_result['test']['r2']:.4f}")
	print(f"  Test MSE: {best_result['test']['mse']:.2f}")
	print(f"  Test MAE: {best_result['test']['mae']:.2f}")
	print(f"{'='*80}")

	return {
		'results': results,
		'tested': tested,
		'skipped': skipped,
		'best_config': best_result
	}


if __name__ == "__main__":
	OUTPUT_DIR = "output"
	regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")

	if os.path.exists(regular_path):
		speed_trials_regular = pd.read_csv(regular_path)
		target_col = 'OPC_12_CPP_ENGINE_POWER'

		optimization_results = run_optimization(speed_trials_regular, target_col, CONFIG)

		# Save results
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_file = f'{CONFIG["output_dir"]}/mlp_optimization_{timestamp}.json'
		with open(output_file, 'w') as f:
			json.dump(optimization_results, f, indent=2, default=str)
		print(f"\nResults saved to: {output_file}")
	else:
		print("ERROR: Run pre_process.py first!")
