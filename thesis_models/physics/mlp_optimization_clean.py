"""
MLP Optimization with Early Pruning
Tests architectures, batch sizes, and physics weights efficiently
Prunes poorly performing configurations early to save compute time
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
	'physics_weights': [0.0, 0.001, 0.01],  # Test multiple physics weights
	'n_lags': 15,
	'epochs': 20,
	'patience': 7,
	'learning_rate': 0.001,
	'dropout_rate': 0.2,
	'prune_threshold': 0.3  # Prune if validation R² < 0.3
}

EXCLUDE_COLS = [
	"OPC_12_CPP_ENGINE_POWER",
	"OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	"elapsed_seconds", "hour", "minute", "second", "dataset_id",
	"GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	"OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"
]

SHIP_PARAMS = {
	'DP': 6.5, 'k0': 0.5453, 'k1': -0.4399, 'k2': -0.0379,
	'tP': 0.1, 'wP0': 0.16, 'xP_prime': -0.5, 'L': 214.0
}

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


def compute_propeller_force_tf(u, v, r, nP, params):
	"""Compute propeller force using ship physics"""
	rho = 1025.0
	beta = tf.math.atan2(-v, u)
	r_prime = tf.where(tf.abs(u) > 1e-6, r * params['L'] / u, 0.0)
	betaP = beta - params['xP_prime'] * r_prime
	wP = params['wP0'] * tf.exp(-4 * betaP**2)
	uP = u * (1 - wP)
	JP = tf.where(tf.abs(nP) > 1e-6, uP / (nP * params['DP']), 0.0)
	KT = params['k0'] + params['k1'] * JP + params['k2'] * JP**2
	Tp = rho * nP**2 * params['DP']**4 * KT
	XP = (1 - params['tP']) * Tp
	return XP


class MLPWithPhysics(keras.Model):
	"""MLP model with optional physics-informed loss"""
	def __init__(self, architecture, dropout_rate, physics_weight, scaler_X, scaler_y, feature_cols, n_lags):
		super().__init__()
		self.dense_layers = []
		self.dropout_layers = []
		for units in architecture:
			self.dense_layers.append(Dense(units, activation='relu'))
			self.dropout_layers.append(Dropout(dropout_rate))
		self.output_layer = Dense(1)

		self.physics_weight = physics_weight
		self.optimizer = Adam(learning_rate=0.001)

		# Store scaler stats for physics loss
		self.scaler_X_mean = tf.constant(scaler_X.mean_, dtype=tf.float32)
		self.scaler_X_std = tf.constant(scaler_X.scale_, dtype=tf.float32)
		self.scaler_y_mean = tf.constant(scaler_y.mean_[0], dtype=tf.float32)
		self.scaler_y_std = tf.constant(scaler_y.scale_[0], dtype=tf.float32)

		# Get indices of physics features
		self.n_lags = n_lags
		self.features_per_timestep = len(feature_cols)
		try:
			self.u_idx = feature_cols.index('OPC_07_WATER_SPEED')
			self.v_idx = feature_cols.index('v_ms')
			self.r_idx = feature_cols.index('GPS_HDG_HEADING_ROT_S')
			self.nP_idx = feature_cols.index('OPC_40_PROP_RPM_FB')
		except ValueError:
			self.physics_weight = 0.0  # Disable physics if features missing

	def call(self, inputs):
		x = inputs
		for dense, dropout in zip(self.dense_layers, self.dropout_layers):
			x = dense(x)
			x = dropout(x)
		return self.output_layer(x)

	def compute_physics_loss(self, inputs, predictions):
		"""Compute physics loss using latest timestep"""
		# Extract last timestep features
		last_timestep_start = (self.n_lags - 1) * self.features_per_timestep
		u_scaled = inputs[:, last_timestep_start + self.u_idx]
		v_scaled = inputs[:, last_timestep_start + self.v_idx]
		r_scaled = inputs[:, last_timestep_start + self.r_idx]
		nP_scaled = inputs[:, last_timestep_start + self.nP_idx]

		# Descale to physical units
		u = u_scaled * self.scaler_X_std[self.u_idx] + self.scaler_X_mean[self.u_idx]
		v = v_scaled * self.scaler_X_std[self.v_idx] + self.scaler_X_mean[self.v_idx]
		r = r_scaled * self.scaler_X_std[self.r_idx] + self.scaler_X_mean[self.r_idx]
		nP = nP_scaled * self.scaler_X_std[self.nP_idx] + self.scaler_X_mean[self.nP_idx]

		# Descale predictions
		predicted_power_kW = predictions[:, 0] * self.scaler_y_std + self.scaler_y_mean
		predicted_power_watts = predicted_power_kW * 1000.0

		# Compute physics-based thrust
		XP = compute_propeller_force_tf(u, v, r, nP, SHIP_PARAMS)
		predicted_thrust = predicted_power_watts / (tf.abs(u) + 1e-6)

		# Physics residual
		physics_residual = tf.reduce_mean(tf.square((XP - predicted_thrust) / 1e6))
		return physics_residual

	@tf.function
	def train_step(self, data):
		x, y = data
		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)
			data_loss = tf.reduce_mean(tf.square(y - y_pred))

			if self.physics_weight > 0:
				physics_loss = self.compute_physics_loss(x, y_pred)
				total_loss = data_loss + self.physics_weight * physics_loss
			else:
				total_loss = data_loss

		gradients = tape.gradient(total_loss, self.trainable_variables)
		gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return {'loss': total_loss}


def prepare_data(df, target_col, n_lags):
	"""Prepare data with proper splitting"""
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

	return splits, scaler_X, scaler_y, feature_cols


def train_single_model(splits, scaler_X, scaler_y, feature_cols, architecture, batch_size, physics_weight, config):
	"""Train a single model configuration"""
	model = MLPWithPhysics(architecture, config['dropout_rate'], physics_weight,
	                       scaler_X, scaler_y, feature_cols, config['n_lags'])
	model.compile(optimizer=model.optimizer, loss='mse')

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
		'physics_weight': physics_weight,
		'train': evaluate(splits['X_train'], splits['y_train']),
		'val': evaluate(splits['X_val'], splits['y_val']),
		'test': evaluate(splits['X_test'], splits['y_test'])
	}


def run_optimization(data, target_col, config):
	"""Run optimization with early pruning"""
	print(f"\n{'='*80}")
	print(f"MLP OPTIMIZATION WITH EARLY PRUNING")
	print(f"Dataset: REGULAR | Prune threshold: R² < {config['prune_threshold']}")
	print(f"{'='*80}")

	splits, scaler_X, scaler_y, feature_cols = prepare_data(data, target_col, config['n_lags'])

	print(f"\nTrain: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
	print(f"\nTesting {len(config['architectures'])} architectures × {len(config['batch_sizes'])} batch sizes × {len(config['physics_weights'])} physics weights")

	results = []
	pruned_architectures = set()
	pruned_physics_weights = set()
	tested = 0
	skipped = 0

	# Phase 1: Test each physics weight with simplest architecture and smallest batch
	print(f"\n{'='*80}")
	print("PHASE 1: Physics Weight Screening")
	print(f"{'='*80}")

	test_arch = config['architectures'][-1]  # Simplest architecture (e.g., [64])
	test_batch = config['batch_sizes'][0]    # Smallest batch size

	for phys_weight in config['physics_weights']:
		tested += 1
		print(f"\n[Screening {tested}/{len(config['physics_weights'])}] Physics weight: {phys_weight}, arch={test_arch}, batch={test_batch}")

		result = train_single_model(splits, scaler_X, scaler_y, feature_cols,
		                            test_arch, test_batch, phys_weight, config)
		results.append(result)

		val_r2 = result['val']['r2']
		print(f"  Val R²: {val_r2:.4f}, Test R²: {result['test']['r2']:.4f}")

		if val_r2 < config['prune_threshold']:
			pruned_physics_weights.add(phys_weight)
			print(f"  ⚠️  PRUNED: Physics weight {phys_weight} (Val R² < {config['prune_threshold']})")

	# Phase 2: Test architectures with remaining physics weights
	print(f"\n{'='*80}")
	print("PHASE 2: Architecture & Batch Size Optimization")
	print(f"{'='*80}")

	active_physics = [pw for pw in config['physics_weights'] if pw not in pruned_physics_weights]
	print(f"\nActive physics weights: {active_physics}")

	for phys_weight in active_physics:
		for arch in config['architectures']:
			# Test with smallest batch size first
			first_batch = config['batch_sizes'][0]

			tested += 1
			print(f"\n[{tested}] arch={arch}, batch={first_batch}, physics={phys_weight}")

			result = train_single_model(splits, scaler_X, scaler_y, feature_cols,
			                            arch, first_batch, phys_weight, config)
			results.append(result)

			val_r2 = result['val']['r2']
			print(f"  Val R²: {val_r2:.4f}, Test R²: {result['test']['r2']:.4f}")

			# Early pruning: if first batch performs poorly, skip other batches
			if val_r2 < config['prune_threshold']:
				pruned_architectures.add((arch, phys_weight))
				skipped_batches = len(config['batch_sizes']) - 1
				skipped += skipped_batches
				print(f"  ⚠️  PRUNED: Skipping {skipped_batches} remaining batch sizes (Val R² < {config['prune_threshold']})")
				continue

			# Test remaining batch sizes
			for batch_size in config['batch_sizes'][1:]:
				tested += 1
				print(f"\n[{tested}] arch={arch}, batch={batch_size}, physics={phys_weight}")

				result = train_single_model(splits, scaler_X, scaler_y, feature_cols,
				                            arch, batch_size, phys_weight, config)
				results.append(result)

				print(f"  Val R²: {result['val']['r2']:.4f}, Test R²: {result['test']['r2']:.4f}")

	# Summary
	print(f"\n{'='*80}")
	print("OPTIMIZATION COMPLETE")
	print(f"{'='*80}")
	print(f"Tested: {tested} configurations")
	print(f"Skipped: {skipped} configurations (early pruning)")
	print(f"Total results: {len(results)}")

	if len(pruned_physics_weights) > 0:
		print(f"\nPruned physics weights: {sorted(pruned_physics_weights)}")
	if len(pruned_architectures) > 0:
		print(f"Pruned architectures: {len(pruned_architectures)} configurations")

	# Find best configuration
	best_result = max(results, key=lambda x: x['test']['r2'])
	print(f"\n{'='*80}")
	print("BEST CONFIGURATION:")
	print(f"{'='*80}")
	print(f"  Architecture: {best_result['architecture']}")
	print(f"  Batch size: {best_result['batch_size']}")
	print(f"  Physics weight: {best_result['physics_weight']}")
	print(f"  Test R²: {best_result['test']['r2']:.4f}")
	print(f"  Test MSE: {best_result['test']['mse']:.2f}")
	print(f"  Test MAE: {best_result['test']['mae']:.2f}")
	print(f"{'='*80}")

	return {
		'results': results,
		'pruned_physics_weights': list(pruned_physics_weights),
		'pruned_architectures': len(pruned_architectures),
		'tested': tested,
		'skipped': skipped
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
			json.dump(optimization_results, f, indent=2)
		print(f"\nResults saved to: {output_file}")
	else:
		print("ERROR: Run pre_process.py first!")
