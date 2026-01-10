"""
RNN with Physics-Informed Neural Networks - OPTIMIZATION VERSION
Tests multiple architectures, batch sizes, and physics weights
CORRECTED: No data leakage - splits before scaling, direct column indices
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
from carbontracker.tracker import CarbonTracker
import matplotlib.pyplot as plt


# Configuration
CONFIG = {
	'output_dir': 'model_performance',
	'architectures': [[64], [64, 32], [128, 64, 32]],
	'batch_sizes': [16, 32],
	'physics_weights': [0.0, 0.001, 0.01],
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

SHIP_PARAMS = {
	'DP': 6.5, 'k0': 0.5453, 'k1': -0.4399, 'k2': -0.0379,
	'tP': 0.1, 'wP0': 0.16, 'xP_prime': -0.5, 'L': 214.0
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

def plot_results_4_panel(y_true, y_pred, history, split_name, model_name, output_dir):
	"""Create 4-panel plot: loss curves, actual vs predicted, errors, error distribution"""
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))

	# Panel 1: Training and Validation Loss
	axes[0, 0].plot(history['loss'], label='Train Loss', linewidth=2)
	axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
	axes[0, 0].set_xlabel('Epoch')
	axes[0, 0].set_ylabel('Total Loss')
	axes[0, 0].set_title('Training and Validation Loss')
	axes[0, 0].legend()
	axes[0, 0].grid(True, alpha=0.3)

	# Panel 2: Actual vs Predicted (first 1000 points)
	n_points = min(1000, len(y_true))
	axes[0, 1].plot(y_true[:n_points], label='Actual', alpha=0.7, linewidth=1.5)
	axes[0, 1].plot(y_pred[:n_points], label='Predicted', alpha=0.7, linewidth=1.5)
	axes[0, 1].set_xlabel('Time Step')
	axes[0, 1].set_ylabel('Engine Power (kW)')
	axes[0, 1].set_title(f'Actual vs Predicted - {split_name}')
	axes[0, 1].legend()
	axes[0, 1].grid(True, alpha=0.3)

	# Panel 3: Prediction Errors Over Time
	errors = y_true - y_pred
	axes[1, 0].plot(errors[:n_points], color='red', alpha=0.6, linewidth=1)
	axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
	axes[1, 0].set_xlabel('Time Step')
	axes[1, 0].set_ylabel('Prediction Error (kW)')
	axes[1, 0].set_title(f'Prediction Errors - {split_name}')
	axes[1, 0].grid(True, alpha=0.3)

	# Panel 4: Error Distribution
	axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
	axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
	axes[1, 1].set_xlabel('Prediction Error (kW)')
	axes[1, 1].set_ylabel('Frequency')
	axes[1, 1].set_title(f'Error Distribution - {split_name}')
	axes[1, 1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(f"{output_dir}/{model_name}_{split_name}_4panel.png", dpi=300, bbox_inches='tight')
	plt.close()

def compute_propeller_force(u, v, r, nP, params):
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


def create_sequences(X, y, timesteps):
	X_seq, y_seq = [], []
	for i in range(timesteps, len(X)):
		X_seq.append(X[i-timesteps:i])
		y_seq.append(y[i])
	return np.array(X_seq), np.array(y_seq)


class RNN_PINN(keras.Model):
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


class PINNTrainer:
	def __init__(self, model, column_indices, scaler_X, scaler_y, physics_weight, learning_rate):
		self.model = model
		self.u_idx, self.v_idx, self.r_idx, self.nP_idx = column_indices
		self.physics_weight = physics_weight
		self.optimizer = Adam(learning_rate=learning_rate)

		self.scaler_X_mean = tf.constant(scaler_X.mean_, dtype=tf.float32)
		self.scaler_X_std = tf.constant(scaler_X.scale_, dtype=tf.float32)
		self.scaler_y_mean = tf.constant(scaler_y.mean_[0], dtype=tf.float32)
		self.scaler_y_std = tf.constant(scaler_y.scale_[0], dtype=tf.float32)

	def descale_features(self, inputs):
		u_scaled = inputs[:, -1, self.u_idx]
		v_scaled = inputs[:, -1, self.v_idx]
		r_scaled = inputs[:, -1, self.r_idx]
		nP_scaled = inputs[:, -1, self.nP_idx]

		u = u_scaled * self.scaler_X_std[self.u_idx] + self.scaler_X_mean[self.u_idx]
		v = v_scaled * self.scaler_X_std[self.v_idx] + self.scaler_X_mean[self.v_idx]
		r = r_scaled * self.scaler_X_std[self.r_idx] + self.scaler_X_mean[self.r_idx]
		nP = nP_scaled * self.scaler_X_std[self.nP_idx] + self.scaler_X_mean[self.nP_idx]

		return u, v, r, nP

	def compute_physics_loss(self, inputs, predictions):
		u, v, r, nP = self.descale_features(inputs)
		predicted_power_kW = predictions[:, 0] * self.scaler_y_std + self.scaler_y_mean
		predicted_power_watts = predicted_power_kW * 1000.0

		XP = compute_propeller_force(u, v, r, nP, SHIP_PARAMS)
		predicted_thrust = predicted_power_watts / (tf.abs(u) + 1e-6)
		physics_residual = tf.reduce_mean(tf.square((XP - predicted_thrust) / 1e6))

		return physics_residual

	@tf.function
	def train_step(self, X_batch, y_batch):
		with tf.GradientTape() as tape:
			predictions = self.model(X_batch, training=True)
			data_loss = tf.reduce_mean(tf.square(y_batch - predictions))

			if self.physics_weight > 0:
				physics_loss = self.compute_physics_loss(X_batch, predictions)
				total_loss = data_loss + self.physics_weight * physics_loss
			else:
				physics_loss = tf.constant(0.0)
				total_loss = data_loss

		gradients = tape.gradient(total_loss, self.model.trainable_variables)
		gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
		return total_loss, data_loss, physics_loss

	def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size, patience):
		train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
		history = {'loss': [], 'val_loss': []}
		best_val_loss = float('inf')
		patience_counter = 0

		for epoch in range(epochs):
			epoch_losses = []
			for X_batch, y_batch in train_dataset:
				total_loss, _, _ = self.train_step(X_batch, y_batch)
				epoch_losses.append(float(total_loss.numpy()))

			val_pred = self.model(X_val, training=False)
			val_loss = tf.reduce_mean(tf.square(y_val - val_pred)).numpy()

			history['loss'].append(np.mean(epoch_losses))
			history['val_loss'].append(float(val_loss))

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
			else:
				patience_counter += 1

			if patience_counter >= patience:
				break

		return history


def prepare_data(data, target_col, timesteps):
	all_exclude = EXCLUDE_COLS + [target_col]
	numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in all_exclude]

	X = data[feature_cols]
	y = data[target_col]

	valid_mask = ~(X.isna().any(axis=1) | y.isna())
	X = X[valid_mask]
	y = y[valid_mask]

	train_size = int(len(X) * 0.6)
	val_size = int(len(X) * 0.2)

	X_train_raw = X.iloc[:train_size]
	y_train_raw = y.iloc[:train_size]
	X_val_raw = X.iloc[train_size:train_size+val_size]
	y_val_raw = y.iloc[train_size:train_size+val_size]
	X_test_raw = X.iloc[train_size+val_size:]
	y_test_raw = y.iloc[train_size+val_size:]

	scaler_X = StandardScaler()
	scaler_y = StandardScaler()
	scaler_X.fit(X_train_raw)
	scaler_y.fit(y_train_raw.values.reshape(-1, 1))

	X_train_scaled = scaler_X.transform(X_train_raw)
	y_train_scaled = scaler_y.transform(y_train_raw.values.reshape(-1, 1)).flatten()
	X_val_scaled = scaler_X.transform(X_val_raw)
	y_val_scaled = scaler_y.transform(y_val_raw.values.reshape(-1, 1)).flatten()
	X_test_scaled = scaler_X.transform(X_test_raw)
	y_test_scaled = scaler_y.transform(y_test_raw.values.reshape(-1, 1)).flatten()

	X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, timesteps)
	X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, timesteps)
	X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, timesteps)

	u_idx = feature_cols.index('OPC_07_WATER_SPEED')
	v_idx = feature_cols.index('v_ms')
	r_idx = feature_cols.index('GPS_HDG_HEADING_ROT_S')
	nP_idx = feature_cols.index('OPC_40_PROP_RPM_FB')

	splits = {
		'X_train': tf.convert_to_tensor(X_train_seq, dtype=tf.float32),
		'y_train': tf.convert_to_tensor(y_train_seq, dtype=tf.float32),
		'X_val': tf.convert_to_tensor(X_val_seq, dtype=tf.float32),
		'y_val': tf.convert_to_tensor(y_val_seq, dtype=tf.float32),
		'X_test': tf.convert_to_tensor(X_test_seq, dtype=tf.float32),
		'y_test': tf.convert_to_tensor(y_test_seq, dtype=tf.float32)
	}

	return splits, scaler_X, scaler_y, (u_idx, v_idx, r_idx, nP_idx)


def train_single_model(splits, scaler_X, scaler_y, column_indices, architecture, batch_size, physics_weight, config, dataset_name):
	model = RNN_PINN(architecture, config['dropout_rate'])
	trainer = PINNTrainer(model, column_indices, scaler_X, scaler_y, physics_weight, config['learning_rate'])
	history = trainer.fit(splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'],
	                      config['epochs'], batch_size, config['patience'])

	def evaluate(X, y, split_name):
		y_pred = model(X, training=False).numpy().flatten()
		y_original = scaler_y.inverse_transform(y.numpy().reshape(-1, 1)).flatten()
		y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

		# Create 4-panel plot only for test set
		if split_name == 'test':
			model_name = f"RNN_arch{architecture}_batch{batch_size}_physics{physics_weight}_{dataset_name}"
			plot_results_4_panel(y_original, y_pred_original, history, split_name, model_name, config['output_dir'])

		return {
			'r2': float(r2_score(y_original, y_pred_original)),
			'mse': float(mean_squared_error(y_original, y_pred_original)),
			'mae': float(mean_absolute_error(y_original, y_pred_original))
		}

	return {
		'architecture': architecture,
		'batch_size': batch_size,
		'physics_weight': physics_weight,
		'train': evaluate(splits['X_train'], splits['y_train'], 'train'),
		'val': evaluate(splits['X_val'], splits['y_val'], 'val'),
		'test': evaluate(splits['X_test'], splits['y_test'], 'test')
	}


def run_optimization(data, dataset_name, target_col, config):
	print(f"\n{'='*70}")
	print(f"OPTIMIZING RNN ON {dataset_name.upper()} DATASET")
	print(f"{'='*70}")

	splits, scaler_X, scaler_y, column_indices = prepare_data(data, target_col, config['timesteps'])

	print(f"Train samples: {len(splits['X_train'])}")
	print(f"Val samples: {len(splits['X_val'])}")
	print(f"Test samples: {len(splits['X_test'])}")

	results = []
	total_configs = len(config['architectures']) * len(config['batch_sizes']) * len(config['physics_weights'])
	current = 0

	for arch in config['architectures']:
		for batch_size in config['batch_sizes']:
			for physics_weight in config['physics_weights']:
				current += 1
				print(f"\n[{current}/{total_configs}] Training: arch={arch}, batch={batch_size}, physics={physics_weight}")

				result = train_single_model(splits, scaler_X, scaler_y, column_indices,
				                            arch, batch_size, physics_weight, config, dataset_name)
				results.append(result)

				print(f"  Test R²: {result['test']['r2']:.4f}, MSE: {result['test']['mse']:.2f}, MAE: {result['test']['mae']:.2f}")

	return results


if __name__ == "__main__":
	OUTPUT_DIR = "output"
	regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")

	if os.path.exists(regular_path):
		speed_trials_regular = pd.read_csv(regular_path)
		target_col = 'OPC_12_CPP_ENGINE_POWER'

		print("\nRunning optimization ONLY on Regular dataset (no weather features)")
		results_regular = run_optimization(speed_trials_regular, "Regular", target_col, CONFIG)

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_file = f'{CONFIG["output_dir"]}/rnn_optimization_results_{timestamp}.json'
		with open(output_file, 'w') as f:
			json.dump({'regular': results_regular}, f, indent=2)
		print(f"\n{'='*70}")
		print(f"Results saved to: {output_file}")
		print(f"{'='*70}")
	else:
		print("ERROR: Run pre_process.py first!")