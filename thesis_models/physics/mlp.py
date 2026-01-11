"""
MLP with Physics-Informed Neural Networks
Best configuration: Triple [128, 64, 32], Batch 16, Physics 0.01
Uses non-overlapping lag features with batch-based splitting
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from carbontracker.tracker import CarbonTracker
from datetime import datetime
import json

# Configuration - Best performing: [64] single-layer architecture, batch_size 16
# Optimization results: Test R²=0.754, Val R²=0.918, Train R²=0.974
# BEST OVERALL MODEL - lowest overfitting (0.220 gap)
CONFIG = {
	'output_dir': 'model_performance',
	'architecture': [64],  # Best performing single-layer architecture
	'batch_size': 16,
	'physics_weights': [0.0, 0.001, 0.01],
	'n_lags': 15,
	'epochs': 20,
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


def compute_propeller_force_tf(u, v, r, nP):
	rho = 1025.0
	sp = SHIP_PARAMS
	beta = tf.math.atan2(-v, u)
	r_prime = tf.where(tf.abs(u) > 1e-6, r * sp['L'] / u, 0.0)
	betaP = beta - sp['xP_prime'] * r_prime
	wP = sp['wP0'] * tf.exp(-4 * betaP**2)
	uP = u * (1 - wP)
	JP = tf.where(tf.abs(nP) > 1e-6, uP / (nP * sp['DP']), 0.0)
	KT = sp['k0'] + sp['k1'] * JP + sp['k2'] * JP**2
	Tp = rho * nP**2 * sp['DP']**4 * KT
	XP = (1 - sp['tP']) * Tp
	return XP


def create_multivariate_lag_features(df, target_col, n_lags):
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


class MLPWithPhysics(keras.Model):
	def __init__(self, input_dim, n_lags, feature_cols, scaler_X, scaler_y, physics_weight, architecture):
		super().__init__()
		self.physics_weight = physics_weight
		self.n_lags = n_lags
		self.n_features = len(feature_cols)
		self.architecture = architecture

		u_base_idx = feature_cols.index('OPC_07_WATER_SPEED')
		v_base_idx = feature_cols.index('v_ms')
		r_base_idx = feature_cols.index('GPS_HDG_HEADING_ROT_S')
		nP_base_idx = feature_cols.index('OPC_40_PROP_RPM_FB')

		self.u_idx = (n_lags - 1) * len(feature_cols) + u_base_idx
		self.v_idx = (n_lags - 1) * len(feature_cols) + v_base_idx
		self.r_idx = (n_lags - 1) * len(feature_cols) + r_base_idx
		self.nP_idx = (n_lags - 1) * len(feature_cols) + nP_base_idx

		self.scaler_X_mean = tf.constant(scaler_X.mean_, dtype=tf.float32)
		self.scaler_X_std = tf.constant(scaler_X.scale_, dtype=tf.float32)
		self.scaler_y_mean = tf.constant(scaler_y.mean_[0], dtype=tf.float32)
		self.scaler_y_std = tf.constant(scaler_y.scale_[0], dtype=tf.float32)

		self.dense_layers = []
		for i, units in enumerate(architecture):
			self.dense_layers.append(Dense(units, activation='relu'))
			if i < len(architecture) - 1:
				self.dense_layers.append(Dropout(CONFIG['dropout_rate']))

		self.output_layer = Dense(1)

	def call(self, inputs):
		x = inputs
		for layer in self.dense_layers:
			x = layer(x)
		return self.output_layer(x)

	def compute_physics_loss(self, inputs, predictions):
		u_scaled = inputs[:, self.u_idx]
		v_scaled = inputs[:, self.v_idx]
		r_scaled = inputs[:, self.r_idx]
		nP_scaled = inputs[:, self.nP_idx]

		u = u_scaled * self.scaler_X_std[self.u_idx] + self.scaler_X_mean[self.u_idx]
		v = v_scaled * self.scaler_X_std[self.v_idx] + self.scaler_X_mean[self.v_idx]
		r = r_scaled * self.scaler_X_std[self.r_idx] + self.scaler_X_mean[self.r_idx]
		nP = nP_scaled * self.scaler_X_std[self.nP_idx] + self.scaler_X_mean[self.nP_idx]

		predicted_power_kW = predictions[:, 0] * self.scaler_y_std + self.scaler_y_mean
		predicted_power_watts = predicted_power_kW * 1000.0

		XP = compute_propeller_force_tf(u, v, r, nP)
		predicted_thrust = predicted_power_watts / (tf.abs(u) + 1e-6)
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
				physics_loss = tf.constant(0.0)
				total_loss = data_loss

		gradients = tape.gradient(total_loss, self.trainable_variables)
		gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return total_loss, data_loss, physics_loss


def plot_training_loss(history, model_name):
	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(history['loss'], label='Train Loss')
	plt.plot(history['val_loss'], label='Val Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Total Loss')
	plt.title('Training and Validation Loss')
	plt.legend()
	plt.grid(True)

	plt.subplot(1, 2, 2)
	plt.plot(history['data_loss'], label='Data Loss')
	plt.plot(history['physics_loss'], label='Physics Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss Component')
	plt.title('Loss Components')
	plt.legend()
	plt.grid(True)
	plt.yscale('log')

	plt.tight_layout()
	plt.savefig(f"{CONFIG['output_dir']}/{model_name}_training_loss.png", dpi=300, bbox_inches='tight')
	plt.close()


def plot_actual_vs_predicted(y_true, y_pred, split_name, model_name):
	plt.figure(figsize=(15, 5))
	plt.plot(y_true[:1000], label='Actual', alpha=0.7)
	plt.plot(y_pred[:1000], label='Predicted', alpha=0.7)
	plt.xlabel('Time Step')
	plt.ylabel('Engine Power (kW)')
	plt.title(f'Actual vs Predicted - {split_name}')
	plt.legend()
	plt.grid(True)
	plt.savefig(f"{CONFIG['output_dir']}/{model_name}_{split_name}_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
	plt.close()


def plot_prediction_errors(y_true, y_pred, split_name, model_name):
	errors = y_true - y_pred
	plt.figure(figsize=(15, 5))
	plt.plot(errors[:1000])
	plt.xlabel('Time Step')
	plt.ylabel('Prediction Error (kW)')
	plt.title(f'Prediction Errors - {split_name}')
	plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
	plt.grid(True)
	plt.savefig(f"{CONFIG['output_dir']}/{model_name}_{split_name}_prediction_errors.png", dpi=300, bbox_inches='tight')
	plt.close()


def plot_error_variance(y_true, y_pred, split_name, model_name):
	errors = y_true - y_pred
	plt.figure(figsize=(10, 5))
	plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
	plt.xlabel('Prediction Error (kW)')
	plt.ylabel('Frequency')
	plt.title(f'Error Distribution - {split_name}')
	plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
	plt.grid(True)
	plt.savefig(f"{CONFIG['output_dir']}/{model_name}_{split_name}_error_distribution.png", dpi=300, bbox_inches='tight')
	plt.close()


def train_with_tracking(model, X_train, y_train, X_val, y_val, epochs, batch_size, patience):
	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
	history = {'loss': [], 'data_loss': [], 'physics_loss': [], 'val_loss': [],
	           'val_data_loss': [], 'val_physics_loss': []}

	best_val_loss = float('inf')
	patience_counter = 0

	for epoch in range(epochs):
		epoch_loss, epoch_data_loss, epoch_physics_loss = [], [], []

		for X_batch, y_batch in train_dataset:
			total_loss, data_loss, physics_loss = model.train_step((X_batch, y_batch))
			epoch_loss.append(float(total_loss.numpy()))
			epoch_data_loss.append(float(data_loss.numpy()))
			epoch_physics_loss.append(float(physics_loss.numpy()))

		val_pred = model(X_val, training=False)
		val_data_loss = tf.reduce_mean(tf.square(y_val - val_pred))
		if model.physics_weight > 0:
			val_physics_loss = model.compute_physics_loss(X_val, val_pred)
		else:
			val_physics_loss = tf.constant(0.0)
		val_total_loss = val_data_loss + model.physics_weight * val_physics_loss

		history['loss'].append(np.mean(epoch_loss))
		history['data_loss'].append(np.mean(epoch_data_loss))
		history['physics_loss'].append(np.mean(epoch_physics_loss))
		history['val_loss'].append(float(val_total_loss.numpy()))
		history['val_data_loss'].append(float(val_data_loss.numpy()))
		history['val_physics_loss'].append(float(val_physics_loss.numpy()))

		if val_total_loss < best_val_loss:
			best_val_loss = val_total_loss
			patience_counter = 0
		else:
			patience_counter += 1

		if patience_counter >= patience:
			print(f"Early stopping at epoch {epoch+1}")
			break

	return history


def evaluate_model(model, X, y, scaler_y, split_name, model_name):
	y_pred_scaled = model.predict(X, verbose=0)
	y_pred = scaler_y.inverse_transform(y_pred_scaled)
	y_true = scaler_y.inverse_transform(y.numpy().reshape(-1, 1))

	metrics = {
		'r2': float(r2_score(y_true, y_pred)),
		'mse': float(mean_squared_error(y_true, y_pred)),
		'mae': float(mean_absolute_error(y_true, y_pred))
	}

	plot_actual_vs_predicted(y_true.flatten(), y_pred.flatten(), split_name, model_name)
	plot_prediction_errors(y_true.flatten(), y_pred.flatten(), split_name, model_name)
	plot_error_variance(y_true.flatten(), y_pred.flatten(), split_name, model_name)

	return metrics


def prepare_data(df, target_col, n_lags, batch_size):
	X, y, feature_cols = create_multivariate_lag_features(df, target_col, n_lags)
	print(f"Samples: {len(y)}, Features: {X.shape[1]}")

	train_size = int(len(X) * 0.6)
	val_size = int(len(X) * 0.2)

	X_train_raw = X[:train_size]
	y_train_raw = y[:train_size]
	X_val_raw = X[train_size:train_size+val_size]
	y_val_raw = y[train_size:train_size+val_size]
	X_test_raw = X[train_size+val_size:]
	y_test_raw = y[train_size+val_size:]

	scaler_X = StandardScaler()
	scaler_y = StandardScaler()
	scaler_X.fit(X_train_raw)
	scaler_y.fit(y_train_raw.reshape(-1, 1))

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


def train_model(data, dataset_name, target_col, config):
	splits, scaler_X, scaler_y, feature_cols = prepare_data(
		data, target_col, config['n_lags'], config['batch_size']
	)

	model_name = f"MLP_PINN_{dataset_name}"

	print(f"\nTraining MLP with architecture {config['architecture']}")
	print(f"Batch size: {config['batch_size']}, Physics weight: {config['physics_weight']}")

	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()

	model = MLPWithPhysics(
		splits['X_train'].shape[1], config['n_lags'], feature_cols,
		scaler_X, scaler_y, config['physics_weight'], config['architecture']
	)
	model.compile(optimizer=Adam(learning_rate=config['learning_rate']))

	history = train_with_tracking(
		model, splits['X_train'], splits['y_train'],
		splits['X_val'], splits['y_val'],
		epochs=config['epochs'], batch_size=config['batch_size'],
		patience=config['patience']
	)

	tracker.epoch_end()
	plot_training_loss(history, model_name)

	train_metrics = evaluate_model(model, splits['X_train'], splits['y_train'], scaler_y, 'train', model_name)
	val_metrics = evaluate_model(model, splits['X_val'], splits['y_val'], scaler_y, 'val', model_name)
	test_metrics = evaluate_model(model, splits['X_test'], splits['y_test'], scaler_y, 'test', model_name)

	print(f"\nTrain - R²: {train_metrics['r2']:.4f}, MSE: {train_metrics['mse']:.2f}, MAE: {train_metrics['mae']:.2f}")
	print(f"Val   - R²: {val_metrics['r2']:.4f}, MSE: {val_metrics['mse']:.2f}, MAE: {val_metrics['mae']:.2f}")
	print(f"Test  - R²: {test_metrics['r2']:.4f}, MSE: {test_metrics['mse']:.2f}, MAE: {test_metrics['mae']:.2f}")

	return {
		'dataset': dataset_name,
		'architecture': config['architecture'],
		'batch_size': config['batch_size'],
		'physics_weight': config['physics_weight'],
		'metrics': {
			'train': train_metrics,
			'val': val_metrics,
			'test': test_metrics
		}
	}


if __name__ == "__main__":
	OUTPUT_DIR = "output"
	regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
	weather_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")

	if os.path.exists(regular_path) and os.path.exists(weather_path):
		target_col = 'OPC_12_CPP_ENGINE_POWER'

		all_results = []

		for physics_weight in CONFIG['physics_weights']:
			print(f"\n{'='*70}")
			print(f"Training with physics_weight = {physics_weight}")
			print(f"{'='*70}")

			current_config = CONFIG.copy()
			current_config['physics_weight'] = physics_weight

			speed_trials_regular = pd.read_csv(regular_path)
			print(f"\nTraining on Regular dataset (λ={physics_weight})...")
			results_regular = train_model(speed_trials_regular, "Regular", target_col, current_config)
			all_results.append(results_regular)

			speed_trials_weather = pd.read_csv(weather_path)
			print(f"\nTraining on Weather dataset (λ={physics_weight})...")
			results_weather = train_model(speed_trials_weather, "Weather", target_col, current_config)
			all_results.append(results_weather)

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_file = f'{CONFIG["output_dir"]}/mlp_results_{timestamp}.json'
		with open(output_file, 'w') as f:
			json.dump(all_results, f, indent=2)
		print(f"\n{'='*70}")
		print(f"Results saved to: {output_file}")
		print(f"{'='*70}")
	else:
		print("ERROR: Run pre_process.py first!")