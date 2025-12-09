import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from carbontracker.tracker import CarbonTracker


# Configuration
CONFIG = {
	'output_dir': 'model_performance',
	'timesteps': 30,
	'epochs': 20,
	'batch_size': 32,
	'patience': 7,
	'learning_rate': 0.001,
	'physics_weights': [0.0, 0.001, 0.01, 0.1],
	'lstm_config': {
		'layer1_units': 32,
		'layer2_units': 16,
		'dropout_rate': 0.2
	}
}

EXCLUDE_COLS = [
	"OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	"elapsed_seconds", "hour", "minute", "second", "dataset_id",
	"GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	"OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"
]

SHIP_PARAMS = {
	'DP': 6.5, 'k0': 0.5453, 'k1': -0.4399, 'k2': -0.0379,
	'tP': 0.1, 'wP0': 0.16, 'xP_prime': -0.5, 'L': 214.0
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# Physics
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


# Plotting
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


# Model
class LSTM_PINN(keras.Model):
	def __init__(self, lstm_config):
		super().__init__()
		self.lstm1 = LSTM(lstm_config['layer1_units'], return_sequences=True)
		self.dropout1 = Dropout(lstm_config['dropout_rate'])
		self.lstm2 = LSTM(lstm_config['layer2_units'], return_sequences=False)
		self.dropout2 = Dropout(lstm_config['dropout_rate'])
		self.output_layer = Dense(1)

	def call(self, inputs):
		x = self.lstm1(inputs)
		x = self.dropout1(x)
		x = self.lstm2(x)
		x = self.dropout2(x)
		return self.output_layer(x)


# Trainer
class PINNTrainer:
	def __init__(self, model, ship_params, column_mapping, scaler_X, scaler_y, physics_weight, learning_rate):
		self.model = model
		self.ship_params = ship_params
		self.column_mapping = column_mapping
		self.physics_weight = physics_weight
		self.optimizer = Adam(learning_rate=learning_rate)

		self.scaler_X_mean = tf.constant(scaler_X.mean_, dtype=tf.float32)
		self.scaler_X_std = tf.constant(scaler_X.scale_, dtype=tf.float32)
		self.scaler_y_mean = tf.constant(scaler_y.mean_[0], dtype=tf.float32)
		self.scaler_y_std = tf.constant(scaler_y.scale_[0], dtype=tf.float32)

	def descale_features(self, inputs):
		u_scaled = inputs[:, -1, self.column_mapping['u']]
		v_scaled = inputs[:, -1, self.column_mapping['v']]
		r_scaled = inputs[:, -1, self.column_mapping['r']]
		nP_rpm_scaled = inputs[:, -1, self.column_mapping['nP']]

		u = u_scaled * self.scaler_X_std[self.column_mapping['u']] + self.scaler_X_mean[self.column_mapping['u']]
		v = v_scaled * self.scaler_X_std[self.column_mapping['v']] + self.scaler_X_mean[self.column_mapping['v']]
		r = r_scaled * self.scaler_X_std[self.column_mapping['r']] + self.scaler_X_mean[self.column_mapping['r']]
		nP_rpm = nP_rpm_scaled * self.scaler_X_std[self.column_mapping['nP']] + self.scaler_X_mean[self.column_mapping['nP']]
		nP = nP_rpm / 60.0

		return u, v, r, nP

	def compute_physics_loss(self, inputs, predictions):
		if len(self.column_mapping) < 4:
			return tf.constant(0.0)

		try:
			u, v, r, nP = self.descale_features(inputs)
			model_power_kW = predictions[:, 0] * self.scaler_y_std + self.scaler_y_mean

			XP = compute_propeller_force(u, v, r, nP, self.ship_params)
			wP = self.ship_params['wP0'] * tf.exp(-4 * (tf.math.atan2(-v, u) - self.ship_params['xP_prime'] *
			                                            tf.where(tf.abs(u) > 1e-6, r * self.ship_params['L'] / u, 0.0))**2)
			uP = u * (1 - wP)
			physics_power_kW = (XP * uP) / 1000.0

			physics_residual = tf.reduce_mean(tf.square((physics_power_kW - model_power_kW) / 1000.0))
			return physics_residual
		except:
			return tf.constant(0.0)

	@tf.function
	def train_step(self, X_batch, y_batch):
		with tf.GradientTape() as tape:
			predictions = self.model(X_batch, training=True)
			data_loss = tf.reduce_mean(tf.square(y_batch - predictions))

			if self.physics_weight > 0:
				physics_loss = self.compute_physics_loss(X_batch, predictions)
				total_loss = data_loss + self.physics_weight * physics_loss
			else:
				physics_loss = 0.0
				total_loss = data_loss

		gradients = tape.gradient(total_loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
		return total_loss, data_loss, physics_loss

	def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size, patience):
		train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
		history = {'loss': [], 'data_loss': [], 'physics_loss': [], 'val_loss': []}
		best_val_loss = float('inf')
		patience_counter = 0

		for epoch in range(epochs):
			epoch_loss, epoch_data_loss, epoch_physics_loss = [], [], []

			for X_batch, y_batch in train_dataset:
				loss, data_loss, physics_loss = self.train_step(X_batch, y_batch)
				epoch_loss.append(loss.numpy())
				epoch_data_loss.append(data_loss.numpy())
				epoch_physics_loss.append(float(physics_loss) if isinstance(physics_loss, tf.Tensor) else physics_loss)

			val_pred = self.model(X_val, training=False)
			val_loss = tf.reduce_mean(tf.square(y_val - val_pred)).numpy()

			history['loss'].append(np.mean(epoch_loss))
			history['data_loss'].append(np.mean(epoch_data_loss))
			history['physics_loss'].append(np.mean(epoch_physics_loss))
			history['val_loss'].append(val_loss)

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
			else:
				patience_counter += 1

			if patience_counter >= patience:
				break

		return history


# Data preparation
def prepare_data(data, target_col, timesteps):
	all_exclude = EXCLUDE_COLS + [target_col]
	numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in all_exclude]

	X = data[feature_cols]
	y = data[target_col]

	valid_mask = ~(X.isna().any(axis=1) | y.isna())
	X = X[valid_mask]
	y = y[valid_mask]

	column_mapping = {}
	for i, col in enumerate(feature_cols):
		if col == 'v_ms':
			column_mapping['v'] = i
		elif col == 'OPC_07_WATER_SPEED':
			column_mapping['u'] = i
		elif col == 'GPS_HDG_HEADING_ROT_S':
			column_mapping['r'] = i
		elif col == 'OPC_40_PROP_RPM_FB':
			column_mapping['nP'] = i

	scaler_X = StandardScaler()
	scaler_y = StandardScaler()
	X_scaled = scaler_X.fit_transform(X)
	y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

	X_sequences, y_sequences = [], []
	for i in range(timesteps, len(X_scaled)):
		X_sequences.append(X_scaled[i-timesteps:i])
		y_sequences.append(y_scaled[i])

	X_sequences = np.array(X_sequences)
	y_sequences = np.array(y_sequences)

	train_size = int(len(X_sequences) * 0.6)
	validate_size = int(len(X_sequences) * 0.2)

	splits = {
		'X_train': X_sequences[:train_size],
		'y_train': y_sequences[:train_size],
		'X_val': X_sequences[train_size:train_size+validate_size],
		'y_val': y_sequences[train_size:train_size+validate_size],
		'X_test': X_sequences[train_size+validate_size:],
		'y_test': y_sequences[train_size+validate_size:]
	}

	for key in splits:
		splits[key] = tf.convert_to_tensor(splits[key], dtype=tf.float32)

	return splits, scaler_X, scaler_y, column_mapping


def evaluate_model(model, X, y, scaler_y, split_name, model_name):
	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()

	y_pred = model.predict(X, verbose=1).flatten()
	y_original = scaler_y.inverse_transform(y.numpy().reshape(-1, 1)).flatten()
	y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

	metrics = {
		'r2': r2_score(y_original, y_pred_original),
		'mse': mean_squared_error(y_original, y_pred_original),
		'mae': mean_absolute_error(y_original, y_pred_original)
	}

	plot_actual_vs_predicted(y_original, y_pred_original, split_name, model_name)
	plot_prediction_errors(y_original, y_pred_original, split_name, model_name)
	plot_error_variance(y_original, y_pred_original, split_name, model_name)

	tracker.epoch_end()
	return metrics


def train_single_model(splits, scaler_X, scaler_y, column_mapping, physics_weight, dataset_name, config):
	model_name = f"LSTM_PINN_pw{physics_weight}_{dataset_name}"

	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()

	model = LSTM_PINN(config['lstm_config'])
	trainer = PINNTrainer(model, SHIP_PARAMS, column_mapping, scaler_X, scaler_y,
	                      physics_weight, config['learning_rate'])
	history = trainer.fit(splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'],
	                      config['epochs'], config['batch_size'], config['patience'])

	tracker.epoch_end()
	plot_training_loss(history, model_name)

	results = {'physics_weight': physics_weight}
	for split_name in ['train', 'val', 'test']:
		X_key = f'X_{split_name}' if split_name != 'val' else 'X_val'
		y_key = f'y_{split_name}' if split_name != 'val' else 'y_val'
		metrics = evaluate_model(model, splits[X_key], splits[y_key], scaler_y, split_name, model_name)
		results[f'{split_name}_r2'] = metrics['r2']
		results[f'{split_name}_mse'] = metrics['mse']
		results[f'{split_name}_mae'] = metrics['mae']

	return results


def train_models(data, dataset_name, target_col, config):
	splits, scaler_X, scaler_y, column_mapping = prepare_data(data, target_col, config['timesteps'])

	results = []
	for pw in config['physics_weights']:
		result = train_single_model(splits, scaler_X, scaler_y, column_mapping, pw, dataset_name, config)
		results.append(result)
		print(f"\nPhysics Weight: {pw}")
		print(f"Train - R²: {result['train_r2']:.4f}, MSE: {result['train_mse']:.2f}, MAE: {result['train_mae']:.2f}")
		print(f"Val   - R²: {result['val_r2']:.4f}, MSE: {result['val_mse']:.2f}, MAE: {result['val_mae']:.2f}")
		print(f"Test  - R²: {result['test_r2']:.4f}, MSE: {result['test_mse']:.2f}, MAE: {result['test_mae']:.2f}")

	return results


if __name__ == "__main__":
	OUTPUT_DIR = "output"
	regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
	weather_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")

	if os.path.exists(regular_path) and os.path.exists(weather_path):
		speed_trials_regular = pd.read_csv(regular_path)
		speed_trials_weather = pd.read_csv(weather_path)
		target_col = 'OPC_12_CPP_ENGINE_POWER'

		results_regular = train_models(speed_trials_regular, "Regular", target_col, CONFIG)
		results_weather = train_models(speed_trials_weather, "Weather", target_col, CONFIG)
	else:
		print("ERROR: Run pre_process.py first!")