import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from carbontracker.tracker import CarbonTracker

os.makedirs('model_performance', exist_ok=True)

# Ship parameters for physics loss
SHIP_PARAMS = {
	'DP': 6.5, 'k0': 0.5453, 'k1': -0.4399, 'k2': -0.0379,
	'tP': 0.1, 'wP0': 0.16, 'xP_prime': -0.5, 'L': 214.0
}

EXCLUDE_COLS = [
	"OPC_12_CPP_ENGINE_POWER",  # Target
	"OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	"elapsed_seconds", "hour", "minute", "second", "dataset_id",
	"GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	"OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"
]


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


def create_multivariate_lag_features(df, target_col, n_lags, forecast_horizon):
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in EXCLUDE_COLS]

	print(f"  Feature columns: {len(feature_cols)}")

	X_list, y_list = [], []
	for i in range(n_lags, len(df) - forecast_horizon):
		features = []
		for col in feature_cols:
			features.extend(df[col].iloc[i - n_lags:i].values)
		X_list.append(features)
		y_list.append(df[target_col].iloc[i + forecast_horizon])

	return np.array(X_list), np.array(y_list), feature_cols


def split_data(X, y, train_ratio=0.6, val_ratio=0.2):
	train_idx = int(len(X) * train_ratio)
	val_idx = int(len(X) * (train_ratio + val_ratio))
	return (X[:train_idx], X[train_idx:val_idx], X[val_idx:],
	        y[:train_idx], y[train_idx:val_idx], y[val_idx:])


class MLPWithPhysics(keras.Model):
	def __init__(self, input_dim, n_lags, feature_cols, scaler_X, scaler_y, physics_weight=0.01):
		super().__init__()
		self.physics_weight = physics_weight
		self.n_lags = n_lags
		self.n_features = len(feature_cols)

		# FIXED: Initialize column_mapping BEFORE using it
		self.column_mapping = {}

		# Map to the LAST timestep in the lagged features
		for i, col in enumerate(feature_cols):
			if col == 'v_ms':
				self.column_mapping['v'] = (n_lags - 1) * len(feature_cols) + i
			elif col == 'OPC_07_WATER_SPEED':
				self.column_mapping['u'] = (n_lags - 1) * len(feature_cols) + i
			elif col == 'GPS_HDG_HEADING_ROT_S':
				self.column_mapping['r'] = (n_lags - 1) * len(feature_cols) + i
			elif col == 'OPC_40_PROP_RPM_FB':
				self.column_mapping['nP'] = (n_lags - 1) * len(feature_cols) + i

		print(f"Column mapping for physics: {self.column_mapping}")

		# Store scaler parameters
		self.scaler_X_mean = tf.constant(scaler_X.mean_, dtype=tf.float32)
		self.scaler_X_std = tf.constant(scaler_X.scale_, dtype=tf.float32)
		self.scaler_y_mean = tf.constant(scaler_y.mean_[0], dtype=tf.float32)
		self.scaler_y_std = tf.constant(scaler_y.scale_[0], dtype=tf.float32)

		self.dense1 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
		self.dense2 = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))
		self.dropout = layers.Dropout(0.2)
		self.output_layer = layers.Dense(1)

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dropout(x)
		return self.output_layer(x)

	def compute_physics_loss(self, inputs, predictions):
		if len(self.column_mapping) < 4:
			return tf.constant(0.0)

		try:
			# FIXED: Input is already flattened, no need to reshape
			# Extract SCALED values directly from flattened input
			u_scaled = inputs[:, self.column_mapping['u']]
			v_scaled = inputs[:, self.column_mapping['v']] if 'v' in self.column_mapping else tf.zeros_like(inputs[:, 0])
			r_scaled = inputs[:, self.column_mapping['r']]
			nP_scaled = inputs[:, self.column_mapping['nP']]

			# INVERSE TRANSFORM to real physical units
			u = u_scaled * self.scaler_X_std[self.column_mapping['u']] + self.scaler_X_mean[self.column_mapping['u']]
			v = v_scaled * self.scaler_X_std[self.column_mapping['v']] + self.scaler_X_mean[self.column_mapping['v']] if 'v' in self.column_mapping else v_scaled
			r = r_scaled * self.scaler_X_std[self.column_mapping['r']] + self.scaler_X_mean[self.column_mapping['r']]
			nP = nP_scaled * self.scaler_X_std[self.column_mapping['nP']] + self.scaler_X_mean[self.column_mapping['nP']]

			# Inverse transform predictions to get real power (kW)
			predicted_power_kW = predictions[:, 0] * self.scaler_y_std + self.scaler_y_mean
			predicted_power_watts = predicted_power_kW * 1000.0

			XP = compute_propeller_force_tf(u, v, r, nP)

			predicted_thrust = predicted_power_watts / (tf.abs(u) + 1e-6)
			physics_residual = tf.reduce_mean(tf.square((XP - predicted_thrust) / 1e6))

			return physics_residual
		except Exception as e:
			print(f"Physics loss error: {e}")
			return tf.constant(0.0)

	def train_step(self, data):
		x, y = data
		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)
			data_loss = self.compiled_loss(y, y_pred)

			if self.physics_weight > 0:
				physics_loss = self.compute_physics_loss(x, y_pred)
				total_loss = data_loss + self.physics_weight * physics_loss
			else:
				physics_loss = tf.constant(0.0)  # CHANGED: Use tf.constant instead of 0.0
				total_loss = data_loss

		gradients = tape.gradient(total_loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		self.compiled_metrics.update_state(y, y_pred)

		# IMPORTANT: Must return a dict with metric results
		results = {m.name: m.result() for m in self.metrics}
		results['loss'] = total_loss  # Add total loss to results
		return results

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


def train_and_evaluate(dataset, dataset_name, n_lags=30):
	target_col = "OPC_12_CPP_ENGINE_POWER"

	print(f"\n{'='*60}")
	print(f"MLP with Physics Loss - {dataset_name}")
	print(f"{'='*60}")

	X, y, feature_cols = create_multivariate_lag_features(dataset, target_col, n_lags, 10)
	print(f"  Samples: {len(y)}, Features: {X.shape[1]}")

	X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

	# Create scalers
	scaler_X = StandardScaler()
	scaler_y = StandardScaler()

	scaler_X.fit(X_train)
	X_train_scaled = scaler_X.transform(X_train)
	X_val_scaled = scaler_X.transform(X_val)
	X_test_scaled = scaler_X.transform(X_test)

	scaler_y.fit(y_train.reshape(-1, 1))
	y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
	y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

	# Train models with different physics weights
	for pw in [0.0]:
		print(f"\n{'='*60}")
		print(f"Training with physics_weight={pw}")
		print(f"{'='*60}")

		tracker = CarbonTracker(epochs=1)
		tracker.epoch_start()

		model = MLPWithPhysics(X_train_scaled.shape[1], n_lags, feature_cols, scaler_X, scaler_y, physics_weight=pw)
		model.compile(optimizer='adam', loss='mse', metrics=['mae'])

		# Custom callback to track physics loss
		class PhysicsLossCallback(keras.callbacks.Callback):
			def __init__(self, model_ref):
				super().__init__()
				self.model_ref = model_ref
				self.physics_losses = []

			def on_epoch_end(self, epoch, logs=None):
				# Compute physics loss on a sample batch
				sample_batch = X_train_scaled[:64]
				sample_pred = self.model_ref(sample_batch, training=False)
				phys_loss = self.model_ref.compute_physics_loss(
					tf.constant(sample_batch, dtype=tf.float32),
					sample_pred
				)
				self.physics_losses.append(float(phys_loss.numpy()) if isinstance(phys_loss, tf.Tensor) else float(phys_loss))

				if (epoch + 1) % 10 == 0 or epoch == 0:
					print(f"Epoch {epoch+1} - Physics Loss: {self.physics_losses[-1]:.6f}")

		physics_callback = PhysicsLossCallback(model)

		history = model.fit(X_train_scaled, y_train_scaled, epochs=10, batch_size=32,
		                    validation_data=(X_val_scaled, y_val_scaled),
		                    callbacks=[physics_callback],
		                    verbose=1)

		tracker.epoch_end()

		model_name = f"MLP_Physics_pw{pw}_lags{n_lags}_{dataset_name}"

		# Plot training loss with physics component
		plt.figure(figsize=(12, 5))

		plt.subplot(1, 2, 1)
		plt.plot(history.history['loss'], label='Train Loss')
		plt.plot(history.history['val_loss'], label='Val Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Total Loss')
		plt.title('Training and Validation Loss')
		plt.legend()
		plt.grid(True)

		plt.subplot(1, 2, 2)
		if pw > 0:
			plt.plot(physics_callback.physics_losses, label='Physics Loss')
			plt.xlabel('Epoch')
			plt.ylabel('Physics Loss')
			plt.title('Physics Loss Over Time')
			plt.legend()
			plt.grid(True)
			plt.yscale('log')

		plt.tight_layout()
		plt.savefig(f'model_performance/{model_name}_training_loss.png', dpi=300, bbox_inches='tight')
		plt.close()

		# Evaluate
		for split_name, X_split, y_split_real in [
			('train', X_train_scaled, y_train),
			('val', X_val_scaled, y_val),
			('test', X_test_scaled, y_test)
		]:
			y_pred_scaled = model.predict(X_split, verbose=0).flatten()
			y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

			r2 = r2_score(y_split_real, y_pred_real)
			mse = mean_squared_error(y_split_real, y_pred_real)
			mae = mean_absolute_error(y_split_real, y_pred_real)
			print(f"{split_name.capitalize():5s} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
			plot_actual_vs_predicted(y_split_real, y_pred_real, split_name, model_name)


OUTPUT_DIR = "output"
regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
weather_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")

if os.path.exists(regular_path) and os.path.exists(weather_path):
	speed_trials_regular = pd.read_csv(regular_path)
	speed_trials_weather = pd.read_csv(weather_path)

	train_and_evaluate(speed_trials_regular, "Regular", n_lags=30)
	train_and_evaluate(speed_trials_weather, "Weather", n_lags=30)
else:
	print("ERROR: Run pre_process.py first!")