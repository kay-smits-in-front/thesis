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
	"OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
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


def normalize_data(X_train, X_val, X_test):
	scaler = StandardScaler()
	scaler.fit(X_train)
	return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


class MLPWithPhysics(keras.Model):
	def __init__(self, input_dim, physics_weight=0.01):
		super().__init__()
		self.physics_weight = physics_weight
		self.dense1 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))
		self.dense2 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
		self.output_layer = layers.Dense(1)

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		return self.output_layer(x)

	def train_step(self, data):
		x, y = data
		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)
			data_loss = self.compiled_loss(y, y_pred)

			# Physics loss (simplified - using last timestep values)
			# Note: This assumes u, v, r, nP are in the input features
			physics_loss = tf.constant(0.0)  # Placeholder

			total_loss = data_loss + self.physics_weight * physics_loss

		gradients = tape.gradient(total_loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		self.compiled_metrics.update_state(y, y_pred)

		return {m.name: m.result() for m in self.metrics}


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


def train_and_evaluate(dataset, dataset_name, n_lags=60):
	target_col = "OPC_12_CPP_ENGINE_POWER"

	print(f"\n{'='*60}")
	print(f"MLP with Physics Loss - {dataset_name}")
	print(f"{'='*60}")

	X, y, feature_cols = create_multivariate_lag_features(dataset, target_col, n_lags, 10)
	print(f"  Samples: {len(y)}, Features: {X.shape[1]}")

	X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
	X_train_scaled, X_val_scaled, X_test_scaled = normalize_data(X_train, X_val, X_test)

	tracker = CarbonTracker(epochs=1)
	tracker.epoch_start()

	model = MLPWithPhysics(X_train_scaled.shape[1], physics_weight=0.01)
	model.compile(optimizer='adam', loss='mse', metrics=['mae'])

	history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32,
	                    validation_data=(X_val_scaled, y_val), verbose=1)

	tracker.epoch_end()

	model_name = f"MLP_Physics_lags{n_lags}_{dataset_name}"

	for split_name, X_split, y_split in [('train', X_train_scaled, y_train),
	                                     ('val', X_val_scaled, y_val),
	                                     ('test', X_test_scaled, y_test)]:
		y_pred = model.predict(X_split, verbose=1).flatten()
		r2 = r2_score(y_split, y_pred)
		mse = mean_squared_error(y_split, y_pred)
		mae = mean_absolute_error(y_split, y_pred)
		print(f"{split_name.capitalize():5s} - R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
		plot_actual_vs_predicted(y_split, y_pred, split_name, model_name)


OUTPUT_DIR = "output"
regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
weather_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")

if os.path.exists(regular_path) and os.path.exists(weather_path):
	speed_trials_regular = pd.read_csv(regular_path)
	speed_trials_weather = pd.read_csv(weather_path)

	train_and_evaluate(speed_trials_regular, "Regular", n_lags=60)
	train_and_evaluate(speed_trials_weather, "Weather", n_lags=60)
else:
	print("ERROR: Run pre_process.py first!")