import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from carbontracker.tracker import CarbonTracker

os.makedirs('model_performance', exist_ok=True)

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


def plot_actual_vs_predicted(y_true, y_pred, split_name, model_name):
	plt.figure(figsize=(15, 5))
	plt.plot(y_true[:1000], label='Actual', alpha=0.7)
	plt.plot(y_pred[:1000], label='Predicted', alpha=0.7)
	plt.xlabel('Time Step')
	plt.ylabel('Engine Power (kW)')
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
	plt.ylabel('Prediction Error (kW)')
	plt.title(f'Prediction Errors - {split_name}')
	plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
	plt.grid(True)
	plt.savefig(f'model_performance/{model_name}_{split_name}_prediction_errors.png', dpi=300, bbox_inches='tight')
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
	plt.savefig(f'model_performance/{model_name}_{split_name}_error_distribution.png', dpi=300, bbox_inches='tight')
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
	plt.savefig(f'model_performance/{model_name}_training_loss.png', dpi=300, bbox_inches='tight')
	plt.close()


def compute_propeller_force_single_tf(u, v, r, nP, DP, k0, k1, k2, tP, wP0, xP_prime, L):
	"""Compute propeller surge force - TensorFlow version."""
	rho = 1025.0
	beta = tf.math.atan2(-v, u)
	r_prime = tf.where(tf.abs(u) > 1e-6, r * L / u, 0.0)
	betaP = beta - xP_prime * r_prime
	wP = wP0 * tf.exp(-4 * betaP**2)
	uP = u * (1 - wP)
	JP = tf.where(tf.abs(nP) > 1e-6, uP / (nP * DP), 0.0)
	KT = k0 + k1 * JP + k2 * JP**2
	Tp = rho * nP**2 * DP**4 * KT
	XP = (1 - tP) * Tp
	return XP


class RNN_PINN(keras.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.rnn1 = SimpleRNN(64, return_sequences=True)
		self.dropout1 = Dropout(0.2)
		self.rnn2 = SimpleRNN(32, return_sequences=True)
		self.dropout2 = Dropout(0.2)
		self.rnn3 = SimpleRNN(16, return_sequences=False)
		self.dropout3 = Dropout(0.2)
		self.output_layer = Dense(1)

	def call(self, inputs):
		x = self.rnn1(inputs)
		x = self.dropout1(x)
		x = self.rnn2(x)
		x = self.dropout2(x)
		x = self.rnn3(x)
		x = self.dropout3(x)
		return self.output_layer(x)


class PINNTrainer:
	def __init__(self, model, ship_params, column_mapping, scaler_X, scaler_y, physics_weight=0.0):
		self.model = model
		self.ship_params = ship_params
		self.column_mapping = column_mapping
		self.physics_weight = physics_weight
		self.optimizer = Adam(learning_rate=0.0001)

		# Store scaler parameters
		self.scaler_X_mean = tf.constant(scaler_X.mean_, dtype=tf.float32)
		self.scaler_X_std = tf.constant(scaler_X.scale_, dtype=tf.float32)
		self.scaler_y_mean = tf.constant(scaler_y.mean_[0], dtype=tf.float32)
		self.scaler_y_std = tf.constant(scaler_y.scale_[0], dtype=tf.float32)

	def compute_physics_loss(self, inputs, predictions):
		if len(self.column_mapping) < 4:
			return tf.constant(0.0)

		try:
			# Extract SCALED values
			u_scaled = inputs[:, -1, self.column_mapping['u']]
			v_scaled = inputs[:, -1, self.column_mapping['v']]
			r_scaled = inputs[:, -1, self.column_mapping['r']]
			nP_scaled = inputs[:, -1, self.column_mapping['nP']]

			# INVERSE TRANSFORM to real physical units
			u = u_scaled * self.scaler_X_std[self.column_mapping['u']] + self.scaler_X_mean[self.column_mapping['u']]
			v = v_scaled * self.scaler_X_std[self.column_mapping['v']] + self.scaler_X_mean[self.column_mapping['v']]
			r = r_scaled * self.scaler_X_std[self.column_mapping['r']] + self.scaler_X_mean[self.column_mapping['r']]
			nP = nP_scaled * self.scaler_X_std[self.column_mapping['nP']] + self.scaler_X_mean[self.column_mapping['nP']]

			# Inverse transform predictions to get real power (kW)
			predicted_power_kW = predictions[:, 0] * self.scaler_y_std + self.scaler_y_mean
			predicted_power_watts = predicted_power_kW * 1000.0

			sp = self.ship_params
			XP = compute_propeller_force_single_tf(
				u, v, r, nP,
				sp['DP'], sp['k0'], sp['k1'], sp['k2'],
				sp['tP'], sp['wP0'], sp['xP_prime'], sp['L']
			)

			predicted_thrust = predicted_power_watts / (tf.abs(u) + 1e-6)
			physics_residual = tf.reduce_mean(tf.square((XP - predicted_thrust) / 1e6))

			return physics_residual
		except Exception as e:
			print(f"Physics loss error: {e}")
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

	def fit(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=64, patience=5):
		train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
		train_dataset = train_dataset.batch(batch_size)

		history = {'loss': [], 'data_loss': [], 'physics_loss': [], 'val_loss': []}

		best_val_loss = float('inf')
		patience_counter = 0

		for epoch in range(epochs):
			epoch_loss = []
			epoch_data_loss = []
			epoch_physics_loss = []

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

			if (epoch + 1) % 10 == 0 or epoch == 0:
				print(f"Epoch {epoch+1}/{epochs} - Loss: {history['loss'][-1]:.4f}, "
				      f"Data Loss: {history['data_loss'][-1]:.4f}, "
				      f"Physics Loss: {history['physics_loss'][-1]:.6f}, "
				      f"Val Loss: {val_loss:.4f}")

			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
			else:
				patience_counter += 1

			if patience_counter >= patience:
				print(f"Early stopping at epoch {epoch+1}")
				break

		return history


def train_models(data, dataset_name, target_col, ship_params, timesteps=30):
	"""Train RNN PINN models with sequences"""
	print("\n" + "="*70)
	print(f"TRAINING RNN PINN - {dataset_name}")
	print("="*70)

	# Select features (exclude target and unwanted columns)
	all_exclude = EXCLUDE_COLS + [target_col]
	numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in all_exclude]

	print(f"\nFeature columns: {len(feature_cols)}")

	X = data[feature_cols]
	y = data[target_col]

	# Drop NaN
	valid_mask = ~(X.isna().any(axis=1) | y.isna())
	X = X[valid_mask]
	y = y[valid_mask]

	print(f"Data shape after cleaning: X={X.shape}, y={y.shape}")

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

	print(f"Physics column mapping: {column_mapping}")

	# Scale data
	scaler_X = StandardScaler()
	scaler_y = StandardScaler()

	X_scaled = scaler_X.fit_transform(X)
	y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

	# Create sequences
	print(f"Creating sequences with {timesteps} timesteps...")
	X_sequences = []
	y_sequences = []

	for i in range(timesteps, len(X_scaled)):
		X_sequences.append(X_scaled[i-timesteps:i])
		y_sequences.append(y_scaled[i])

	X_sequences = np.array(X_sequences)
	y_sequences = np.array(y_sequences)

	print(f"Sequence shape: X={X_sequences.shape}, y={y_sequences.shape}")

	# Split data
	print("Splitting data into train/validate/test (60/20/20)...")
	train_size = int(len(X_sequences) * 0.6)
	validate_size = int(len(X_sequences) * 0.2)

	X_train = X_sequences[:train_size]
	y_train = y_sequences[:train_size]

	X_validate = X_sequences[train_size:train_size+validate_size]
	y_validate = y_sequences[train_size:train_size+validate_size]

	X_test = X_sequences[train_size+validate_size:]
	y_test = y_sequences[train_size+validate_size:]

	print(f"Train: {len(X_train)}, Val: {len(X_validate)}, Test: {len(X_test)}")

	# Convert to tensors
	X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
	y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
	X_validate = tf.convert_to_tensor(X_validate, dtype=tf.float32)
	y_validate = tf.convert_to_tensor(y_validate, dtype=tf.float32)
	X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
	y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

	results = []

	# Train models with different physics weights
	for pw in [0.0, 0.001, 0.01]:
		print("\n" + "="*70)
		print(f"TRAINING WITH PHYSICS_WEIGHT = {pw}")
		print("="*70)

		tracker = CarbonTracker(epochs=1)
		tracker.epoch_start()

		model = RNN_PINN()
		trainer = PINNTrainer(model, ship_params, column_mapping, scaler_X, scaler_y, physics_weight=pw)
		history = trainer.fit(X_train, y_train, X_validate, y_validate,
		                      epochs=20, batch_size=32, patience=5)

		tracker.epoch_end()

		model_name = f"RNN_PINN_pw{pw}_{dataset_name}"
		plot_training_loss(history, model_name)

		# Evaluate on TRAIN set
		print("\nEvaluating on train set...")
		tracker = CarbonTracker(epochs=1)
		tracker.epoch_start()
		y_train_pred = model.predict(X_train, verbose=1).flatten()
		y_train_original = scaler_y.inverse_transform(y_train.numpy().reshape(-1, 1)).flatten()
		y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()

		r2_train = r2_score(y_train_original, y_train_pred_original)
		mse_train = mean_squared_error(y_train_original, y_train_pred_original)
		mae_train = mean_absolute_error(y_train_original, y_train_pred_original)

		plot_actual_vs_predicted(y_train_original, y_train_pred_original, 'train', model_name)
		plot_prediction_errors(y_train_original, y_train_pred_original, 'train', model_name)
		plot_error_variance(y_train_original, y_train_pred_original, 'train', model_name)
		tracker.epoch_end()

		# Evaluate on VAL set
		print("\nEvaluating on validation set...")
		tracker = CarbonTracker(epochs=1)
		tracker.epoch_start()
		y_val_pred = model.predict(X_validate, verbose=1).flatten()
		y_val_original = scaler_y.inverse_transform(y_validate.numpy().reshape(-1, 1)).flatten()
		y_val_pred_original = scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()

		r2_val = r2_score(y_val_original, y_val_pred_original)
		mse_val = mean_squared_error(y_val_original, y_val_pred_original)
		mae_val = mean_absolute_error(y_val_original, y_val_pred_original)

		plot_actual_vs_predicted(y_val_original, y_val_pred_original, 'val', model_name)
		plot_prediction_errors(y_val_original, y_val_pred_original, 'val', model_name)
		plot_error_variance(y_val_original, y_val_pred_original, 'val', model_name)
		tracker.epoch_end()

		# Evaluate on TEST set
		print("\nEvaluating on test set...")
		tracker = CarbonTracker(epochs=1)
		tracker.epoch_start()
		y_test_pred = model.predict(X_test, verbose=1).flatten()
		y_test_original = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
		y_test_pred_original = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

		r2_test = r2_score(y_test_original, y_test_pred_original)
		mse_test = mean_squared_error(y_test_original, y_test_pred_original)
		mae_test = mean_absolute_error(y_test_original, y_test_pred_original)

		plot_actual_vs_predicted(y_test_original, y_test_pred_original, 'test', model_name)
		plot_prediction_errors(y_test_original, y_test_pred_original, 'test', model_name)
		plot_error_variance(y_test_original, y_test_pred_original, 'test', model_name)
		tracker.epoch_end()

		print(f"\nRNN PINN (physics_weight={pw}) - {dataset_name}:")
		print(f"Train - R²: {r2_train:.4f}, MSE: {mse_train:.4f}, MAE: {mae_train:.4f}")
		print(f"Val   - R²: {r2_val:.4f}, MSE: {mse_val:.4f}, MAE: {mae_val:.4f}")
		print(f"Test  - R²: {r2_test:.4f}, MSE: {mse_test:.4f}, MAE: {mae_test:.4f}")

		results.append({
			'physics_weight': pw,
			'train_r2': r2_train, 'train_mse': mse_train, 'train_mae': mae_train,
			'val_r2': r2_val, 'val_mse': mse_val, 'val_mae': mae_val,
			'test_r2': r2_test, 'test_mse': mse_test, 'test_mae': mae_test
		})

	# Summary
	print("\n" + "="*70)
	print(f"RESULTS SUMMARY - {dataset_name}")
	print("="*70)
	for res in results:
		print(f"\nPhysics Weight: {res['physics_weight']:.5f}")
		print(f"  Train - R²: {res['train_r2']:.4f}, MSE: {res['train_mse']:.2f}, MAE: {res['train_mae']:.2f}")
		print(f"  Val   - R²: {res['val_r2']:.4f}, MSE: {res['val_mse']:.2f}, MAE: {res['val_mae']:.2f}")
		print(f"  Test  - R²: {res['test_r2']:.4f}, MSE: {res['test_mse']:.2f}, MAE: {res['test_mae']:.2f}")

	return results


# Load data and run
OUTPUT_DIR = "output"
regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
weather_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")

if os.path.exists(regular_path) and os.path.exists(weather_path):
	print("Loading preprocessed data...")
	speed_trials_regular = pd.read_csv(regular_path)
	speed_trials_weather = pd.read_csv(weather_path)

	print(f"Loaded SPEED_TRIALS_REGULAR_FINAL: {speed_trials_regular.shape}")
	print(f"Loaded SPEED_TRIALS_WEATHER_FINAL: {speed_trials_weather.shape}")

	target_col = 'OPC_12_CPP_ENGINE_POWER'

	# Train on regular data
	results_regular = train_models(speed_trials_regular, "Regular", target_col, SHIP_PARAMS, timesteps=30)

	# Train on weather data
	results_weather = train_models(speed_trials_weather, "Weather", target_col, SHIP_PARAMS, timesteps=30)
else:
	print("ERROR: Run pre_process.py first!")