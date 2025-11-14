""""Physics-Informed Neural Network (PINN) for solving a simple ODE and measuring training speed."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR

speed_trials = SPEED_TRIALS_REGULAR

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


class LSTM_PINN(keras.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.lstm1 = keras.layers.LSTM(64, return_sequences=True)
		self.dropout1 = keras.layers.Dropout(0.2)
		self.lstm2 = keras.layers.LSTM(32, return_sequences=True)
		self.dropout2 = keras.layers.Dropout(0.2)
		self.lstm3 = keras.layers.LSTM(16, return_sequences=False)
		self.dropout3 = keras.layers.Dropout(0.2)
		self.output_layer = keras.layers.Dense(1)

	def call(self, inputs):
		x = self.lstm1(inputs)
		x = self.dropout1(x)
		x = self.lstm2(x)
		x = self.dropout2(x)
		x = self.lstm3(x)
		x = self.dropout3(x)
		return self.output_layer(x)


class PINNTrainer:
	def __init__(self, model, ship_params, column_mapping, physics_weight=0.0):
		self.model = model
		self.ship_params = ship_params
		self.column_mapping = column_mapping
		self.physics_weight = physics_weight
		self.optimizer = Adam(learning_rate=0.0001)

	def compute_physics_loss(self, inputs, predictions):
		u = inputs[:, -1, self.column_mapping['u']]
		v = inputs[:, -1, self.column_mapping['v']]
		r = inputs[:, -1, self.column_mapping['r']]
		nP = inputs[:, -1, self.column_mapping['nP']]

		predicted_power_watts = predictions[:, 0] * 1000.0

		sp = self.ship_params
		XP = compute_propeller_force_single_tf(
			u, v, r, nP,
			sp['DP'], sp['k0'], sp['k1'], sp['k2'],
			sp['tP'], sp['wP0'], sp['xP_prime'], sp['L']
		)

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
				physics_loss = 0.0
				total_loss = data_loss

		gradients = tape.gradient(total_loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
		return total_loss, data_loss, physics_loss

	def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, patience=5):
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

			if (epoch + 1) % 10 == 0:
				print(f"Epoch {epoch+1}/{epochs} - Loss: {history['loss'][-1]:.4f}, "
				      f"Data Loss: {history['data_loss'][-1]:.4f}, "
				      f"Physics Loss: {history['physics_loss'][-1]:.4f}, "
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

def prepare_data_with_unit_conversions(speed_trials, target_col='OPC_12_CPP_ENGINE_POWER'):
	"""
	Prepare data with unit conversions
	"""
	print("="*70)
	print("PREPARING DATA WITH UNIT CONVERSIONS")
	print("="*70)

	data_converted = speed_trials.copy()

	data_converted['u_ms'] = speed_trials['OPC_07_WATER_SPEED'] * 0.514444
	data_converted['v_ms'] = 0.0
	data_converted['r_rads'] = speed_trials['GPS_HDG_HEADING_ROT_S'] * 0.0174533
	data_converted['nP_revs'] = speed_trials['OPC_40_PROP_RPM_FB'] / 60.0

	print("\nConversions applied:")
	print(f"✓ OPC_07_WATER_SPEED (kts) -> u_ms (m/s)")
	print(f"✓ v_ms = 0.0 (m/s) - sway velocity not available")
	print(f"✓ GPS_HDG_HEADING_ROT_S (deg/s) -> r_rads (rad/s)")
	print(f"✓ OPC_40_PROP_RPM_FB (RPM) -> nP_revs (rev/s)")

	print("\nConverted value ranges:")
	print(f"u (m/s): min={data_converted['u_ms'].min():.2f}, max={data_converted['u_ms'].max():.2f}, mean={data_converted['u_ms'].mean():.2f}")
	print(f"v (m/s): {data_converted['v_ms'].iloc[0]:.2f} (constant)")
	print(f"r (rad/s): min={data_converted['r_rads'].min():.6f}, max={data_converted['r_rads'].max():.6f}, mean={data_converted['r_rads'].mean():.6f}")
	print(f"nP (rev/s): min={data_converted['nP_revs'].min():.2f}, max={data_converted['nP_revs'].max():.2f}, mean={data_converted['nP_revs'].mean():.2f}")

	print(f"\nTarget: {target_col}")
	print(f"Power (kW): min={data_converted[target_col].min():.2f}, max={data_converted[target_col].max():.2f}, mean={data_converted[target_col].mean():.2f}")

	return data_converted

def train_models(data_converted, target_col, ship_params, timesteps=30):
	"""
	Train LSTM PINN models with sequences
	"""
	print("\n" + "="*70)
	print("TRAINING LSTM PINN MODELS")
	print("="*70)

	X = data_converted.drop(columns=[target_col])
	y = data_converted[target_col]

	valid_mask = ~(X.isna().any(axis=1) | y.isna())
	X = X[valid_mask]
	y = y[valid_mask]

	print(f"\nData shape after cleaning: X={X.shape}, y={y.shape}")

	X_columns = list(X.columns)
	column_mapping_updated = {
		'u': X_columns.index('u_ms'),
		'v': X_columns.index('v_ms'),
		'r': X_columns.index('r_rads'),
		'nP': X_columns.index('nP_revs')
	}

	scaler_X = StandardScaler()
	scaler_y = StandardScaler()

	X_scaled = scaler_X.fit_transform(X)
	y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

	print(f"Creating sequences with {timesteps} timesteps...")
	X_sequences = []
	y_sequences = []

	for i in range(timesteps, len(X_scaled)):
		X_sequences.append(X_scaled[i-timesteps:i])
		y_sequences.append(y_scaled[i])

	X_sequences = np.array(X_sequences)
	y_sequences = np.array(y_sequences)

	print(f"Sequence shape: X={X_sequences.shape}, y={y_sequences.shape}")

	print("Splitting data into train/validate/test (60/20/20)...")
	train_size = int(len(X_sequences) * 0.6)
	validate_size = int(len(X_sequences) * 0.2)

	X_train = X_sequences[:train_size]
	y_train = y_sequences[:train_size]

	X_validate = X_sequences[train_size:train_size+validate_size]
	y_validate = y_sequences[train_size:train_size+validate_size]

	X_test = X_sequences[train_size+validate_size:]
	y_test = y_sequences[train_size+validate_size:]

	print(f"Train size: {len(X_train)}, Validate size: {len(X_validate)}, Test size: {len(X_test)}")

	X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
	y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
	X_validate = tf.convert_to_tensor(X_validate, dtype=tf.float32)
	y_validate = tf.convert_to_tensor(y_validate, dtype=tf.float32)
	X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
	y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

	results = []

	print("\n" + "="*50)
	print("BASELINE MODEL (physics_weight=0)")
	print("="*50)

	model_baseline = LSTM_PINN()
	trainer_baseline = PINNTrainer(model_baseline, ship_params, column_mapping_updated, physics_weight=0.0)
	history_baseline = trainer_baseline.fit(X_train, y_train, X_validate, y_validate, epochs=10, batch_size=64, patience=5)

	y_pred = model_baseline.predict(X_test, verbose=0)
	y_pred_original = scaler_y.inverse_transform(y_pred.flatten().reshape(-1, 1))
	y_test_original = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1))

	r2 = r2_score(y_test_original, y_pred_original)
	mse = mean_squared_error(y_test_original, y_pred_original)
	mae = mean_absolute_error(y_test_original, y_pred_original)

	print(f"\nBaseline Test Results: R²={r2:.6f}, MSE={mse:.2f}, MAE={mae:.2f}")
	results.append({'physics_weight': 0.0, 'r2': r2, 'mse': mse, 'mae': mae})

	print("\n" + "="*50)
	print("PHYSICS-INFORMED MODELS")
	print("="*50)

	for pw in [0.00001, 0.0001, 0.001]:
		print(f"\nTraining with physics_weight={pw}")

		model_pinn = LSTM_PINN()
		trainer_pinn = PINNTrainer(model_pinn, ship_params, column_mapping_updated, physics_weight=pw)
		history_pinn = trainer_pinn.fit(X_train, y_train, X_validate, y_validate, epochs=10, batch_size=64, patience=5)

		y_pred = model_pinn.predict(X_test, verbose=0)
		y_pred_original = scaler_y.inverse_transform(y_pred.flatten().reshape(-1, 1))

		r2 = r2_score(y_test_original, y_pred_original)
		mse = mean_squared_error(y_test_original, y_pred_original)
		mae = mean_absolute_error(y_test_original, y_pred_original)

		print(f"LSTM PINN Test Results: R²={r2:.6f}, MSE={mse:.2f}, MAE={mae:.2f}")
		results.append({'physics_weight': pw, 'r2': r2, 'mse': mse, 'mae': mae})

	print("\n" + "="*70)
	print("TEST RESULTS SUMMARY")
	print("="*70)
	for res in results:
		print(f"Physics Weight: {res['physics_weight']:.5f} - R²: {res['r2']:.6f}, MSE: {res['mse']:.2f}, MAE: {res['mae']:.2f}")

	return results

def run_complete_pinn_pipeline(speed_trials, target_col='OPC_12_CPP_ENGINE_POWER'):
	"""
	Complete LSTM PINN pipeline with automatic unit conversions
	"""

	ship_params = {
		'mx': 1240393.736,
		'my': 31009843.3935,
		'Iz': 8.87579e10,
		'Jz': 8.88e9,
		'xG': 0.0,
		'L': 214.0,
		'DP': 6.5,
		'k0': 0.5453,
		'k1': -0.4399,
		'k2': -0.0379,
		'tP': 0.1,
		'wP0': 0.16,
		'xP_prime': -0.5,
		'X0': 0.022,
		'Xvv': -0.040,
		'Xvr': 0.002,
		'Xrr': 0.011,
		'Xvvvv': 0.771,
		'Yv': -0.315,
		'Yr': 0.083,
		'Yvvv': -1.607,
		'Yvvr': 0.379,
		'Yvrr': -0.391,
		'Yrrr': 0.008,
		'Nv': -0.137,
		'Nr': -0.049,
		'Nvvv': -0.030,
		'Nvvr': -0.294,
		'Nvrr': 0.055,
		'Nrrr': -0.013,
	}

	data_converted = prepare_data_with_unit_conversions(speed_trials, target_col)

	results = train_models(data_converted, target_col, ship_params, timesteps=30)

	return results, data_converted


if __name__ == "__main__":
	print("LSTM PINN Training Pipeline with Physics Loss")
	print("\nUsage:")
results, data_converted = run_complete_pinn_pipeline(speed_trials)