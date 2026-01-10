"""
LSTM Optimization with Early Pruning
Tests architectures, batch sizes, and physics weights efficiently
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CONFIG = {
	'output_dir': 'model_performance',
	'architectures': [[64], [64, 32], [128, 64, 32]],
	'batch_sizes': [16, 32],
	'physics_weights': [0.0, 0.001, 0.01],
	'timesteps': 30,
	'epochs': 15,
	'patience': 7,
	'learning_rate': 0.001,
	'dropout_rate': 0.2,
	'prune_threshold': 0.3
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

class LSTMWithPhysics(keras.Model):
	def __init__(self, architecture, dropout_rate, physics_weight, scaler_X, scaler_y, column_indices):
		super().__init__()
		self.rnn_layers, self.dropout_layers = [], []
		for i, units in enumerate(architecture):
			self.rnn_layers.append(LSTM(units, return_sequences=(i < len(architecture)-1)))
			self.dropout_layers.append(Dropout(dropout_rate))
		self.output_layer = Dense(1)
		self.physics_weight = physics_weight
		self.optimizer = Adam(learning_rate=0.001)
		self.scaler_X_mean = tf.constant(scaler_X.mean_, dtype=tf.float32)
		self.scaler_X_std = tf.constant(scaler_X.scale_, dtype=tf.float32)
		self.scaler_y_mean = tf.constant(scaler_y.mean_[0], dtype=tf.float32)
		self.scaler_y_std = tf.constant(scaler_y.scale_[0], dtype=tf.float32)
		self.u_idx, self.v_idx, self.r_idx, self.nP_idx = column_indices

	def call(self, inputs):
		x = inputs
		for rnn, dropout in zip(self.rnn_layers, self.dropout_layers):
			x = rnn(x)
			x = dropout(x)
		return self.output_layer(x)

	def compute_physics_loss(self, inputs, predictions):
		u_scaled = inputs[:, -1, self.u_idx]
		v_scaled = inputs[:, -1, self.v_idx]
		r_scaled = inputs[:, -1, self.r_idx]
		nP_scaled = inputs[:, -1, self.nP_idx]
		u = u_scaled * self.scaler_X_std[self.u_idx] + self.scaler_X_mean[self.u_idx]
		v = v_scaled * self.scaler_X_std[self.v_idx] + self.scaler_X_mean[self.v_idx]
		r = r_scaled * self.scaler_X_std[self.r_idx] + self.scaler_X_mean[self.r_idx]
		nP = nP_scaled * self.scaler_X_std[self.nP_idx] + self.scaler_X_mean[self.nP_idx]
		predicted_power_kW = predictions[:, 0] * self.scaler_y_std + self.scaler_y_mean
		predicted_power_watts = predicted_power_kW * 1000.0
		XP = compute_propeller_force(u, v, r, nP, SHIP_PARAMS)
		predicted_thrust = predicted_power_watts / (tf.abs(u) + 1e-6)
		return tf.reduce_mean(tf.square((XP - predicted_thrust) / 1e6))

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

def create_sequences(X, y, timesteps):
	X_seq, y_seq = [], []
	for i in range(timesteps, len(X)):
		X_seq.append(X[i-timesteps:i])
		y_seq.append(y[i])
	return np.array(X_seq), np.array(y_seq)

def prepare_data(data, target_col, timesteps):
	all_exclude = EXCLUDE_COLS + [target_col]
	numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = [col for col in numeric_cols if col not in all_exclude]
	X, y = data[feature_cols], data[target_col]
	valid_mask = ~(X.isna().any(axis=1) | y.isna())
	X, y = X[valid_mask], y[valid_mask]
	train_size, val_size = int(len(X) * 0.6), int(len(X) * 0.2)
	X_train_raw, y_train_raw = X.iloc[:train_size], y.iloc[:train_size]
	X_val_raw, y_val_raw = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
	X_test_raw, y_test_raw = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]
	scaler_X, scaler_y = StandardScaler(), StandardScaler()
	scaler_X.fit(X_train_raw)
	scaler_y.fit(y_train_raw.values.reshape(-1, 1))
	X_train_seq, y_train_seq = create_sequences(scaler_X.transform(X_train_raw), scaler_y.transform(y_train_raw.values.reshape(-1, 1)).flatten(), timesteps)
	X_val_seq, y_val_seq = create_sequences(scaler_X.transform(X_val_raw), scaler_y.transform(y_val_raw.values.reshape(-1, 1)).flatten(), timesteps)
	X_test_seq, y_test_seq = create_sequences(scaler_X.transform(X_test_raw), scaler_y.transform(y_test_raw.values.reshape(-1, 1)).flatten(), timesteps)
	splits = {
		'X_train': tf.convert_to_tensor(X_train_seq, dtype=tf.float32),
		'y_train': tf.convert_to_tensor(y_train_seq, dtype=tf.float32),
		'X_val': tf.convert_to_tensor(X_val_seq, dtype=tf.float32),
		'y_val': tf.convert_to_tensor(y_val_seq, dtype=tf.float32),
		'X_test': tf.convert_to_tensor(X_test_seq, dtype=tf.float32),
		'y_test': tf.convert_to_tensor(y_test_seq, dtype=tf.float32)
	}
	u_idx = feature_cols.index('OPC_07_WATER_SPEED')
	v_idx = feature_cols.index('v_ms')
	r_idx = feature_cols.index('GPS_HDG_HEADING_ROT_S')
	nP_idx = feature_cols.index('OPC_40_PROP_RPM_FB')
	return splits, scaler_X, scaler_y, (u_idx, v_idx, r_idx, nP_idx)

def train_single_model(splits, scaler_X, scaler_y, column_indices, architecture, batch_size, physics_weight, config):
	model = LSTMWithPhysics(architecture, config['dropout_rate'], physics_weight, scaler_X, scaler_y, column_indices)
	model.compile(optimizer=model.optimizer, loss='mse')
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True)
	model.fit(splits['X_train'], splits['y_train'], validation_data=(splits['X_val'], splits['y_val']),
	         epochs=config['epochs'], batch_size=batch_size, callbacks=[early_stop], verbose=0)
	def evaluate(X, y):
		y_pred = model.predict(X, verbose=0).flatten()
		y_original = scaler_y.inverse_transform(y.numpy().reshape(-1, 1)).flatten()
		y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
		return {'r2': float(r2_score(y_original, y_pred_original)), 'mse': float(mean_squared_error(y_original, y_pred_original)), 'mae': float(mean_absolute_error(y_original, y_pred_original))}
	return {'architecture': architecture, 'batch_size': batch_size, 'physics_weight': physics_weight,
	        'train': evaluate(splits['X_train'], splits['y_train']), 'val': evaluate(splits['X_val'], splits['y_val']), 'test': evaluate(splits['X_test'], splits['y_test'])}

def run_optimization(data, target_col, config):
	print(f"\n{'='*80}\nLSTM OPTIMIZATION WITH EARLY PRUNING\nDataset: REGULAR | Prune threshold: R² < {config['prune_threshold']}\n{'='*80}")
	splits, scaler_X, scaler_y, column_indices = prepare_data(data, target_col, config['timesteps'])
	print(f"\nTrain: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
	print(f"\nTesting {len(config['architectures'])} architectures × {len(config['batch_sizes'])} batch sizes × {len(config['physics_weights'])} physics weights")
	results, pruned_architectures, pruned_physics_weights, tested, skipped = [], set(), set(), 0, 0
	print(f"\n{'='*80}\nPHASE 1: Physics Weight Screening\n{'='*80}")
	test_arch, test_batch = config['architectures'][0], config['batch_sizes'][0]
	for phys_weight in config['physics_weights']:
		tested += 1
		print(f"\n[Screening {tested}/{len(config['physics_weights'])}] Physics weight: {phys_weight}, arch={test_arch}, batch={test_batch}")
		result = train_single_model(splits, scaler_X, scaler_y, column_indices, test_arch, test_batch, phys_weight, config)
		results.append(result)
		val_r2 = result['val']['r2']
		print(f"  Val R²: {val_r2:.4f}, Test R²: {result['test']['r2']:.4f}")
		if val_r2 < config['prune_threshold']:
			pruned_physics_weights.add(phys_weight)
			print(f"  ⚠️  PRUNED: Physics weight {phys_weight} (Val R² < {config['prune_threshold']})")
	print(f"\n{'='*80}\nPHASE 2: Architecture & Batch Size Optimization\n{'='*80}")
	active_physics = [pw for pw in config['physics_weights'] if pw not in pruned_physics_weights]
	print(f"\nActive physics weights: {active_physics}")
	for phys_weight in active_physics:
		for arch in config['architectures']:
			first_batch = config['batch_sizes'][0]
			tested += 1
			print(f"\n[{tested}] arch={arch}, batch={first_batch}, physics={phys_weight}")
			result = train_single_model(splits, scaler_X, scaler_y, column_indices, arch, first_batch, phys_weight, config)
			results.append(result)
			val_r2 = result['val']['r2']
			print(f"  Val R²: {val_r2:.4f}, Test R²: {result['test']['r2']:.4f}")
			if val_r2 < config['prune_threshold']:
				pruned_architectures.add((arch, phys_weight))
				skipped_batches = len(config['batch_sizes']) - 1
				skipped += skipped_batches
				print(f"  ⚠️  PRUNED: Skipping {skipped_batches} remaining batch sizes (Val R² < {config['prune_threshold']})")
				continue
			for batch_size in config['batch_sizes'][1:]:
				tested += 1
				print(f"\n[{tested}] arch={arch}, batch={batch_size}, physics={phys_weight}")
				result = train_single_model(splits, scaler_X, scaler_y, column_indices, arch, batch_size, phys_weight, config)
				results.append(result)
				print(f"  Val R²: {result['val']['r2']:.4f}, Test R²: {result['test']['r2']:.4f}")
	print(f"\n{'='*80}\nOPTIMIZATION COMPLETE\n{'='*80}\nTested: {tested} configurations\nSkipped: {skipped} configurations (early pruning)\nTotal results: {len(results)}")
	if len(pruned_physics_weights) > 0:
		print(f"\nPruned physics weights: {sorted(pruned_physics_weights)}")
	if len(pruned_architectures) > 0:
		print(f"Pruned architectures: {len(pruned_architectures)} configurations")
	best_result = max(results, key=lambda x: x['test']['r2'])
	print(f"\n{'='*80}\nBEST CONFIGURATION:\n{'='*80}\n  Architecture: {best_result['architecture']}\n  Batch size: {best_result['batch_size']}\n  Physics weight: {best_result['physics_weight']}\n  Test R²: {best_result['test']['r2']:.4f}\n  Test MSE: {best_result['test']['mse']:.2f}\n  Test MAE: {best_result['test']['mae']:.2f}\n{'='*80}")
	return {'results': results, 'pruned_physics_weights': list(pruned_physics_weights), 'pruned_architectures': len(pruned_architectures), 'tested': tested, 'skipped': skipped}

if __name__ == "__main__":
	OUTPUT_DIR = "output"
	regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
	if os.path.exists(regular_path):
		speed_trials_regular = pd.read_csv(regular_path)
		target_col = 'OPC_12_CPP_ENGINE_POWER'
		optimization_results = run_optimization(speed_trials_regular, target_col, CONFIG)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_file = f'{CONFIG["output_dir"]}/rnn_optimization_{timestamp}.json'
		with open(output_file, 'w') as f:
			json.dump(optimization_results, f, indent=2)
		print(f"\nResults saved to: {output_file}")
	else:
		print("ERROR: Run pre_process.py first!")
