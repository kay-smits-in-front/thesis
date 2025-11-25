import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from carbontracker.tracker import CarbonTracker
from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR, create_lag_features_with_boundaries
from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN

speed_trials = SPEED_TRIALS_REGULAR
speed_trials_weather = SPEED_TRIALS_WEATHER_CLEAN

tracker = CarbonTracker(epochs=1)

for epoch in range(1):
	tracker.epoch_start()
	target_col = "OPC_12_CPP_ENGINE_POWER"

	X, y = create_lag_features_with_boundaries(speed_trials, target_col, n_lags=60, forecast_horizon=10)

	split_idx = int(len(X) * 0.8)
	X_train = X[:split_idx]
	X_test = X[split_idx:]
	y_train = y[:split_idx]
	y_test = y[split_idx:]

	model = LinearRegression(n_jobs=-1)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	r2 = r2_score(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	tracker.epoch_end()

print(f"Linear Regression with Lags (boundaries respected) - Speed Trials:")
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")


tracker = CarbonTracker(epochs=1)

for epoch in range(1):
	tracker.epoch_start()
	target_col = "OPC_12_CPP_ENGINE_POWER"

	X, y = create_lag_features_with_boundaries(speed_trials_weather, target_col, n_lags=60, forecast_horizon=10)

	split_idx = int(len(X) * 0.8)
	X_train = X[:split_idx]
	X_test = X[split_idx:]
	y_train = y[:split_idx]
	y_test = y[split_idx:]

	model = LinearRegression(n_jobs=-1)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	r2 = r2_score(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	tracker.epoch_end()

print(f"Linear Regression with Lags (boundaries respected) - Weather:")
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")


tracker = CarbonTracker(epochs=1)

for epoch in range(1):
	tracker.epoch_start()
	target_col = "OPC_12_CPP_ENGINE_POWER"

	X, y = create_lag_features_with_boundaries(speed_trials, target_col, n_lags=60, forecast_horizon=10)

	split_idx = int(len(X) * 0.8)
	X_train = X[:split_idx]
	X_test = X[split_idx:]
	y_train = y[:split_idx]
	y_test = y[split_idx:]

	model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	r2 = r2_score(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	tracker.epoch_end()

print(f"XGBoost with Lags (boundaries respected) - Speed Trials:")
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")


tracker = CarbonTracker(epochs=1)

for epoch in range(1):
	tracker.epoch_start()
	target_col = "OPC_12_CPP_ENGINE_POWER"

	X, y = create_lag_features_with_boundaries(speed_trials_weather, target_col, n_lags=60, forecast_horizon=10)

	split_idx = int(len(X) * 0.8)
	X_train = X[:split_idx]
	X_test = X[split_idx:]
	y_train = y[:split_idx]
	y_test = y[split_idx:]

	model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	r2 = r2_score(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	tracker.epoch_end()

print(f"XGBoost with Lags (boundaries respected) - Weather:")
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")

