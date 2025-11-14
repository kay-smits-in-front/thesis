from carbontracker.tracker import CarbonTracker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN
speed_trials = SPEED_TRIALS_REGULAR
speed_trials_weather = SPEED_TRIALS_WEATHER_CLEAN

tracker = CarbonTracker(epochs=1)

for epoch in range(1):
	tracker.epoch_start()
	target_col = "OPC_12_CPP_ENGINE_POWER"
	col_drop1 = "OPC_41_PITCH_FB"
	col_drop2 = "OPC_13_PROP_POWER"
	col_drop3 = "PROP_SHAFT_POWER_KMT"
	col_drop4 = "OPC_08_GROUND_SPEED"

	X = speed_trials.drop(columns=[col_drop4, col_drop3, col_drop2, col_drop1, target_col])
	y = speed_trials[target_col]


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	rf_model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1, n_jobs= -3)
	rf_model.fit(X_train, y_train)

	y_pred = rf_model.predict(X_test)

	r2 = r2_score(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	tracker.epoch_end()

RANDOM_FORREST_MODEL = rf_model

print(f"Ramdom forrest results on Speed Trials data:")
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")



for epoch in range(1):
	tracker.epoch_start()
	target_col = "OPC_12_CPP_ENGINE_POWER"
	col_drop1 = "OPC_41_PITCH_FB"
	col_drop2 = "OPC_13_PROP_POWER"
	col_drop3 = "PROP_SHAFT_POWER_KMT"
	col_drop4 = "OPC_08_GROUND_SPEED"

	X = speed_trials_weather.drop(columns=[col_drop4, col_drop3, col_drop2, col_drop1, target_col])
	y = speed_trials_weather[target_col]


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	rf_model_weather = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1, n_jobs= -3)
	rf_model_weather.fit(X_train, y_train)

	y_pred = rf_model_weather.predict(X_test)

	r2 = r2_score(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)
	tracker.epoch_end()

RANDOM_FORREST_MODEL_WEATHER= rf_model_weather

print(f"Ramdom forrest results on Speed Trials weather data:")
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")