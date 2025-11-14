"""Visualisations for feature importance and relationships with engine power."""

import matplotlib.pyplot as plt
import pandas as pd
from clean_data.clean_speed_trials import clean_speed_trials
from clean_data.clean_weather_data import clean_weather_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

if __name__ == "__main__":
	# Load and prepare data
	speed_trials = clean_speed_trials(verbose=False)
	speed_trials_weather = clean_weather_data(verbose=False)

	# Train Random Forest model on speed trials data
	target_col = "OPC_12_CPP_ENGINE_POWER"
	col_drop1 = "OPC_41_PITCH_FB"
	col_drop2 = "OPC_13_PROP_POWER"
	col_drop3 = "PROP_SHAFT_POWER_KMT"
	col_drop4 = "OPC_08_GROUND_SPEED"

	X = speed_trials.drop(columns=[col_drop4, col_drop3, col_drop2, col_drop1, target_col])
	y = speed_trials[target_col]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	rf_model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=0, n_jobs=-3)
	rf_model.fit(X_train, y_train)

	feature_importance = pd.DataFrame({
	    'feature': X.columns,
	    'importance': rf_model.feature_importances_
	}).sort_values('importance', ascending=False)

	fig, ax = plt.subplots(figsize=(10, 8))
	top_features = feature_importance.head(15)
	ax.barh(top_features['feature'], top_features['importance'])
	ax.set_xlabel('Importance Score')
	ax.set_title('Top 15 Feature Importances for Predicting Engine Power')
	ax.invert_yaxis()
	plt.tight_layout()
	plt.show()

	top_5_features = feature_importance.head(6)['feature'].values
	fig, axes = plt.subplots(2, 3, figsize=(18, 12))
	axes = axes.flatten()

	for i, feature in enumerate(top_5_features):
	    correlation = speed_trials[feature].corr(speed_trials[target_col])
	    axes[i].scatter(speed_trials[feature], speed_trials[target_col], alpha=0.3, s=1, color='blue', label='Data points')
	    axes[i].set_xlabel(f'{feature} (units)', fontsize=10)
	    axes[i].set_ylabel(f'{target_col} (Power)', fontsize=10)
	    axes[i].set_title(f'{feature} vs Engine Power\nCorrelation: {correlation:.4f}, Importance: {feature_importance.iloc[i]["importance"]:.4f}', fontsize=11)
	    axes[i].grid(True, alpha=0.3)
	    axes[i].legend()

	axes[5].axis('off')
	plt.suptitle('T6 Most Important Features vs Engine Power', fontsize=14, y=1.00)
	plt.tight_layout()
	plt.show()

	print("\nFeature details:")
	for i, feature in enumerate(top_5_features):
	    print(f"{feature}: min={speed_trials[feature].min():.2f}, max={speed_trials[feature].max():.2f}, mean={speed_trials[feature].mean():.2f}")


	print(f"\n\nNow for weather data:\n")

	# Train Random Forest model on weather data
	X_weather = speed_trials_weather.drop(columns=[col_drop4, col_drop3, col_drop2, col_drop1, target_col])
	y_weather = speed_trials_weather[target_col]

	X_train_weather, X_test_weather, y_train_weather, y_test_weather = train_test_split(X_weather, y_weather, test_size=0.2, random_state=42)
	rf_model_weather = RandomForestRegressor(n_estimators=100, random_state=42, verbose=0, n_jobs=-3)
	rf_model_weather.fit(X_train_weather, y_train_weather)

	feature_importance = pd.DataFrame({
	    'feature': X_weather.columns,
	    'importance': rf_model_weather.feature_importances_
	}).sort_values('importance', ascending=False)

	fig, ax = plt.subplots(figsize=(10, 8))
	top_features = feature_importance.head(15)
	ax.barh(top_features['feature'], top_features['importance'])
	ax.set_xlabel('Importance Score')
	ax.set_title('Top 15 Feature Importances for Predicting Engine Power')
	ax.invert_yaxis()
	plt.tight_layout()
	plt.show()

	top_5_features = feature_importance.head(6)['feature'].values
	fig, axes = plt.subplots(2, 3, figsize=(18, 12))
	axes = axes.flatten()

	for i, feature in enumerate(top_5_features):
	    correlation = speed_trials_weather[feature].corr(speed_trials_weather[target_col])
	    axes[i].scatter(speed_trials_weather[feature], speed_trials_weather[target_col], alpha=0.3, s=1, color='blue', label='Data points')
	    axes[i].set_xlabel(f'{feature} (units)', fontsize=10)
	    axes[i].set_ylabel(f'{target_col} (Power)', fontsize=10)
	    axes[i].set_title(f'{feature} vs Engine Power\nCorrelation: {correlation:.4f}, Importance: {feature_importance.iloc[i]["importance"]:.4f}', fontsize=11)
	    axes[i].grid(True, alpha=0.3)
	    axes[i].legend()

	axes[5].axis('off')
	plt.suptitle('T6 Most Important Features vs Engine Power', fontsize=14, y=1.00)
	plt.tight_layout()
	plt.show()

	print("\nFeature details:")
	for i, feature in enumerate(top_5_features):
	    print(f"{feature}: min={speed_trials_weather[feature].min():.2f}, max={speed_trials_weather[feature].max():.2f}, mean={speed_trials_weather[feature].mean():.2f}")
