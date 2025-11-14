"""Visualisations for feature importance and relationships with engine power."""

import matplotlib.pyplot as plt
import pandas as pd
from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN
speed_trials = SPEED_TRIALS_REGULAR
speed_trials_weather = SPEED_TRIALS_WEATHER_CLEAN
from baseline_models.random_forrest import RANDOM_FORREST_MODEL
from baseline_models.random_forrest import RANDOM_FORREST_MODEL_WEATHER

rf_model = RANDOM_FORREST_MODEL
target_col = "OPC_12_CPP_ENGINE_POWER"
X = speed_trials.drop(columns=["OPC_08_GROUND_SPEED", "PROP_SHAFT_POWER_KMT", "OPC_13_PROP_POWER", "OPC_41_PITCH_FB", target_col])
y = speed_trials[target_col]

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


rf_model_weather = RANDOM_FORREST_MODEL_WEATHER
target_col = "OPC_12_CPP_ENGINE_POWER"
X = speed_trials.drop(columns=["OPC_08_GROUND_SPEED", "PROP_SHAFT_POWER_KMT", "OPC_13_PROP_POWER", "OPC_41_PITCH_FB", target_col])
y = speed_trials_weather[target_col]

feature_importance = pd.DataFrame({
    'feature': X.columns,
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
