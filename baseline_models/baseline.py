"""MLP model for speed trials data using TensorFlow."""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from carbontracker.tracker import CarbonTracker
from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN

speed_trials = SPEED_TRIALS_REGULAR
speed_trials_weather = SPEED_TRIALS_WEATHER_CLEAN

tracker = CarbonTracker(epochs=1)

# Hyperparameters
HIDDEN_LAYERS = [48]
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10

# Training loop.
for epoch in range(1):
    tracker.epoch_start()
    target_col = "OPC_12_CPP_ENGINE_POWER"
    col_drop1 = "OPC_41_PITCH_FB"
    col_drop2 = "OPC_13_PROP_POWER"
    col_drop3 = "PROP_SHAFT_POWER_KMT"
    col_drop5 = "OPC_08_GROUND_SPEED"

    X = speed_trials.drop(columns=[col_drop5, col_drop3, col_drop2, col_drop1, target_col])
    y = speed_trials[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(HIDDEN_LAYERS[0], activation='Adam', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    tracker.epoch_end()

print(f"MLP Model Results on Speed Trials data:")
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")

# Training loop.
for epoch in range(1):
    tracker.epoch_start()
    target_col = "OPC_12_CPP_ENGINE_POWER"
    col_drop1 = "OPC_41_PITCH_FB"
    col_drop2 = "OPC_13_PROP_POWER"
    col_drop3 = "PROP_SHAFT_POWER_KMT"
    col_drop5 = "OPC_08_GROUND_SPEED"

    X = speed_trials_weather.drop(columns=[col_drop5, col_drop3, col_drop2, col_drop1, target_col])
    y = speed_trials_weather[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(HIDDEN_LAYERS[0], activation='Adam', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    tracker.epoch_end()

print(f"MLP Model Results on Speed Trials + Weather data:")
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")