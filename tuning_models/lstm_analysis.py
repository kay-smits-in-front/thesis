"""
Uitgebreide LSTM Analyse en Visualisatie
Bevat: loss curves, architectuur vergelijking, sequence length impact, en performance diagnostics
Note: Dropout=0.2, LR=0.001, en batch sizes 16/32/64 zijn al geoptimaliseerd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN

# Styling voor plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class LSTMAnalyzer:
    """Klasse voor uitgebreide LSTM analyse en visualisatie"""

    def __init__(self, data, data_name="Speed Trials", target_col="OPC_12_CPP_ENGINE_POWER"):
        self.data = data
        self.data_name = data_name
        self.target_col = target_col
        self.results = []
        self.histories = {}

        # Optimale hyperparameters (al getest)
        self.optimal_dropout = 0.2
        self.optimal_lr = 0.001
        self.optimal_batch_size = 32  # Middenweg van 16, 32, 64

        print(f"LSTM Analyzer geïnitialiseerd met optimale params:")
        print(f"  Dropout: {self.optimal_dropout}")
        print(f"  Learning Rate: {self.optimal_lr}")
        print(f"  Batch Size: {self.optimal_batch_size}")

    def prepare_sequences(self, timesteps=30):
        """Bereid sequentiële data voor"""
        # Drop kolommen
        col_drop = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED"]

        X = self.data.drop(columns=col_drop + [self.target_col])
        y = self.data[self.target_col]

        # Remove NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        # Normalisatie
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        # Maak sequences
        X_sequences = []
        y_sequences = []

        for i in range(timesteps, len(X_scaled)):
            X_sequences.append(X_scaled[i-timesteps:i])
            y_sequences.append(y_scaled[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        # Train/val/test split (60/20/20)
        train_size = int(len(X_sequences) * 0.6)
        val_size = int(len(X_sequences) * 0.2)

        X_train = X_sequences[:train_size]
        y_train = y_sequences[:train_size]

        X_val = X_sequences[train_size:train_size+val_size]
        y_val = y_sequences[train_size:train_size+val_size]

        X_test = X_sequences[train_size+val_size:]
        y_test = y_sequences[train_size+val_size:]

        # Convert to tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

        print(f"\nSequences voorbereid (timesteps={timesteps}):")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return (X_train, y_train, X_val, y_val, X_test, y_test), scaler_y

    def create_lstm(self, layers=[64, 32, 16], dropout=None, learning_rate=None):
        """Maak LSTM model met gespecificeerde architectuur"""
        if dropout is None:
            dropout = self.optimal_dropout
        if learning_rate is None:
            learning_rate = self.optimal_lr

        model = Sequential()

        # Eerste LSTM layer
        model.add(LSTM(layers[0], return_sequences=len(layers) > 1, input_shape=(None, None)))
        model.add(Dropout(dropout))

        # Hidden LSTM layers
        for i, units in enumerate(layers[1:-1]):
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout))

        # Laatste LSTM layer (als er meerdere zijn)
        if len(layers) > 1:
            model.add(LSTM(layers[-1], return_sequences=False))
            model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(1))

        # Compileer
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                     loss='mse',
                     metrics=['mae'])

        return model

    def train_and_evaluate(self, model_name, model, data_tuple, scaler_y, epochs=100, verbose=0):
        """Train model en evalueer performance"""
        X_train, y_train, X_val, y_val, X_test, y_test = data_tuple

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.optimal_batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        # Sla history op
        self.histories[model_name] = history.history

        # Evalueer op test set
        y_pred = model.predict(X_test, verbose=0)

        # Inverse transform
        y_pred_orig = scaler_y.inverse_transform(y_pred)
        y_test_orig = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1))

        # Metrics
        r2 = r2_score(y_test_orig, y_pred_orig)
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)

        result = {
            'model_name': model_name,
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        }

        self.results.append(result)

        print(f"{model_name}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, Epochs={len(history.history['loss'])}")

        return result

    def visualize_learning_curves(self, save_path=None):
        """Visualiseer learning curves voor alle getrainde modellen"""
        n_models = len(self.histories)

        if n_models == 0:
            print("Geen modellen getraind om te visualiseren!")
            return

        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5*n_models))

        if n_models == 1:
            axes = axes.reshape(1, -1)

        for idx, (model_name, history) in enumerate(self.histories.items()):
            # Loss curves
            axes[idx, 0].plot(history['loss'], label='Train Loss', linewidth=2, color='blue')
            axes[idx, 0].plot(history['val_loss'], label='Val Loss', linewidth=2, color='orange')
            axes[idx, 0].set_xlabel('Epoch')
            axes[idx, 0].set_ylabel('Loss (MSE)')
            axes[idx, 0].set_title(f'{model_name} - Loss Curves')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].set_yscale('log')  # Log scale voor betere visualisatie

            # MAE curves
            axes[idx, 1].plot(history['mae'], label='Train MAE', linewidth=2, color='green')
            axes[idx, 1].plot(history['val_mae'], label='Val MAE', linewidth=2, color='red')
            axes[idx, 1].set_xlabel('Epoch')
            axes[idx, 1].set_ylabel('MAE')
            axes[idx, 1].set_title(f'{model_name} - MAE Curves')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)

        plt.suptitle(f'LSTM Learning Curves - {self.data_name}', fontsize=16, y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves opgeslagen: {save_path}")
        else:
            plt.show()

    def visualize_predictions(self, model, model_name, data_tuple, scaler_y, save_path=None):
        """Visualiseer voorspellingen vs werkelijke waarden"""
        X_train, y_train, X_val, y_val, X_test, y_test = data_tuple

        y_pred = model.predict(X_test, verbose=0)

        # Inverse transform
        y_pred_orig = scaler_y.inverse_transform(y_pred)
        y_test_orig = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Scatter plot
        axes[0, 0].scatter(y_test_orig, y_pred_orig, alpha=0.5, s=20)
        axes[0, 0].plot([y_test_orig.min(), y_test_orig.max()],
                       [y_test_orig.min(), y_test_orig.max()],
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Werkelijke Waarden (kW)')
        axes[0, 0].set_ylabel('Voorspelde Waarden (kW)')
        axes[0, 0].set_title(f'{model_name} - Voorspellingen vs Werkelijkheid')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals
        residuals = y_test_orig - y_pred_orig
        axes[0, 1].scatter(y_pred_orig, residuals, alpha=0.5, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Voorspelde Waarden (kW)')
        axes[0, 1].set_ylabel('Residuals (kW)')
        axes[0, 1].set_title(f'{model_name} - Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Time series van voorspellingen (eerste 500 samples)
        n_samples = min(500, len(y_test_orig))
        axes[1, 0].plot(y_test_orig[:n_samples], label='Werkelijk', linewidth=1.5, alpha=0.7)
        axes[1, 0].plot(y_pred_orig[:n_samples], label='Voorspeld', linewidth=1.5, alpha=0.7)
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Engine Power (kW)')
        axes[1, 0].set_title(f'{model_name} - Time Series Comparison (eerste {n_samples} samples)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Histogram van residuals
        axes[1, 1].hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residuals (kW)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'{model_name} - Residual Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Bereken statistieken
        r2 = r2_score(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae = mean_absolute_error(y_test_orig, y_pred_orig)

        plt.suptitle(f'LSTM Prediction Analysis - {self.data_name}\nR²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}',
                    fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction visualisatie opgeslagen: {save_path}")
        else:
            plt.show()

    def compare_architectures(self, timesteps=30):
        """Vergelijk verschillende LSTM architecturen"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN LSTM ARCHITECTUREN")
        print("="*70)

        # Bereid data voor
        data_tuple, scaler_y = self.prepare_sequences(timesteps=timesteps)

        architectures = [
            {"name": "1-Layer Narrow", "layers": [32]},
            {"name": "1-Layer Medium", "layers": [64]},
            {"name": "1-Layer Wide", "layers": [128]},
            {"name": "2-Layer Standard", "layers": [64, 32]},
            {"name": "3-Layer Standard", "layers": [64, 32, 16]},
            {"name": "3-Layer Wide", "layers": [128, 64, 32]},
            {"name": "4-Layer Deep", "layers": [128, 64, 32, 16]},
        ]

        for arch in architectures:
            print(f"\nTrainen: {arch['name']} - {arch['layers']}")
            model = self.create_lstm(layers=arch['layers'])
            self.train_and_evaluate(arch['name'], model, data_tuple, scaler_y, epochs=100)

        return data_tuple, scaler_y

    def compare_sequence_lengths(self):
        """Vergelijk verschillende sequence lengths"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN SEQUENCE LENGTHS (timesteps)")
        print("="*70)

        sequence_lengths = [10, 20, 30, 50, 100]

        for seq_len in sequence_lengths:
            print(f"\nTrainen met sequence_length={seq_len}")
            data_tuple, scaler_y = self.prepare_sequences(timesteps=seq_len)

            model = self.create_lstm(layers=[64, 32, 16])
            self.train_and_evaluate(f"SeqLen {seq_len}", model, data_tuple, scaler_y, epochs=100)

    def analyze_convergence_speed(self, save_path=None):
        """Analyseer convergentie snelheid van verschillende architecturen"""
        print("\n" + "="*70)
        print("CONVERGENTIE ANALYSE")
        print("="*70)

        if not self.histories:
            print("Geen training histories beschikbaar!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot alle loss curves samen
        for model_name, history in self.histories.items():
            axes[0].plot(history['val_loss'], label=model_name, linewidth=2, alpha=0.7)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Validation Loss (MSE)')
        axes[0].set_title('Convergentie Snelheid Vergelijking')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # Plot aantal epochs tot convergentie
        epochs_to_converge = [len(hist['loss']) for hist in self.histories.values()]
        model_names = list(self.histories.keys())

        axes[1].barh(model_names, epochs_to_converge, color='steelblue')
        axes[1].set_xlabel('Epochs to Convergence')
        axes[1].set_title('Training Duration')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'LSTM Convergence Analysis - {self.data_name}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence analysis opgeslagen: {save_path}")
        else:
            plt.show()

    def visualize_results_comparison(self, save_path=None):
        """Visualiseer vergelijking van alle resultaten"""
        if not self.results:
            print("Geen resultaten om te visualiseren!")
            return

        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # R² comparison
        axes[0, 0].barh(range(len(df)), df['r2'], color='steelblue')
        axes[0, 0].set_yticks(range(len(df)))
        axes[0, 0].set_yticklabels(df['model_name'], fontsize=8)
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Model R² Vergelijking')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # RMSE comparison
        axes[0, 1].barh(range(len(df)), df['rmse'], color='coral')
        axes[0, 1].set_yticks(range(len(df)))
        axes[0, 1].set_yticklabels(df['model_name'], fontsize=8)
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_title('Model RMSE Vergelijking')
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # MAE comparison
        axes[1, 0].barh(range(len(df)), df['mae'], color='seagreen')
        axes[1, 0].set_yticks(range(len(df)))
        axes[1, 0].set_yticklabels(df['model_name'], fontsize=8)
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].set_title('Model MAE Vergelijking')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # Training efficiency (R² / epochs)
        efficiency = df['r2'] / df['epochs_trained'] * 100
        axes[1, 1].barh(range(len(df)), efficiency, color='mediumpurple')
        axes[1, 1].set_yticks(range(len(df)))
        axes[1, 1].set_yticklabels(df['model_name'], fontsize=8)
        axes[1, 1].set_xlabel('Training Efficiency (R² / Epoch * 100)')
        axes[1, 1].set_title('Model Efficiency Vergelijking')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'LSTM Model Vergelijking - {self.data_name}', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results comparison opgeslagen: {save_path}")
        else:
            plt.show()

    def save_results(self, filepath):
        """Sla resultaten op als JSON"""
        output = {
            'data_name': self.data_name,
            'timestamp': datetime.now().isoformat(),
            'optimal_hyperparameters': {
                'dropout': self.optimal_dropout,
                'learning_rate': self.optimal_lr,
                'batch_size': self.optimal_batch_size
            },
            'results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Resultaten opgeslagen: {filepath}")


def run_complete_lstm_analysis():
    """Voer volledige LSTM analyse uit voor beide datasets"""

    # === SPEED TRIALS DATA ===
    print("\n" + "="*70)
    print("LSTM ANALYSE: SPEED TRIALS DATA")
    print("="*70)

    analyzer_st = LSTMAnalyzer(SPEED_TRIALS_REGULAR, data_name="Speed Trials")

    # Vergelijk architecturen
    data_tuple_st, scaler_y_st = analyzer_st.compare_architectures(timesteps=30)

    # Vergelijk sequence lengths
    analyzer_st.compare_sequence_lengths()

    # Visualiseer learning curves
    analyzer_st.visualize_learning_curves(
        save_path='tuning_models/visualisations/lstm_learning_curves_speed_trials.png'
    )

    # Convergence analyse
    analyzer_st.analyze_convergence_speed(
        save_path='tuning_models/visualisations/lstm_convergence_speed_trials.png'
    )

    # Visualiseer resultaten vergelijking
    analyzer_st.visualize_results_comparison(
        save_path='tuning_models/visualisations/lstm_comparison_speed_trials.png'
    )

    # Train beste model en visualiseer voorspellingen
    best_model_st = analyzer_st.create_lstm(layers=[64, 32, 16])
    analyzer_st.train_and_evaluate("Best Model", best_model_st, data_tuple_st, scaler_y_st, epochs=100, verbose=1)

    analyzer_st.visualize_predictions(
        best_model_st, "Best Model", data_tuple_st, scaler_y_st,
        save_path='tuning_models/visualisations/lstm_predictions_speed_trials.png'
    )

    # Sla resultaten op
    analyzer_st.save_results('tuning_models/visualisations/lstm_results_speed_trials.json')

    # === WEATHER DATA ===
    print("\n" + "="*70)
    print("LSTM ANALYSE: SPEED TRIALS + WEATHER DATA")
    print("="*70)

    analyzer_weather = LSTMAnalyzer(SPEED_TRIALS_WEATHER_CLEAN, data_name="Speed Trials + Weather")

    # Vergelijk architecturen
    data_tuple_weather, scaler_y_weather = analyzer_weather.compare_architectures(timesteps=30)

    # Vergelijk sequence lengths
    analyzer_weather.compare_sequence_lengths()

    # Visualiseer learning curves
    analyzer_weather.visualize_learning_curves(
        save_path='tuning_models/visualisations/lstm_learning_curves_weather.png'
    )

    # Convergence analyse
    analyzer_weather.analyze_convergence_speed(
        save_path='tuning_models/visualisations/lstm_convergence_weather.png'
    )

    # Visualiseer resultaten vergelijking
    analyzer_weather.visualize_results_comparison(
        save_path='tuning_models/visualisations/lstm_comparison_weather.png'
    )

    # Train beste model en visualiseer voorspellingen
    best_model_weather = analyzer_weather.create_lstm(layers=[64, 32, 16])
    analyzer_weather.train_and_evaluate("Best Model", best_model_weather, data_tuple_weather, scaler_y_weather, epochs=100, verbose=1)

    analyzer_weather.visualize_predictions(
        best_model_weather, "Best Model", data_tuple_weather, scaler_y_weather,
        save_path='tuning_models/visualisations/lstm_predictions_weather.png'
    )

    # Sla resultaten op
    analyzer_weather.save_results('tuning_models/visualisations/lstm_results_weather.json')

    print("\n" + "="*70)
    print("LSTM ANALYSE VOLTOOID")
    print("Alle visualisaties opgeslagen in tuning_models/visualisations/")
    print("="*70)


if __name__ == "__main__":
    run_complete_lstm_analysis()
