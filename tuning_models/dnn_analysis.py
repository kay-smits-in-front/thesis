"""
Uitgebreide DNN (Deep Neural Network) Analyse en Tuning
Bevat: deep architecturen, activation functions, regularization, en advanced diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN

# Styling voor plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class DNNAnalyzer:
    """Klasse voor uitgebreide DNN analyse en visualisatie"""

    def __init__(self, data, data_name="Speed Trials", target_col="OPC_12_CPP_ENGINE_POWER"):
        self.data = data
        self.data_name = data_name
        self.target_col = target_col
        self.results = []
        self.histories = {}

        # Bereid data voor
        self._prepare_data()

    def _prepare_data(self):
        """Bereid data voor met train/val/test splits"""
        # Drop kolommen
        col_drop = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED"]

        X = self.data.drop(columns=col_drop + [self.target_col])
        y = self.data[self.target_col]

        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Verder splitsen train in train/val (75/25 van train = 60/20 totaal)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        # Normalisatie
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.X_train = self.scaler_X.fit_transform(X_train)
        self.X_val = self.scaler_X.transform(X_val)
        self.X_test = self.scaler_X.transform(X_test)

        self.y_train = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        self.y_val = self.scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
        self.y_test = self.scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

        self.input_dim = self.X_train.shape[1]

        print(f"Data voorbereid: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")

    def create_dnn(self, layers=[512, 256, 128, 64, 32], dropout=0.3, learning_rate=0.001,
                   activation='relu', use_batch_norm=True, optimizer='adam', l2_reg=0.0):
        """Maak DNN model met gespecificeerde architectuur"""
        model = Sequential()

        # Input layer
        if l2_reg > 0:
            model.add(Dense(layers[0], activation=activation, input_dim=self.input_dim,
                          kernel_regularizer=l2(l2_reg)))
        else:
            model.add(Dense(layers[0], activation=activation, input_dim=self.input_dim))

        if use_batch_norm:
            model.add(BatchNormalization())
        if dropout > 0:
            model.add(Dropout(dropout))

        # Hidden layers
        for units in layers[1:]:
            if l2_reg > 0:
                model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))
            else:
                model.add(Dense(units, activation=activation))

            if use_batch_norm:
                model.add(BatchNormalization())
            if dropout > 0:
                model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(1, activation='linear'))

        # Select optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            opt = Adam(learning_rate=learning_rate)

        # Compileer
        model.compile(optimizer=opt,
                     loss='mse',
                     metrics=['mae'])

        return model

    def train_and_evaluate(self, model_name, model, epochs=100, batch_size=32, verbose=0):
        """Train model en evalueer performance"""
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)

        # Train
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        # Sla history op
        self.histories[model_name] = history.history

        # Evalueer op test set
        y_pred = model.predict(self.X_test, verbose=0)

        # Inverse transform
        y_pred_orig = self.scaler_y.inverse_transform(y_pred)
        y_test_orig = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1))

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

        print(f"{model_name}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

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
            axes[idx, 0].plot(history['loss'], label='Train Loss', linewidth=2)
            axes[idx, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
            axes[idx, 0].set_xlabel('Epoch')
            axes[idx, 0].set_ylabel('Loss (MSE)')
            axes[idx, 0].set_title(f'{model_name} - Loss Curves')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].set_yscale('log')

            # MAE curves
            axes[idx, 1].plot(history['mae'], label='Train MAE', linewidth=2)
            axes[idx, 1].plot(history['val_mae'], label='Val MAE', linewidth=2)
            axes[idx, 1].set_xlabel('Epoch')
            axes[idx, 1].set_ylabel('MAE')
            axes[idx, 1].set_title(f'{model_name} - MAE Curves')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)

        plt.suptitle(f'DNN Learning Curves - {self.data_name}', fontsize=16, y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves opgeslagen: {save_path}")
        else:
            plt.show()

    def visualize_predictions(self, model, model_name, save_path=None):
        """Visualiseer voorspellingen vs werkelijke waarden"""
        y_pred = model.predict(self.X_test, verbose=0)

        # Inverse transform
        y_pred_orig = self.scaler_y.inverse_transform(y_pred)
        y_test_orig = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1))

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

        # Error distribution
        errors = np.abs(residuals.flatten())
        axes[1, 0].hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Absolute Error (kW)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'{model_name} - Absolute Error Distribution')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Q-Q plot (residuals normality check)
        from scipy import stats
        stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{model_name} - Q-Q Plot (Normality Check)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'DNN Prediction Analysis - {self.data_name}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction visualisatie opgeslagen: {save_path}")
        else:
            plt.show()

    def compare_depths(self):
        """Vergelijk verschillende network depths"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN NETWORK DEPTHS")
        print("="*70)

        architectures = [
            {"name": "Shallow (3 layers)", "layers": [256, 128, 64]},
            {"name": "Medium (5 layers)", "layers": [512, 256, 128, 64, 32]},
            {"name": "Deep (7 layers)", "layers": [512, 256, 128, 64, 32, 16, 8]},
            {"name": "Very Deep (10 layers)", "layers": [512, 512, 256, 256, 128, 128, 64, 64, 32, 16]},
        ]

        for arch in architectures:
            print(f"\nTrainen: {arch['name']} - {len(arch['layers'])} layers")
            model = self.create_dnn(layers=arch['layers'], dropout=0.3, learning_rate=0.001)
            self.train_and_evaluate(arch['name'], model, epochs=100, batch_size=32)

    def compare_activations(self):
        """Vergelijk verschillende activation functions"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN ACTIVATION FUNCTIONS")
        print("="*70)

        activations = ['relu', 'tanh', 'sigmoid', 'elu']

        for act in activations:
            print(f"\nTrainen met activation={act}")
            model = self.create_dnn(layers=[256, 128, 64, 32], activation=act, dropout=0.3)
            self.train_and_evaluate(f"Activation {act}", model, epochs=100, batch_size=32)

    def compare_regularization(self):
        """Vergelijk verschillende regularization strategieën"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN REGULARIZATION")
        print("="*70)

        configs = [
            {"name": "No Regularization", "dropout": 0.0, "l2_reg": 0.0, "batch_norm": False},
            {"name": "Dropout only (0.3)", "dropout": 0.3, "l2_reg": 0.0, "batch_norm": False},
            {"name": "L2 only (0.001)", "dropout": 0.0, "l2_reg": 0.001, "batch_norm": False},
            {"name": "BatchNorm only", "dropout": 0.0, "l2_reg": 0.0, "batch_norm": True},
            {"name": "Dropout + BatchNorm", "dropout": 0.3, "l2_reg": 0.0, "batch_norm": True},
            {"name": "All Combined", "dropout": 0.3, "l2_reg": 0.001, "batch_norm": True},
        ]

        for config in configs:
            print(f"\nTrainen: {config['name']}")
            model = self.create_dnn(
                layers=[256, 128, 64, 32],
                dropout=config['dropout'],
                l2_reg=config['l2_reg'],
                use_batch_norm=config['batch_norm']
            )
            self.train_and_evaluate(config['name'], model, epochs=100, batch_size=32)

    def compare_optimizers(self):
        """Vergelijk verschillende optimizers"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN OPTIMIZERS")
        print("="*70)

        optimizers = ['adam', 'sgd', 'rmsprop']

        for opt in optimizers:
            print(f"\nTrainen met optimizer={opt}")
            model = self.create_dnn(
                layers=[256, 128, 64, 32],
                optimizer=opt,
                dropout=0.3,
                learning_rate=0.001 if opt == 'adam' else 0.01
            )
            self.train_and_evaluate(f"Optimizer {opt}", model, epochs=100, batch_size=32)

    def compare_widths(self):
        """Vergelijk verschillende network widths"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN NETWORK WIDTHS")
        print("="*70)

        widths = [
            {"name": "Narrow", "layers": [64, 32, 16]},
            {"name": "Medium", "layers": [128, 64, 32]},
            {"name": "Wide", "layers": [256, 128, 64]},
            {"name": "Very Wide", "layers": [512, 256, 128]},
            {"name": "Extra Wide", "layers": [1024, 512, 256]},
        ]

        for width in widths:
            print(f"\nTrainen: {width['name']}")
            model = self.create_dnn(layers=width['layers'], dropout=0.3)
            self.train_and_evaluate(width['name'], model, epochs=100, batch_size=32)

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
        axes[0, 0].set_yticklabels(df['model_name'], fontsize=7)
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Model R² Vergelijking')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # RMSE comparison
        axes[0, 1].barh(range(len(df)), df['rmse'], color='coral')
        axes[0, 1].set_yticks(range(len(df)))
        axes[0, 1].set_yticklabels(df['model_name'], fontsize=7)
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_title('Model RMSE Vergelijking')
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # MAE comparison
        axes[1, 0].barh(range(len(df)), df['mae'], color='seagreen')
        axes[1, 0].set_yticks(range(len(df)))
        axes[1, 0].set_yticklabels(df['model_name'], fontsize=7)
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].set_title('Model MAE Vergelijking')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # Training epochs
        axes[1, 1].barh(range(len(df)), df['epochs_trained'], color='mediumpurple')
        axes[1, 1].set_yticks(range(len(df)))
        axes[1, 1].set_yticklabels(df['model_name'], fontsize=7)
        axes[1, 1].set_xlabel('Epochs Trained')
        axes[1, 1].set_title('Training Duration (Epochs)')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'DNN Model Vergelijking - {self.data_name}', fontsize=16)
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
            'results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Resultaten opgeslagen: {filepath}")


def run_complete_dnn_analysis():
    """Voer volledige DNN analyse uit voor beide datasets"""

    # === SPEED TRIALS DATA ===
    print("\n" + "="*70)
    print("DNN ANALYSE: SPEED TRIALS DATA")
    print("="*70)

    analyzer_st = DNNAnalyzer(SPEED_TRIALS_REGULAR, data_name="Speed Trials")

    # Vergelijk verschillende aspecten
    analyzer_st.compare_depths()
    analyzer_st.compare_widths()
    analyzer_st.compare_activations()
    analyzer_st.compare_regularization()
    analyzer_st.compare_optimizers()

    # Visualiseer learning curves
    analyzer_st.visualize_learning_curves(
        save_path='tuning_models/visualisations/dnn_learning_curves_speed_trials.png'
    )

    # Visualiseer resultaten vergelijking
    analyzer_st.visualize_results_comparison(
        save_path='tuning_models/visualisations/dnn_comparison_speed_trials.png'
    )

    # Train beste model en visualiseer voorspellingen
    best_model_st = analyzer_st.create_dnn(
        layers=[512, 256, 128, 64, 32],
        dropout=0.3,
        learning_rate=0.001,
        use_batch_norm=True
    )
    analyzer_st.train_and_evaluate("Best Model", best_model_st, epochs=100, batch_size=32, verbose=1)

    analyzer_st.visualize_predictions(
        best_model_st, "Best Model",
        save_path='tuning_models/visualisations/dnn_predictions_speed_trials.png'
    )

    # Sla resultaten op
    analyzer_st.save_results('tuning_models/visualisations/dnn_results_speed_trials.json')

    # === WEATHER DATA ===
    print("\n" + "="*70)
    print("DNN ANALYSE: SPEED TRIALS + WEATHER DATA")
    print("="*70)

    analyzer_weather = DNNAnalyzer(SPEED_TRIALS_WEATHER_CLEAN, data_name="Speed Trials + Weather")

    # Vergelijk verschillende aspecten
    analyzer_weather.compare_depths()
    analyzer_weather.compare_widths()
    analyzer_weather.compare_activations()
    analyzer_weather.compare_regularization()
    analyzer_weather.compare_optimizers()

    # Visualiseer learning curves
    analyzer_weather.visualize_learning_curves(
        save_path='tuning_models/visualisations/dnn_learning_curves_weather.png'
    )

    # Visualiseer resultaten vergelijking
    analyzer_weather.visualize_results_comparison(
        save_path='tuning_models/visualisations/dnn_comparison_weather.png'
    )

    # Train beste model en visualiseer voorspellingen
    best_model_weather = analyzer_weather.create_dnn(
        layers=[512, 256, 128, 64, 32],
        dropout=0.3,
        learning_rate=0.001,
        use_batch_norm=True
    )
    analyzer_weather.train_and_evaluate("Best Model", best_model_weather, epochs=100, batch_size=32, verbose=1)

    analyzer_weather.visualize_predictions(
        best_model_weather, "Best Model",
        save_path='tuning_models/visualisations/dnn_predictions_weather.png'
    )

    # Sla resultaten op
    analyzer_weather.save_results('tuning_models/visualisations/dnn_results_weather.json')

    print("\n" + "="*70)
    print("DNN ANALYSE VOLTOOID")
    print("Alle visualisaties opgeslagen in tuning_models/visualisations/")
    print("="*70)


if __name__ == "__main__":
    run_complete_dnn_analysis()
