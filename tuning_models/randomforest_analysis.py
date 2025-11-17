"""
Uitgebreide Random Forest Analyse en Tuning
Bevat: feature importance, hyperparameter tuning, tree diagnostics, en performance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import json

from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN

# Styling voor plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class RandomForestAnalyzer:
    """Klasse voor uitgebreide Random Forest analyse en visualisatie"""

    def __init__(self, data, data_name="Speed Trials", target_col="OPC_12_CPP_ENGINE_POWER"):
        self.data = data
        self.data_name = data_name
        self.target_col = target_col
        self.results = []

        # Bereid data voor
        self._prepare_data()

    def _prepare_data(self):
        """Bereid data voor met train/test split"""
        # Drop kolommen
        col_drop = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED"]

        self.X = self.data.drop(columns=col_drop + [self.target_col])
        self.y = self.data[self.target_col]

        # Train/test split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.feature_names = list(self.X.columns)

        print(f"Data voorbereid: Train={self.X_train.shape}, Test={self.X_test.shape}")
        print(f"Aantal features: {len(self.feature_names)}")

    def train_and_evaluate(self, model_name, n_estimators=100, max_depth=None, min_samples_split=2,
                          min_samples_leaf=1, max_features='sqrt', n_jobs=-1, verbose=0):
        """Train Random Forest model en evalueer performance"""

        print(f"\nTrainen: {model_name}")
        print(f"  n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"  min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
        print(f"  max_features={max_features}")

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=n_jobs,
            verbose=verbose
        )

        # Train
        model.fit(self.X_train, self.y_train)

        # Voorspellingen
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        # Metrics
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        # Overfitting check
        overfit_score = r2_train - r2_test

        result = {
            'model_name': model_name,
            'n_estimators': n_estimators,
            'max_depth': max_depth if max_depth else 'None',
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mse_test': mse_test,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'overfit_score': overfit_score
        }

        self.results.append(result)

        print(f"  R² Train={r2_train:.4f}, R² Test={r2_test:.4f}")
        print(f"  RMSE={rmse_test:.2f}, MAE={mae_test:.2f}")
        print(f"  Overfitting Score={overfit_score:.4f}")

        return model, result

    def visualize_feature_importance(self, model, model_name, top_n=20, save_path=None):
        """Visualiseer feature importance"""
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = feature_importance.head(top_n)

        ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{model_name} - Top {top_n} Feature Importances\n{self.data_name}')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance opgeslagen: {save_path}")
        else:
            plt.show()

        return feature_importance

    def visualize_predictions(self, model, model_name, save_path=None):
        """Visualiseer voorspellingen vs werkelijke waarden"""
        y_pred = model.predict(self.X_test)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot
        axes[0].scatter(self.y_test, y_pred, alpha=0.5, s=20)
        axes[0].plot([self.y_test.min(), self.y_test.max()],
                     [self.y_test.min(), self.y_test.max()],
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Werkelijke Waarden (kW)')
        axes[0].set_ylabel('Voorspelde Waarden (kW)')
        axes[0].set_title(f'{model_name} - Voorspellingen vs Werkelijkheid')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuals
        residuals = self.y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Voorspelde Waarden (kW)')
        axes[1].set_ylabel('Residuals (kW)')
        axes[1].set_title(f'{model_name} - Residual Plot')
        axes[1].grid(True, alpha=0.3)

        # Bereken statistieken
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

        plt.suptitle(f'Random Forest Prediction Analysis - {self.data_name}\nR²={r2:.4f}, RMSE={rmse:.2f}',
                    fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction visualisatie opgeslagen: {save_path}")
        else:
            plt.show()

    def compare_n_estimators(self):
        """Vergelijk verschillende aantallen trees"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN N_ESTIMATORS (aantal trees)")
        print("="*70)

        n_estimators_list = [10, 50, 100, 200, 500]

        for n_est in n_estimators_list:
            self.train_and_evaluate(
                f"RF n_est={n_est}",
                n_estimators=n_est
            )

    def compare_max_depth(self):
        """Vergelijk verschillende max_depth waarden"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN MAX_DEPTH")
        print("="*70)

        max_depth_list = [5, 10, 20, 30, None]

        for depth in max_depth_list:
            depth_str = depth if depth else "None"
            self.train_and_evaluate(
                f"RF depth={depth_str}",
                n_estimators=100,
                max_depth=depth
            )

    def compare_min_samples_split(self):
        """Vergelijk verschillende min_samples_split waarden"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN MIN_SAMPLES_SPLIT")
        print("="*70)

        min_samples_split_list = [2, 5, 10, 20, 50]

        for min_split in min_samples_split_list:
            self.train_and_evaluate(
                f"RF min_split={min_split}",
                n_estimators=100,
                min_samples_split=min_split
            )

    def compare_max_features(self):
        """Vergelijk verschillende max_features strategieën"""
        print("\n" + "="*70)
        print("VERGELIJKING VAN MAX_FEATURES")
        print("="*70)

        max_features_list = ['sqrt', 'log2', 0.5, 1.0]

        for max_feat in max_features_list:
            self.train_and_evaluate(
                f"RF max_feat={max_feat}",
                n_estimators=100,
                max_features=max_feat
            )

    def analyze_overfitting_vs_trees(self, save_path=None):
        """Analyseer relatie tussen aantal trees en overfitting"""
        print("\n" + "="*70)
        print("OVERFITTING ANALYSE: Trees vs Performance")
        print("="*70)

        n_estimators_range = range(10, 501, 20)
        train_scores = []
        test_scores = []

        for n_est in n_estimators_range:
            model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
            model.fit(self.X_train, self.y_train)

            train_score = r2_score(self.y_train, model.predict(self.X_train))
            test_score = r2_score(self.y_test, model.predict(self.X_test))

            train_scores.append(train_score)
            test_scores.append(test_score)

            if n_est % 100 == 0:
                print(f"  n_estimators={n_est}: Train R²={train_score:.4f}, Test R²={test_score:.4f}")

        # Visualiseer
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(n_estimators_range, train_scores, label='Train R²', linewidth=2, marker='o', markersize=4)
        ax.plot(n_estimators_range, test_scores, label='Test R²', linewidth=2, marker='s', markersize=4)
        ax.fill_between(n_estimators_range, train_scores, test_scores, alpha=0.2, color='gray')

        ax.set_xlabel('Number of Trees (n_estimators)')
        ax.set_ylabel('R² Score')
        ax.set_title(f'Random Forest: Train vs Test Performance\n{self.data_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overfitting analyse opgeslagen: {save_path}")
        else:
            plt.show()

    def visualize_results_comparison(self, save_path=None):
        """Visualiseer vergelijking van alle resultaten"""
        if not self.results:
            print("Geen resultaten om te visualiseren!")
            return

        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # R² test comparison
        axes[0, 0].barh(range(len(df)), df['r2_test'], color='steelblue')
        axes[0, 0].set_yticks(range(len(df)))
        axes[0, 0].set_yticklabels(df['model_name'], fontsize=8)
        axes[0, 0].set_xlabel('R² Score (Test)')
        axes[0, 0].set_title('Model R² Vergelijking')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # RMSE comparison
        axes[0, 1].barh(range(len(df)), df['rmse_test'], color='coral')
        axes[0, 1].set_yticks(range(len(df)))
        axes[0, 1].set_yticklabels(df['model_name'], fontsize=8)
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_title('Model RMSE Vergelijking')
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # Overfitting score
        axes[1, 0].barh(range(len(df)), df['overfit_score'], color='indianred')
        axes[1, 0].set_yticks(range(len(df)))
        axes[1, 0].set_yticklabels(df['model_name'], fontsize=8)
        axes[1, 0].set_xlabel('Overfitting Score (Train R² - Test R²)')
        axes[1, 0].set_title('Overfitting Analysis')
        axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # Train vs Test R²
        axes[1, 1].scatter(df['r2_train'], df['r2_test'], s=100, alpha=0.6, color='seagreen')
        axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect (no overfit)')
        for i, name in enumerate(df['model_name']):
            axes[1, 1].annotate(name, (df['r2_train'].iloc[i], df['r2_test'].iloc[i]),
                              fontsize=7, alpha=0.7)
        axes[1, 1].set_xlabel('Train R²')
        axes[1, 1].set_ylabel('Test R²')
        axes[1, 1].set_title('Train vs Test R² (Overfitting Check)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Random Forest Model Vergelijking - {self.data_name}', fontsize=16)
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


def run_complete_rf_analysis():
    """Voer volledige Random Forest analyse uit voor beide datasets"""

    # === SPEED TRIALS DATA ===
    print("\n" + "="*70)
    print("RANDOM FOREST ANALYSE: SPEED TRIALS DATA")
    print("="*70)

    analyzer_st = RandomForestAnalyzer(SPEED_TRIALS_REGULAR, data_name="Speed Trials")

    # Vergelijk hyperparameters
    analyzer_st.compare_n_estimators()
    analyzer_st.compare_max_depth()
    analyzer_st.compare_min_samples_split()
    analyzer_st.compare_max_features()

    # Overfitting analyse
    analyzer_st.analyze_overfitting_vs_trees(
        save_path='tuning_models/visualisations/rf_overfitting_analysis_speed_trials.png'
    )

    # Visualiseer resultaten vergelijking
    analyzer_st.visualize_results_comparison(
        save_path='tuning_models/visualisations/rf_comparison_speed_trials.png'
    )

    # Train beste model
    best_model_st, _ = analyzer_st.train_and_evaluate(
        "Best Model",
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt'
    )

    # Visualiseer feature importance
    analyzer_st.visualize_feature_importance(
        best_model_st, "Best Model",
        save_path='tuning_models/visualisations/rf_feature_importance_speed_trials.png'
    )

    # Visualiseer voorspellingen
    analyzer_st.visualize_predictions(
        best_model_st, "Best Model",
        save_path='tuning_models/visualisations/rf_predictions_speed_trials.png'
    )

    # Sla resultaten op
    analyzer_st.save_results('tuning_models/visualisations/rf_results_speed_trials.json')

    # === WEATHER DATA ===
    print("\n" + "="*70)
    print("RANDOM FOREST ANALYSE: SPEED TRIALS + WEATHER DATA")
    print("="*70)

    analyzer_weather = RandomForestAnalyzer(SPEED_TRIALS_WEATHER_CLEAN, data_name="Speed Trials + Weather")

    # Vergelijk hyperparameters
    analyzer_weather.compare_n_estimators()
    analyzer_weather.compare_max_depth()
    analyzer_weather.compare_min_samples_split()
    analyzer_weather.compare_max_features()

    # Overfitting analyse
    analyzer_weather.analyze_overfitting_vs_trees(
        save_path='tuning_models/visualisations/rf_overfitting_analysis_weather.png'
    )

    # Visualiseer resultaten vergelijking
    analyzer_weather.visualize_results_comparison(
        save_path='tuning_models/visualisations/rf_comparison_weather.png'
    )

    # Train beste model
    best_model_weather, _ = analyzer_weather.train_and_evaluate(
        "Best Model",
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt'
    )

    # Visualiseer feature importance
    analyzer_weather.visualize_feature_importance(
        best_model_weather, "Best Model",
        save_path='tuning_models/visualisations/rf_feature_importance_weather.png'
    )

    # Visualiseer voorspellingen
    analyzer_weather.visualize_predictions(
        best_model_weather, "Best Model",
        save_path='tuning_models/visualisations/rf_predictions_weather.png'
    )

    # Sla resultaten op
    analyzer_weather.save_results('tuning_models/visualisations/rf_results_weather.json')

    print("\n" + "="*70)
    print("RANDOM FOREST ANALYSE VOLTOOID")
    print("Alle visualisaties opgeslagen in tuning_models/visualisations/")
    print("="*70)


if __name__ == "__main__":
    run_complete_rf_analysis()
