"""
Snelle Analyse - Belangrijkste Visualisaties
Voor een snel overzicht zonder uitgebreide hyperparameter tuning
Geschatte duur: 15-20 minuten
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from mlp_analysis import MLPAnalyzer
from randomforest_analysis import RandomForestAnalyzer
from lstm_analysis import LSTMAnalyzer
from dnn_analysis import DNNAnalyzer

from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN

sns.set_style("whitegrid")


def quick_model_comparison():
    """Maak een snelle vergelijking van alle modellen met standaard configuraties"""

    print("\n" + "="*80)
    print("  SNELLE MODEL VERGELIJKING")
    print("="*80)
    print("\nTraint één configuratie per model type voor snelle vergelijking")
    print("Dit geeft een goed eerste inzicht in model performance\n")

    results = []

    # ========== RANDOM FOREST ==========
    print("\n[1/4] Random Forest...")
    rf_analyzer = RandomForestAnalyzer(SPEED_TRIALS_REGULAR, "Speed Trials")
    rf_model, rf_result = rf_analyzer.train_and_evaluate(
        "Random Forest",
        n_estimators=100,
        max_depth=20
    )
    results.append(rf_result)

    # Feature importance
    rf_analyzer.visualize_feature_importance(
        rf_model, "Random Forest - Quick Analysis",
        save_path='tuning_models/visualisations/quick_rf_importance.png'
    )

    # ========== MLP ==========
    print("\n[2/4] MLP (Multi-Layer Perceptron)...")
    mlp_analyzer = MLPAnalyzer(SPEED_TRIALS_REGULAR, "Speed Trials")
    mlp_model = mlp_analyzer.create_mlp(layers=[128, 64, 32], dropout=0.2, learning_rate=0.001)
    mlp_result = mlp_analyzer.train_and_evaluate("MLP", mlp_model, epochs=50, batch_size=32)
    results.append(mlp_result)

    mlp_analyzer.visualize_learning_curves(
        save_path='tuning_models/visualisations/quick_mlp_curves.png'
    )

    # ========== DNN ==========
    print("\n[3/4] DNN (Deep Neural Network)...")
    dnn_analyzer = DNNAnalyzer(SPEED_TRIALS_REGULAR, "Speed Trials")
    dnn_model = dnn_analyzer.create_dnn(
        layers=[256, 128, 64, 32],
        dropout=0.3,
        use_batch_norm=True
    )
    dnn_result = dnn_analyzer.train_and_evaluate("DNN", dnn_model, epochs=50, batch_size=32)
    results.append(dnn_result)

    dnn_analyzer.visualize_learning_curves(
        save_path='tuning_models/visualisations/quick_dnn_curves.png'
    )

    # ========== LSTM ==========
    print("\n[4/4] LSTM (Long Short-Term Memory)...")
    lstm_analyzer = LSTMAnalyzer(SPEED_TRIALS_REGULAR, "Speed Trials")
    data_tuple, scaler_y = lstm_analyzer.prepare_sequences(timesteps=30)
    lstm_model = lstm_analyzer.create_lstm(layers=[64, 32, 16])
    lstm_result = lstm_analyzer.train_and_evaluate("LSTM", lstm_model, data_tuple, scaler_y, epochs=50)
    results.append(lstm_result)

    lstm_analyzer.visualize_learning_curves(
        save_path='tuning_models/visualisations/quick_lstm_curves.png'
    )

    # ========== VERGELIJKING ==========
    print("\n" + "="*80)
    print("  RESULTATEN OVERZICHT")
    print("="*80)

    # Maak vergelijkings dataframe
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    # Visualiseer vergelijking
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # R² vergelijking
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
    axes[0, 0].bar(df['model_name'], df['r2'], color=colors)
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Score Vergelijking')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1])

    # RMSE vergelijking
    axes[0, 1].bar(df['model_name'], df['rmse'], color=colors)
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE Vergelijking (lager = beter)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # MAE vergelijking
    axes[1, 0].bar(df['model_name'], df['mae'], color=colors)
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAE Vergelijking (lager = beter)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Training tijd (epochs)
    axes[1, 1].bar(df['model_name'], df['epochs_trained'], color=colors)
    axes[1, 1].set_ylabel('Epochs')
    axes[1, 1].set_title('Training Duur (epochs)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Snelle Model Vergelijking - Speed Trials Data', fontsize=16)
    plt.tight_layout()
    plt.savefig('tuning_models/visualisations/quick_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVergelijking opgeslagen: tuning_models/visualisations/quick_model_comparison.png")

    # Beste model
    best_idx = df['r2'].idxmax()
    best_model = df.iloc[best_idx]

    print("\n" + "="*80)
    print("  BESTE MODEL")
    print("="*80)
    print(f"\nModel: {best_model['model_name']}")
    print(f"R²: {best_model['r2']:.4f}")
    print(f"RMSE: {best_model['rmse']:.2f}")
    print(f"MAE: {best_model['mae']:.2f}")

    # Sla resultaten op
    output = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'quick_comparison',
        'dataset': 'Speed Trials',
        'results': results,
        'best_model': {
            'name': best_model['model_name'],
            'r2': float(best_model['r2']),
            'rmse': float(best_model['rmse']),
            'mae': float(best_model['mae'])
        }
    }

    import json
    with open('tuning_models/visualisations/quick_analysis_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResultaten opgeslagen: tuning_models/visualisations/quick_analysis_results.json")

    return results


def analyze_weather_impact():
    """Analyseer de impact van weather features"""

    print("\n" + "="*80)
    print("  WEATHER IMPACT ANALYSE")
    print("="*80)
    print("\nVergelijkt model performance met en zonder weather data\n")

    results = []

    # Test met Random Forest (snel)
    print("\n[1/2] Training zonder weather data...")
    rf_no_weather = RandomForestAnalyzer(SPEED_TRIALS_REGULAR, "Zonder Weather")
    model_nw, result_nw = rf_no_weather.train_and_evaluate(
        "RF - Zonder Weather",
        n_estimators=100
    )
    results.append(result_nw)

    print("\n[2/2] Training met weather data...")
    rf_weather = RandomForestAnalyzer(SPEED_TRIALS_WEATHER_CLEAN, "Met Weather")
    model_w, result_w = rf_weather.train_and_evaluate(
        "RF - Met Weather",
        n_estimators=100
    )
    results.append(result_w)

    # Feature importance vergelijking
    fi_weather = rf_weather.visualize_feature_importance(
        model_w, "Random Forest - Met Weather Features", top_n=25,
        save_path='tuning_models/visualisations/quick_weather_features.png'
    )

    # Vergelijking
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("  WEATHER IMPACT RESULTATEN")
    print("="*80)
    print("\n" + df.to_string(index=False))

    improvement_r2 = (result_w['r2_test'] - result_nw['r2_test']) / result_nw['r2_test'] * 100
    improvement_rmse = (result_nw['rmse_test'] - result_w['rmse_test']) / result_nw['rmse_test'] * 100

    print(f"\nImpact van Weather Features:")
    print(f"  R² verbetering: {improvement_r2:+.2f}%")
    print(f"  RMSE verbetering: {improvement_rmse:+.2f}%")

    # Visualiseer
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ['r2_test', 'rmse_test', 'mae_test']
    titles = ['R² Score', 'RMSE', 'MAE']
    colors_pair = [['lightcoral', 'darkgreen'], ['lightcoral', 'darkgreen'], ['lightcoral', 'darkgreen']]

    for idx, (metric, title, colors) in enumerate(zip(metrics, titles, colors_pair)):
        values = df[metric].values
        axes[idx].bar(['Zonder Weather', 'Met Weather'], values, color=colors)
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(title)
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Voeg percentage toe
        if idx == 0:  # R²
            pct = improvement_r2
        else:  # RMSE of MAE
            pct = improvement_rmse

        axes[idx].text(1, values[1], f'{pct:+.1f}%',
                      ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.suptitle('Impact van Weather Features op Model Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig('tuning_models/visualisations/quick_weather_impact.png', dpi=300, bbox_inches='tight')
    print("\nWeather impact opgeslagen: tuning_models/visualisations/quick_weather_impact.png")

    return results


def main():
    """Hoofdfunctie voor snelle analyse"""

    print("\n" + "="*80)
    print("  SNELLE MODEL ANALYSE")
    print("  Geschatte duur: 15-20 minuten")
    print("="*80)

    start_time = datetime.now()
    print(f"\nStart: {start_time.strftime('%H:%M:%S')}")

    # Model vergelijking
    model_results = quick_model_comparison()

    # Weather impact
    weather_results = analyze_weather_impact()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print("\n" + "="*80)
    print("  SNELLE ANALYSE VOLTOOID")
    print("="*80)
    print(f"\nStart: {start_time.strftime('%H:%M:%S')}")
    print(f"Eind:  {end_time.strftime('%H:%M:%S')}")
    print(f"Duur:  {duration:.1f} minuten")

    print("\n📊 Gegenereerde visualisaties:")
    print("  - tuning_models/visualisations/quick_model_comparison.png")
    print("  - tuning_models/visualisations/quick_weather_impact.png")
    print("  - tuning_models/visualisations/quick_rf_importance.png")
    print("  - tuning_models/visualisations/quick_mlp_curves.png")
    print("  - tuning_models/visualisations/quick_dnn_curves.png")
    print("  - tuning_models/visualisations/quick_lstm_curves.png")
    print("  - tuning_models/visualisations/quick_weather_features.png")

    print("\n📄 Resultaten:")
    print("  - tuning_models/visualisations/quick_analysis_results.json")

    print("\n💡 Voor uitgebreide analyse, gebruik:")
    print("  python tuning_models/run_all_analyses.py")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
