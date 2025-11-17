"""
Master script om alle model analyses uit te voeren
Gebruik dit om de complete analyse suite te draaien
"""

import sys
import time
from datetime import datetime

def print_header(text):
    """Print een mooie header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_section(text):
    """Print een sectie header"""
    print("\n" + "-"*80)
    print(f"  {text}")
    print("-"*80)

def run_analysis(name, module_name, function_name):
    """Voer een enkele analyse uit met timing"""
    print_section(f"Start: {name}")
    start_time = time.time()

    try:
        # Dynamisch importeren en uitvoeren
        module = __import__(module_name)
        analysis_func = getattr(module, function_name)
        analysis_func()

        elapsed = time.time() - start_time
        print_section(f"✅ Voltooid: {name} (duur: {elapsed/60:.1f} minuten)")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print_section(f"❌ Fout in {name}: {str(e)}")
        print(f"Duur tot fout: {elapsed/60:.1f} minuten")
        return False, elapsed


def main():
    """Hoofdfunctie om alle analyses uit te voeren"""

    print_header("MODEL ANALYSE SUITE - THESIS")
    print(f"Start tijd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nDit script voert alle model analyses uit en genereert visualisaties.")
    print("Geschatte totale duur: 1-2 uur\n")

    # Vraag bevestiging
    response = input("Wil je doorgaan met alle analyses? (ja/nee): ").lower()
    if response not in ['ja', 'yes', 'y', 'j']:
        print("Analyses geannuleerd.")
        return

    total_start = time.time()
    results = {}

    # Analyses in optimale volgorde (minst → meest tijdrovend)
    analyses = [
        {
            'name': 'Random Forest Analyse',
            'module': 'randomforest_analysis',
            'function': 'run_complete_rf_analysis',
            'description': 'Feature importance, hyperparameter tuning, overfitting analyse'
        },
        {
            'name': 'MLP Analyse',
            'module': 'mlp_analysis',
            'function': 'run_complete_mlp_analysis',
            'description': 'Architectuur vergelijking, dropout, learning rates'
        },
        {
            'name': 'DNN Analyse',
            'module': 'dnn_analysis',
            'function': 'run_complete_dnn_analysis',
            'description': 'Deep networks, activations, regularization, optimizers'
        },
        {
            'name': 'LSTM Analyse',
            'module': 'lstm_analysis',
            'function': 'run_complete_lstm_analysis',
            'description': 'Sequence models, architecturen, timesteps'
        },
    ]

    # Voer elke analyse uit
    for i, analysis in enumerate(analyses, 1):
        print_header(f"Analyse {i}/{len(analyses)}: {analysis['name']}")
        print(f"Beschrijving: {analysis['description']}")

        success, duration = run_analysis(
            analysis['name'],
            analysis['module'],
            analysis['function']
        )

        results[analysis['name']] = {
            'success': success,
            'duration': duration
        }

        if not success:
            response = input(f"\nAnalyse gefaald. Doorgaan met volgende? (ja/nee): ").lower()
            if response not in ['ja', 'yes', 'y', 'j']:
                print("Analyses gestopt.")
                break

    # Samenvatting
    total_time = time.time() - total_start

    print_header("ANALYSE SAMENVATTING")
    print(f"Totale duur: {total_time/60:.1f} minuten ({total_time/3600:.2f} uur)")
    print(f"Eind tijd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("Resultaten per analyse:")
    for name, result in results.items():
        status = "✅ Succes" if result['success'] else "❌ Gefaald"
        duration = result['duration'] / 60
        print(f"  {name:30s} {status:12s} ({duration:.1f} min)")

    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)

    print(f"\nSuccesvolle analyses: {successful}/{total}")

    if successful == total:
        print("\n🎉 Alle analyses succesvol voltooid!")
        print("Bekijk de visualisaties in: tuning_models/visualisations/")
    else:
        print("\n⚠️  Sommige analyses zijn gefaald. Controleer de output hierboven.")

    print_header("KLAAR")


if __name__ == "__main__":
    # Check of we in de juiste directory zijn
    import os
    if not os.path.exists('tuning_models'):
        print("FOUT: Voer dit script uit vanuit de thesis root directory!")
        print("Gebruik: python tuning_models/run_all_analyses.py")
        sys.exit(1)

    main()
