"""Filter data based on OPC_4 and OPC_5 values."""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "opc_filter")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def filter_opc_columns(data, min_val=-3, max_val=3):
	"""Filter data where OPC_04 and OPC_05 columns are within range."""
	opc_04_cols = [col for col in data.columns if col.startswith('OPC_04')]
	opc_05_cols = [col for col in data.columns if col.startswith('OPC_05')]

	print(f"OPC_04 columns: {opc_04_cols}")
	print(f"OPC_05 columns: {opc_05_cols}")

	mask = pd.Series(True, index=data.index)

	for col in opc_04_cols:
		mask &= (data[col] >= min_val) & (data[col] <= max_val)

	for col in opc_05_cols:
		mask &= (data[col] >= min_val) & (data[col] <= max_val)

	filtered = data[mask]

	return filtered


def plot_opc_over_time(data, dataset_name):
	"""Plot OPC_04 and OPC_05 columns over time."""
	opc_04_cols = [col for col in data.columns if col.startswith('OPC_04')]
	opc_05_cols = [col for col in data.columns if col.startswith('OPC_05')]

	fig, axes = plt.subplots(2, 1, figsize=(16, 10))

	ax = axes[0]
	for col in opc_04_cols:
		ax.plot(data.index, data[col], linewidth=0.5, alpha=0.7, label=col)
	ax.axhline(y=-3, color='red', linestyle='--', linewidth=1, alpha=0.5)
	ax.axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.5)
	ax.set_xlabel('Index')
	ax.set_ylabel('Value')
	ax.set_title(f'OPC_04 Over Time - {dataset_name}')
	ax.legend()
	ax.grid(True, alpha=0.3)

	ax = axes[1]
	for col in opc_05_cols:
		ax.plot(data.index, data[col], linewidth=0.5, alpha=0.7, label=col)
	ax.axhline(y=-3, color='red', linestyle='--', linewidth=1, alpha=0.5)
	ax.axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.5)
	ax.set_xlabel('Index')
	ax.set_ylabel('Value')
	ax.set_title(f'OPC_05 Over Time - {dataset_name}')
	ax.legend()
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	filename = os.path.join(OUTPUT_DIR, f'{dataset_name}_opc_over_time.png')
	plt.savefig(filename, dpi=300, bbox_inches='tight')
	plt.close()

	print(f"Saved: {filename}")


def main():
	print("SPEED_TRIALS_REGULAR")
	print(f"Original rows: {len(SPEED_TRIALS_REGULAR)}")
	filtered1 = filter_opc_columns(SPEED_TRIALS_REGULAR)
	print(f"Filtered rows: {len(filtered1)}")
	print(f"Removed: {len(SPEED_TRIALS_REGULAR) - len(filtered1)}")
	plot_opc_over_time(SPEED_TRIALS_REGULAR, 'speed_trials_regular')

	print("\n" + "="*70 + "\n")

	print("SPEED_TRIALS_WEATHER_CLEAN")
	print(f"Original rows: {len(SPEED_TRIALS_WEATHER_CLEAN)}")
	filtered2 = filter_opc_columns(SPEED_TRIALS_WEATHER_CLEAN)
	print(f"Filtered rows: {len(filtered2)}")
	print(f"Removed: {len(SPEED_TRIALS_WEATHER_CLEAN) - len(filtered2)}")
	plot_opc_over_time(SPEED_TRIALS_WEATHER_CLEAN, 'speed_trials_weather')

	print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
	main()