"""Check dataset dimensions and column counts."""
import pandas as pd
from pathlib import Path


EXCLUDE_COLS = [
	"OPC_12_CPP_ENGINE_POWER",
	"OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
	"elapsed_seconds", "hour", "minute", "second", "dataset_id",
	"GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time", "Date", "Time",
	"OPC_17_VES_DRAFT_MID_SB", "OPC_14_VES_DRAFT_FWD", "OPC_16_VES_DRAFT_MID_PS", "OPC_15_VES_DRAFT_AFT"
]


def check_excel_file(filepath):
	df = pd.read_excel(filepath)
	return df.shape[0], df.shape[1]


def check_csv_file(filepath):
	df = pd.read_csv(filepath)
	return df.shape[0], df.shape[1]


def count_model_columns(csv_filepath):
	df = pd.read_csv(csv_filepath)
	total_cols = df.shape[1]
	exclude_present = [col for col in EXCLUDE_COLS if col in df.columns]
	model_cols = total_cols - len(exclude_present)
	return total_cols, len(exclude_present), model_cols


def main():
	project_root = Path(__file__).parent

	print("=== Initial Data ===")
	speed_rows, speed_cols = check_excel_file(project_root / "speed_trials.xlsx")
	print(f"speed_trials.xlsx: {speed_rows} rows, {speed_cols} columns")

	weather_rows, weather_cols = check_excel_file(project_root / "sea_trial_GPS_weather.xlsx")
	print(f"sea_trial_GPS_weather.xlsx: {weather_rows} rows, {weather_cols} columns")

	print("\n=== Final Data ===")
	regular_rows, regular_cols = check_csv_file(project_root / "output" / "SPEED_TRIALS_REGULAR_FINAL.csv")
	print(f"SPEED_TRIALS_REGULAR_FINAL.csv: {regular_rows} rows, {regular_cols} columns")

	weather_final_rows, weather_final_cols = check_csv_file(project_root / "output" / "SPEED_TRIALS_WEATHER_FINAL.csv")
	print(f"SPEED_TRIALS_WEATHER_FINAL.csv: {weather_final_rows} rows, {weather_final_cols} columns")

	print("\n=== Model Input - REGULAR ===")
	total, excluded, remaining = count_model_columns(project_root / "output" / "SPEED_TRIALS_REGULAR_FINAL.csv")
	print(f"Total columns: {total}")
	print(f"Excluded columns: {excluded}")
	print(f"Columns for models: {remaining}")

	print("\n=== Model Input - WEATHER ===")
	total, excluded, remaining = count_model_columns(project_root / "output" / "SPEED_TRIALS_WEATHER_FINAL.csv")
	print(f"Total columns: {total}")
	print(f"Excluded columns: {excluded}")
	print(f"Columns for models: {remaining}")


if __name__ == "__main__":
	main()