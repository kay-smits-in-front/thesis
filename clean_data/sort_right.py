"""Script to export raw data with all columns combined into single column."""
import pandas as pd
from load_data.speed_trials import SPEED_TRIALS


def export_all_columns_to_single(speed_trials_raw, output_file='speed_trials_single_column.xlsx'):
	"""Combine all columns into single column separated by semicolons."""

	single_column_data = []

	for idx in range(len(speed_trials_raw)):
		row_values = []
		for col_idx in range(len(speed_trials_raw.columns)):
			value = speed_trials_raw.iloc[idx, col_idx]
			if pd.isna(value):
				row_values.append('')
			else:
				row_values.append(str(value))

		combined = ','.join(row_values)
		single_column_data.append([combined])

	df_output = pd.DataFrame(single_column_data, columns=['Data'])

	df_output.to_excel(output_file, index=False)

	print(f"Saved to: {output_file}")
	print(f"Total rows: {len(df_output)}")
	print(f"Sample first row: {df_output.iloc[0, 0][:200]}")

	return df_output


SINGLE_COLUMN = export_all_columns_to_single(SPEED_TRIALS)