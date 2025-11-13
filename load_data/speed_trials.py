"""Module to load speed trials data from an Excel file."""
import pandas as pd

DATA_PATH = r"/mnt/c/Users/KaySmitsInfront/PycharmProjects/thesis_KaySmits_Wärtsilä/speed_trials.xlsx"

def load_speed_trials(data_path):
    """Load and clean speed trials data from Excel file."""
    speed_trials = pd.read_excel(data_path, header=None)
    print("Speed trials data loaded with shape:", speed_trials.shape)
    return speed_trials

SPEED_TRIALS = load_speed_trials(DATA_PATH)
