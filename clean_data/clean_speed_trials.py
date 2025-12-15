"""Clean speed trials data."""
from load_data.speed_trials import SPEED_TRIALS


def clean_speed_trials():
    """Remove rows where 50% or more values are missing and add unit conversions."""
    speed_trials = SPEED_TRIALS.copy()

    threshold = 0.5 * len(speed_trials.columns)
    speed_trials = speed_trials.dropna(thresh=threshold).reset_index(drop=True)

    # Add unit conversions
    speed_trials['OPC_07_WATER_SPEED'] = speed_trials['OPC_07_WATER_SPEED'] * 0.514444
    speed_trials['v_ms'] = 0.0
    speed_trials['GPS_HDG_HEADING_ROT_S'] = speed_trials['GPS_HDG_HEADING_ROT_S'] * 0.0174533
    speed_trials['OPC_40_PROP_RPM_FB'] = speed_trials['OPC_40_PROP_RPM_FB'] / 60.0

    return speed_trials

SPEED_TRIALS_REGULAR = clean_speed_trials()

# Save to CSV for verification
import os
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)
SPEED_TRIALS_REGULAR.to_csv(os.path.join(OUTPUT_PATH, "SPEED_TRIALS_REGULAR.csv"), index=False)
print(f"SPEED_TRIALS_REGULAR saved to: {os.path.join(OUTPUT_PATH, 'SPEED_TRIALS_REGULAR.csv')}")


if __name__ == "__main__":
    print(f"SPEED_TRIALS_REGULAR shape: {SPEED_TRIALS_REGULAR.shape}")