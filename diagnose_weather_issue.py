"""
Diagnostic script to identify why weather data performs worse than regular data.
Run this AFTER running pre_process.py to generate the datasets.
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "output"

def diagnose_weather_data_issue():
    """Comprehensive diagnosis of weather data underperformance."""

    print("\n" + "="*80)
    print("WEATHER DATA DIAGNOSTIC REPORT")
    print("="*80)

    # Check if files exist
    regular_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_REGULAR_FINAL.csv")
    weather_path = os.path.join(OUTPUT_DIR, "SPEED_TRIALS_WEATHER_FINAL.csv")

    if not os.path.exists(regular_path) or not os.path.exists(weather_path):
        print("\n❌ ERROR: Run pre_process.py first to generate datasets!")
        return

    # Load datasets
    regular = pd.read_csv(regular_path)
    weather = pd.read_csv(weather_path)

    print(f"\n{'='*80}")
    print("ISSUE #1: SAMPLE SIZE COMPARISON")
    print('='*80)
    print(f"\nRegular dataset: {len(regular):,} samples")
    print(f"Weather dataset: {len(weather):,} samples")
    sample_loss = len(regular) - len(weather)
    sample_loss_pct = (sample_loss / len(regular)) * 100
    print(f"\nSample loss: {sample_loss:,} samples ({sample_loss_pct:.2f}%)")

    if sample_loss > 0:
        print("\n⚠️  PROBLEM: Weather dataset has FEWER samples!")
        print("   This reduces training data and could hurt performance.")
        print("   Likely cause: Rows dropped during NaN removal after interpolation.")

    # Compare feature counts
    print(f"\n{'='*80}")
    print("ISSUE #2: FEATURE COUNT COMPARISON")
    print('='*80)
    regular_cols = set(regular.columns)
    weather_cols = set(weather.columns)
    weather_only = weather_cols - regular_cols

    print(f"\nRegular features: {len(regular_cols)}")
    print(f"Weather features: {len(weather_cols)}")
    print(f"\nWeather-only features ({len(weather_only)}):")
    for col in sorted(weather_only):
        print(f"  - {col}")

    # Check for elapsed_seconds inconsistency
    print(f"\n{'='*80}")
    print("ISSUE #3: elapsed_seconds INCONSISTENCY")
    print('='*80)
    if 'elapsed_seconds' in weather_only:
        print("\n⚠️  PROBLEM: 'elapsed_seconds' is in weather data but NOT in regular data!")
        print("   This creates feature space mismatch between datasets.")
        print("   Models trained on weather data see different features.")
        print("   This is a time-based feature that could cause data leakage.")

    # Check distribution of target variable
    print(f"\n{'='*80}")
    print("ISSUE #4: TARGET VARIABLE DISTRIBUTION")
    print('='*80)
    target_col = 'OPC_12_CPP_ENGINE_POWER'

    reg_mean = regular[target_col].mean()
    reg_std = regular[target_col].std()
    weather_mean = weather[target_col].mean()
    weather_std = weather[target_col].std()

    print(f"\nRegular - Mean: {reg_mean:.2f} kW, Std: {reg_std:.2f} kW")
    print(f"Weather - Mean: {weather_mean:.2f} kW, Std: {weather_std:.2f} kW")
    print(f"\nDifference: {abs(reg_mean - weather_mean):.2f} kW ({abs((weather_mean-reg_mean)/reg_mean*100):.2f}%)")

    if abs((weather_mean - reg_mean) / reg_mean) > 0.05:
        print("\n⚠️  PROBLEM: Target distribution shifted by >5%!")
        print("   Weather preprocessing may have filtered different data patterns.")

    # Check for physics features
    print(f"\n{'='*80}")
    print("ISSUE #5: PHYSICS FEATURES COMPARISON")
    print('='*80)
    physics_features = ['OPC_07_WATER_SPEED', 'v_ms', 'GPS_HDG_HEADING_ROT_S', 'OPC_40_PROP_RPM_FB']

    print("\nComparing key physics features used in loss function:")
    for feat in physics_features:
        if feat in regular.columns and feat in weather.columns:
            reg_feat_mean = regular[feat].mean()
            weather_feat_mean = weather[feat].mean()
            diff_pct = abs((weather_feat_mean - reg_feat_mean) / (reg_feat_mean + 1e-10)) * 100

            print(f"\n{feat}:")
            print(f"  Regular: mean={reg_feat_mean:.4f}, std={regular[feat].std():.4f}")
            print(f"  Weather: mean={weather_feat_mean:.4f}, std={weather[feat].std():.4f}")
            print(f"  Difference: {diff_pct:.2f}%")

            if diff_pct > 5:
                print(f"  ⚠️  Distribution shifted by >5%!")

    # Check for NaN patterns in weather-only features
    if len(weather_only) > 0:
        print(f"\n{'='*80}")
        print("ISSUE #6: WEATHER FEATURE QUALITY")
        print('='*80)
        print("\nMissing values in weather-specific features:")

        has_missing = False
        for col in sorted(weather_only):
            if col in weather.columns:
                missing = weather[col].isna().sum()
                pct = (missing / len(weather)) * 100
                if missing > 0:
                    print(f"  {col}: {missing}/{len(weather)} ({pct:.2f}%)")
                    has_missing = True

        if not has_missing:
            print("  ✓ No missing values in weather features")
        else:
            print("\n⚠️  PROBLEM: Weather features have missing values!")
            print("   Interpolation may have failed for some features.")

    # Final summary
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print('='*80)

    issues_found = []
    if sample_loss > len(regular) * 0.01:
        issues_found.append("Sample loss > 1%")
    if 'elapsed_seconds' in weather_only:
        issues_found.append("elapsed_seconds inconsistency")
    if abs((weather_mean - reg_mean) / reg_mean) > 0.05:
        issues_found.append("Target distribution shift > 5%")

    if len(issues_found) > 0:
        print("\n⚠️  Issues Found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")

        print("\n📋 RECOMMENDED FIXES:")
        print("  1. Remove 'elapsed_seconds' from weather data (time-based feature)")
        print("  2. Investigate why sample sizes differ after preprocessing")
        print("  3. Use weather data features ADDITIVELY (keep all regular samples)")
        print("  4. Consider separate anomaly detection for weather features")
    else:
        print("\n✓ No major issues detected.")
        print("  Performance difference may be due to:")
        print("  - Model capacity not sufficient for additional features")
        print("  - Need different hyperparameters for weather data")
        print("  - Weather features may not be predictive for this task")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    diagnose_weather_data_issue()
