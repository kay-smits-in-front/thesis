"""Data analysis: features, anomaly detection, comparison."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DataAnalyzer:
    """Analyzer for features and anomaly detection."""

    def __init__(self, data, dataset_name, target_col='OPC_12_CPP_ENGINE_POWER'):
        self.data = data.copy()
        self.dataset_name = dataset_name
        self.target_col = target_col
        self.output_dir = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.exclude_cols = ["OPC_41_PITCH_FB", "OPC_13_PROP_POWER", "PROP_SHAFT_POWER_KMT", "OPC_08_GROUND_SPEED",
                             "elapsed_seconds", "hour", "minute", "second", "dataset_id",
                             "GPS_GPGGA_Latitude", "GPS_GPGGA_Longitude", "GPS_GPGGA_UTC_time"]

        self.numeric_data_all = self.data.select_dtypes(include=[np.number])
        self.numeric_data_filtered = self.numeric_data_all.drop(columns=self.exclude_cols, errors='ignore')

        print(f"\n{'='*70}")
        print(f"ANALYZING: {dataset_name}")
        print(f"{'='*70}")
        print(f"Shape: {self.data.shape}")
        print(f"All numeric features: {len(self.numeric_data_all.columns)}")
        print(f"Filtered features: {len(self.numeric_data_filtered.columns)}")
        print(f"Target: {self.target_col}")

    def plot_target_over_time(self):
        """Plot target variable over time."""
        print(f"\n{'-'*70}")
        print("TARGET OVER TIME")
        print(f"{'-'*70}")

        if self.target_col not in self.data.columns:
            print(f"Target {self.target_col} not found")
            return

        target_data = self.data[self.target_col].dropna()

        plt.figure(figsize=(16, 6))
        plt.plot(target_data.index, target_data.values, linewidth=0.5, alpha=0.7, color='steelblue')
        plt.xlabel('Index')
        plt.ylabel(self.target_col)
        plt.title(f'{self.target_col} Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_target_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: 01_target_over_time.png")

    def feature_importance_comparison(self):
        """Feature importance: all features vs filtered features."""
        print(f"\n{'-'*70}")
        print("FEATURE IMPORTANCE COMPARISON")
        print(f"{'-'*70}")

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # All features
        ax = axes[0]
        top_all = self._calc_feature_importance(self.numeric_data_all, ax, "All Features (before exclusion)")

        # Filtered features
        ax = axes[1]
        top_filtered = self._calc_feature_importance(self.numeric_data_filtered, ax, "Filtered Features (after exclusion)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_feature_importance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: 02_feature_importance_comparison.png")

        return top_filtered

    def _calc_feature_importance(self, data, ax, title):
        """Calculate and plot feature importance."""
        if self.target_col not in data.columns:
            ax.text(0.5, 0.5, 'Target not found', ha='center', va='center')
            return None

        X = data.drop(columns=[self.target_col], errors='ignore')
        y = data[self.target_col]

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) < 100:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
            return None

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_clean, y_clean)

        importance_df = pd.DataFrame({
            'feature': X_clean.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        top_10 = importance_df.head(10)

        ax.barh(range(len(top_10)), top_10['importance'].values, color='steelblue')
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['feature'].values)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        return top_10

    def detect_anomalies_isolation_forest(self, contamination=0.05):
        """Detect anomalies using Isolation Forest."""
        print(f"\n{'-'*70}")
        print("ANOMALY DETECTION: Isolation Forest")
        print(f"{'-'*70}")

        data_clean = self.numeric_data_filtered.dropna()

        if len(data_clean) < 100:
            print("Not enough data")
            return None, None

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)

        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        predictions = iso_forest.fit_predict(data_scaled)
        anomaly_scores = iso_forest.decision_function(data_scaled)

        anomalies = predictions == -1
        anomaly_count = anomalies.sum()

        print(f"Anomalies detected: {anomaly_count} ({(anomaly_count/len(data_clean))*100:.2f}%)")

        # PCA for visualization
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # PCA scatter
        ax = axes[0]
        ax.scatter(data_pca[~anomalies, 0], data_pca[~anomalies, 1], c='blue', alpha=0.5, s=20, label='Normal')
        ax.scatter(data_pca[anomalies, 0], data_pca[anomalies, 1], c='red', alpha=0.8, s=50, marker='x', label='Anomaly')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'Isolation Forest: {anomaly_count} anomalies')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Anomaly score distribution
        ax = axes[1]
        ax.hist(anomaly_scores, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.percentile(anomaly_scores, contamination*100), color='red', linestyle='--',
                   linewidth=2, label=f'Threshold (bottom {contamination*100:.0f}%)')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Anomaly Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_anomaly_isolation_forest.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: 03_anomaly_isolation_forest.png")

        return anomalies, data_clean.index[anomalies].tolist()

    def detect_anomalies_lof(self, n_neighbors=20, contamination=0.05):
        """Detect anomalies using Local Outlier Factor."""
        print(f"\n{'-'*70}")
        print("ANOMALY DETECTION: Local Outlier Factor")
        print(f"{'-'*70}")

        data_clean = self.numeric_data_filtered.dropna()

        if len(data_clean) < 100:
            print("Not enough data")
            return None, None

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
        predictions = lof.fit_predict(data_scaled)
        lof_scores = -lof.negative_outlier_factor_

        anomalies = predictions == -1
        anomaly_count = anomalies.sum()

        print(f"Anomalies detected: {anomaly_count} ({(anomaly_count/len(data_clean))*100:.2f}%)")

        # PCA for visualization
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # PCA scatter
        ax = axes[0]
        ax.scatter(data_pca[~anomalies, 0], data_pca[~anomalies, 1], c='blue', alpha=0.5, s=20, label='Normal')
        ax.scatter(data_pca[anomalies, 0], data_pca[anomalies, 1], c='red', alpha=0.8, s=50, marker='x', label='Anomaly')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'LOF: {anomaly_count} anomalies')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # LOF score distribution
        ax = axes[1]
        ax.hist(lof_scores, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.percentile(lof_scores, (1-contamination)*100), color='red', linestyle='--',
                   linewidth=2, label=f'Threshold (top {contamination*100:.0f}%)')
        ax.set_xlabel('LOF Score')
        ax.set_ylabel('Frequency')
        ax.set_title('LOF Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_anomaly_lof.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: 04_anomaly_lof.png")

        return anomalies, data_clean.index[anomalies].tolist()

    def compare_anomalies(self, iso_anomalies, lof_anomalies):
        """Compare anomalies detected by both methods."""
        print(f"\n{'-'*70}")
        print("ANOMALY COMPARISON")
        print(f"{'-'*70}")

        if iso_anomalies is None or lof_anomalies is None:
            print("Cannot compare - missing anomaly data")
            return

        data_clean = self.numeric_data_filtered.dropna()

        # Overlap
        both = iso_anomalies & lof_anomalies
        only_iso = iso_anomalies & ~lof_anomalies
        only_lof = ~iso_anomalies & lof_anomalies
        neither = ~iso_anomalies & ~lof_anomalies

        both_count = both.sum()
        only_iso_count = only_iso.sum()
        only_lof_count = only_lof.sum()
        neither_count = neither.sum()

        print(f"Both methods: {both_count}")
        print(f"Only Isolation Forest: {only_iso_count}")
        print(f"Only LOF: {only_lof_count}")
        print(f"Neither (normal): {neither_count}")

        # PCA for visualization
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Comparison scatter
        ax = axes[0]
        ax.scatter(data_pca[neither, 0], data_pca[neither, 1], c='blue', alpha=0.3, s=20, label=f'Normal ({neither_count})')
        ax.scatter(data_pca[only_iso, 0], data_pca[only_iso, 1], c='orange', alpha=0.7, s=40, label=f'Only IF ({only_iso_count})')
        ax.scatter(data_pca[only_lof, 0], data_pca[only_lof, 1], c='green', alpha=0.7, s=40, label=f'Only LOF ({only_lof_count})')
        ax.scatter(data_pca[both, 0], data_pca[both, 1], c='red', alpha=0.9, s=60, marker='x', label=f'Both ({both_count})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Anomaly Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Venn-style summary
        ax = axes[1]
        categories = ['Both Methods', 'Only IF', 'Only LOF', 'Normal']
        counts = [both_count, only_iso_count, only_lof_count, neither_count]
        colors = ['red', 'orange', 'green', 'blue']

        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Anomaly Detection Summary')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count}\n({count/len(data_clean)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_anomaly_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: 05_anomaly_comparison.png")

        # Recommendation
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)

        overlap_pct = both_count / (both_count + only_iso_count + only_lof_count) * 100 if (both_count + only_iso_count + only_lof_count) > 0 else 0

        if overlap_pct > 50:
            print(f"✓ High agreement ({overlap_pct:.1f}% overlap)")
            print(f"  → Remove {both_count} anomalies detected by BOTH methods")
            print(f"  → These are likely true anomalies")
        else:
            print(f"⚠ Low agreement ({overlap_pct:.1f}% overlap)")
            print(f"  → Methods disagree on what is anomalous")
            print(f"  → Option 1: Remove only {both_count} points detected by BOTH (conservative)")
            print(f"  → Option 2: Remove all {both_count + only_iso_count + only_lof_count} flagged points (aggressive)")
            print(f"  → Recommendation: Use conservative approach (only BOTH)")

        return {
            'both': both_count,
            'only_iso': only_iso_count,
            'only_lof': only_lof_count,
            'normal': neither_count,
            'both_indices': data_clean.index[both].tolist()
        }

    def run_analysis(self):
        """Run complete analysis."""
        print(f"\n{'='*70}")
        print(f"RUNNING ANALYSIS: {self.dataset_name}")
        print(f"{'='*70}")

        self.plot_target_over_time()
        self.feature_importance_comparison()
        iso_anomalies, iso_indices = self.detect_anomalies_isolation_forest()
        lof_anomalies, lof_indices = self.detect_anomalies_lof()
        comparison = self.compare_anomalies(iso_anomalies, lof_anomalies)

        print(f"\n{'='*70}")
        print(f"COMPLETE: {self.dataset_name}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")

        return comparison


def main():
    from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
    from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN

    print("\n" + "="*70)
    print("DATA ANALYSIS")
    print("="*70)

    analyzer1 = DataAnalyzer(SPEED_TRIALS_REGULAR, 'speed_trials')
    result1 = analyzer1.run_analysis()

    analyzer2 = DataAnalyzer(SPEED_TRIALS_WEATHER_CLEAN, 'speed_trials_weather')
    result2 = analyzer2.run_analysis()

    print("\n" + "="*70)
    print("DONE")
    print(f"Results: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()