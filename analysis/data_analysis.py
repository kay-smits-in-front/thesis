"""Comprehensive data analysis script for speed trials data.

This script performs:
1. Outlier detection (IQR, Z-score, Isolation Forest)
2. Anomaly detection (LOF, One-Class SVM)
3. Distribution analysis (histograms, box plots, Q-Q plots, KDE)
4. Time series analysis of output features
5. Additional visualizations for better understanding
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Get the project root directory (parent of analysis folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DataAnalyzer:
    """Comprehensive data analyzer for speed trials data."""

    def __init__(self, data, dataset_name, target_col='OPC_12_CPP_ENGINE_POWER'):
        """
        Initialize the analyzer.

        Args:
            data: DataFrame containing the data
            dataset_name: Name of the dataset (for file naming)
            target_col: Name of the target/output column
        """
        self.data = data.copy()
        self.dataset_name = dataset_name
        self.target_col = target_col
        self.output_dir = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Select only numeric columns for analysis
        self.numeric_data = self.data.select_dtypes(include=[np.number])

        print(f"\n{'='*70}")
        print(f"ANALYZING DATASET: {dataset_name}")
        print(f"{'='*70}")
        print(f"Shape: {self.data.shape}")
        print(f"Numeric columns: {len(self.numeric_data.columns)}")
        print(f"Target column: {self.target_col}")

    def detect_outliers_iqr(self, threshold=1.5):
        """Detect outliers using Interquartile Range (IQR) method."""
        print(f"\n{'-'*70}")
        print("OUTLIER DETECTION: IQR Method")
        print(f"{'-'*70}")

        outliers_summary = {}

        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for idx, col in enumerate(self.numeric_data.columns[:12]):
            if idx >= 12:
                break

            Q1 = self.numeric_data[col].quantile(0.25)
            Q3 = self.numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = self.numeric_data[(self.numeric_data[col] < lower_bound) |
                                         (self.numeric_data[col] > upper_bound)]

            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.numeric_data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

            # Plot
            ax = axes[idx]
            ax.boxplot(self.numeric_data[col].dropna(), vert=True)
            ax.set_title(f'{col}\n{len(outliers)} outliers ({outliers_summary[col]["percentage"]:.2f}%)',
                        fontsize=9)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(self.numeric_data.columns[:12]), 12):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'outliers_iqr_boxplots.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Print summary
        print(f"\nOutliers detected (threshold={threshold}):")
        for col, summary in outliers_summary.items():
            print(f"  {col}: {summary['count']} ({summary['percentage']:.2f}%)")

        return outliers_summary

    def detect_outliers_zscore(self, threshold=3):
        """Detect outliers using Z-score method."""
        print(f"\n{'-'*70}")
        print(f"OUTLIER DETECTION: Z-Score Method (threshold={threshold})")
        print(f"{'-'*70}")

        outliers_summary = {}

        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for idx, col in enumerate(self.numeric_data.columns[:12]):
            if idx >= 12:
                break

            z_scores = np.abs(stats.zscore(self.numeric_data[col].dropna()))
            outliers_mask = z_scores > threshold
            outliers_count = outliers_mask.sum()

            outliers_summary[col] = {
                'count': outliers_count,
                'percentage': (outliers_count / len(self.numeric_data)) * 100
            }

            # Plot
            ax = axes[idx]
            ax.hist(z_scores, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
            ax.set_title(f'{col}\n{outliers_count} outliers ({outliers_summary[col]["percentage"]:.2f}%)',
                        fontsize=9)
            ax.set_xlabel('Z-Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(self.numeric_data.columns[:12]), 12):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'outliers_zscore_histograms.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Print summary
        print(f"\nOutliers detected:")
        for col, summary in outliers_summary.items():
            print(f"  {col}: {summary['count']} ({summary['percentage']:.2f}%)")

        return outliers_summary

    def detect_anomalies_isolation_forest(self, contamination=0.1):
        """Detect anomalies using Isolation Forest."""
        print(f"\n{'-'*70}")
        print(f"ANOMALY DETECTION: Isolation Forest (contamination={contamination})")
        print(f"{'-'*70}")

        # Prepare data
        data_clean = self.numeric_data.dropna()

        if len(data_clean) == 0:
            print("No data available after removing NaN values")
            return None

        # Scale the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(data_scaled)

        # -1 for anomalies, 1 for normal
        anomalies_mask = predictions == -1
        anomalies_count = anomalies_mask.sum()

        print(f"Anomalies detected: {anomalies_count} ({(anomalies_count/len(data_clean))*100:.2f}%)")

        # Visualize anomalies in 2D using first two principal components
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)

        plt.figure(figsize=(12, 8))
        plt.scatter(data_pca[~anomalies_mask, 0], data_pca[~anomalies_mask, 1],
                   c='blue', alpha=0.5, label='Normal', s=20)
        plt.scatter(data_pca[anomalies_mask, 0], data_pca[anomalies_mask, 1],
                   c='red', alpha=0.8, label='Anomaly', s=50, marker='x')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.title(f'Isolation Forest: Anomaly Detection\n{anomalies_count} anomalies detected')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anomalies_isolation_forest.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        return {'anomalies_count': anomalies_count,
                'anomalies_percentage': (anomalies_count/len(data_clean))*100,
                'anomalies_indices': data_clean.index[anomalies_mask].tolist()}

    def detect_anomalies_lof(self, n_neighbors=20):
        """Detect anomalies using Local Outlier Factor."""
        print(f"\n{'-'*70}")
        print(f"ANOMALY DETECTION: Local Outlier Factor (n_neighbors={n_neighbors})")
        print(f"{'-'*70}")

        # Prepare data
        data_clean = self.numeric_data.dropna()

        if len(data_clean) == 0:
            print("No data available after removing NaN values")
            return None

        # Scale the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)

        # Fit LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        predictions = lof.fit_predict(data_scaled)

        # -1 for anomalies, 1 for normal
        anomalies_mask = predictions == -1
        anomalies_count = anomalies_mask.sum()

        print(f"Anomalies detected: {anomalies_count} ({(anomalies_count/len(data_clean))*100:.2f}%)")

        # Visualize anomalies in 2D using first two principal components
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)

        plt.figure(figsize=(12, 8))
        plt.scatter(data_pca[~anomalies_mask, 0], data_pca[~anomalies_mask, 1],
                   c='blue', alpha=0.5, label='Normal', s=20)
        plt.scatter(data_pca[anomalies_mask, 0], data_pca[anomalies_mask, 1],
                   c='red', alpha=0.8, label='Anomaly', s=50, marker='x')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.title(f'Local Outlier Factor: Anomaly Detection\n{anomalies_count} anomalies detected')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anomalies_lof.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        return {'anomalies_count': anomalies_count,
                'anomalies_percentage': (anomalies_count/len(data_clean))*100,
                'anomalies_indices': data_clean.index[anomalies_mask].tolist()}

    def analyze_distributions(self):
        """Analyze and visualize distributions of all numeric features."""
        print(f"\n{'-'*70}")
        print("DISTRIBUTION ANALYSIS")
        print(f"{'-'*70}")

        # Histograms with KDE
        n_cols = min(12, len(self.numeric_data.columns))
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for idx, col in enumerate(self.numeric_data.columns[:n_cols]):
            ax = axes[idx]
            data_clean = self.numeric_data[col].dropna()

            ax.hist(data_clean, bins=50, edgecolor='black', alpha=0.7, density=True, label='Histogram')

            # Add KDE
            try:
                kde_x = np.linspace(data_clean.min(), data_clean.max(), 200)
                kde = stats.gaussian_kde(data_clean)
                ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
            except:
                pass

            ax.set_title(f'{col}\nMean={data_clean.mean():.2f}, Std={data_clean.std():.2f}',
                        fontsize=9)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_cols, 12):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'distributions_histograms_kde.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Q-Q plots
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for idx, col in enumerate(self.numeric_data.columns[:n_cols]):
            ax = axes[idx]
            data_clean = self.numeric_data[col].dropna()

            stats.probplot(data_clean, dist="norm", plot=ax)
            ax.set_title(f'{col}\nQ-Q Plot', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_cols, 12):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'distributions_qq_plots.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Violin plots
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for idx, col in enumerate(self.numeric_data.columns[:n_cols]):
            ax = axes[idx]
            data_clean = self.numeric_data[col].dropna()

            parts = ax.violinplot([data_clean], positions=[0], showmeans=True, showmedians=True)
            ax.set_title(f'{col}', fontsize=9)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])

        # Hide unused subplots
        for idx in range(n_cols, 12):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'distributions_violin_plots.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("Distribution analysis completed")

    def plot_target_over_time(self):
        """Plot the target/output feature over time."""
        print(f"\n{'-'*70}")
        print(f"TIME SERIES ANALYSIS: {self.target_col}")
        print(f"{'-'*70}")

        if self.target_col not in self.data.columns:
            print(f"Target column '{self.target_col}' not found in data")
            return

        # Check if there's a time column
        time_cols = ['elapsed_seconds', 'datetime_parsed', 'timestamp']
        time_col = None
        for col in time_cols:
            if col in self.data.columns:
                time_col = col
                break

        if time_col is None:
            # Use index as time
            time_data = self.data.index
            xlabel = 'Index'
        else:
            time_data = self.data[time_col]
            xlabel = time_col

        target_data = self.data[self.target_col].dropna()

        # Main time series plot
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # Full time series
        ax = axes[0]
        if time_col is None:
            ax.plot(target_data.index, target_data.values, linewidth=0.5, alpha=0.7)
        else:
            ax.plot(time_data[target_data.index], target_data.values, linewidth=0.5, alpha=0.7)
        ax.set_title(f'{self.target_col} Over Time (Full Series)', fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.target_col)
        ax.grid(True, alpha=0.3)

        # Rolling statistics
        ax = axes[1]
        window = min(100, len(target_data) // 10)
        rolling_mean = target_data.rolling(window=window).mean()
        rolling_std = target_data.rolling(window=window).std()

        if time_col is None:
            ax.plot(target_data.index, target_data.values, linewidth=0.5, alpha=0.3, label='Original')
            ax.plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, label=f'Rolling Mean (window={window})')
            ax.fill_between(rolling_mean.index,
                           (rolling_mean - rolling_std).values,
                           (rolling_mean + rolling_std).values,
                           alpha=0.2, color='red', label='±1 Std Dev')
        else:
            ax.plot(time_data[target_data.index], target_data.values, linewidth=0.5, alpha=0.3, label='Original')
            ax.plot(time_data[rolling_mean.index], rolling_mean.values, 'r-', linewidth=2, label=f'Rolling Mean (window={window})')

        ax.set_title(f'{self.target_col} with Rolling Statistics', fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.target_col)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Distribution
        ax = axes[2]
        ax.hist(target_data.values, bins=50, edgecolor='black', alpha=0.7, density=True)

        # Add KDE
        try:
            kde_x = np.linspace(target_data.min(), target_data.max(), 200)
            kde = stats.gaussian_kde(target_data)
            ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
        except:
            pass

        ax.set_title(f'{self.target_col} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(self.target_col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'target_time_series.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Time series plot saved")

    def create_correlation_heatmap(self):
        """Create correlation heatmap for numeric features."""
        print(f"\n{'-'*70}")
        print("CORRELATION ANALYSIS")
        print(f"{'-'*70}")

        # Select subset of columns if too many
        max_cols = 20
        if len(self.numeric_data.columns) > max_cols:
            cols_to_use = self.numeric_data.columns[:max_cols]
            print(f"Using first {max_cols} columns for correlation analysis")
        else:
            cols_to_use = self.numeric_data.columns

        corr_matrix = self.numeric_data[cols_to_use].corr()

        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Find highly correlated features
        high_corr_threshold = 0.8
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i],
                                          corr_matrix.columns[j],
                                          corr_matrix.iloc[i, j]))

        if high_corr_pairs:
            print(f"\nHighly correlated pairs (|r| > {high_corr_threshold}):")
            for col1, col2, corr_val in high_corr_pairs[:10]:
                print(f"  {col1} <-> {col2}: {corr_val:.3f}")

        print("Correlation analysis completed")

    def create_additional_visualizations(self):
        """Create additional visualizations for better understanding."""
        print(f"\n{'-'*70}")
        print("ADDITIONAL VISUALIZATIONS")
        print(f"{'-'*70}")

        # Statistical summary
        summary_stats = self.numeric_data.describe()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Missing values
        ax = axes[0, 0]
        missing_pct = (self.numeric_data.isnull().sum() / len(self.numeric_data)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)[:20]

        if len(missing_pct) > 0:
            missing_pct.plot(kind='barh', ax=ax, color='coral')
            ax.set_title('Missing Values (%)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Percentage Missing')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=14)
            ax.axis('off')

        # Plot 2: Feature ranges (normalized)
        ax = axes[0, 1]
        data_normalized = (self.numeric_data - self.numeric_data.min()) / (self.numeric_data.max() - self.numeric_data.min())
        data_normalized.iloc[:, :10].boxplot(ax=ax, rot=90)
        ax.set_title('Feature Ranges (Normalized)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Value [0, 1]')
        ax.grid(True, alpha=0.3)

        # Plot 3: Data density over time
        ax = axes[1, 0]
        if 'elapsed_seconds' in self.data.columns:
            time_bins = pd.cut(self.data['elapsed_seconds'], bins=50)
            counts = time_bins.value_counts().sort_index()
            ax.bar(range(len(counts)), counts.values, alpha=0.7, color='steelblue')
            ax.set_title('Data Density Over Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Bin')
            ax.set_ylabel('Number of Records')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No time data available', ha='center', va='center', fontsize=14)
            ax.axis('off')

        # Plot 4: Skewness
        ax = axes[1, 1]
        skewness = self.numeric_data.skew().sort_values(ascending=False)[:20]
        skewness.plot(kind='barh', ax=ax, color='lightgreen')
        ax.set_title('Feature Skewness (Top 20)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Skewness')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'additional_visualizations.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print("Additional visualizations completed")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print(f"\n{'='*70}")
        print(f"RUNNING COMPLETE ANALYSIS FOR: {self.dataset_name}")
        print(f"{'='*70}")

        # Outlier detection
        self.detect_outliers_iqr(threshold=1.5)
        self.detect_outliers_zscore(threshold=3)

        # Anomaly detection
        self.detect_anomalies_isolation_forest(contamination=0.1)
        self.detect_anomalies_lof(n_neighbors=20)

        # Distribution analysis
        self.analyze_distributions()

        # Time series analysis
        self.plot_target_over_time()

        # Correlation analysis
        self.create_correlation_heatmap()

        # Additional visualizations
        self.create_additional_visualizations()

        print(f"\n{'='*70}")
        print(f"ANALYSIS COMPLETE: {self.dataset_name}")
        print(f"All visualizations saved to: {self.output_dir}")
        print(f"{'='*70}\n")


def main():
    """Main function to run analysis on both datasets."""
    from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR
    from clean_data.clean_weather_data import SPEED_TRIALS_WEATHER_CLEAN

    print("\n" + "="*70)
    print("COMPREHENSIVE DATA ANALYSIS")
    print("="*70)

    # Analyze speed_trials dataset
    print("\n\n" + "#"*70)
    print("# DATASET 1: SPEED TRIALS (REGULAR)")
    print("#"*70)
    analyzer1 = DataAnalyzer(SPEED_TRIALS_REGULAR, 'speed_trials')
    analyzer1.run_complete_analysis()

    # Analyze speed_trials_weather dataset
    print("\n\n" + "#"*70)
    print("# DATASET 2: SPEED TRIALS WITH WEATHER")
    print("#"*70)
    analyzer2 = DataAnalyzer(SPEED_TRIALS_WEATHER_CLEAN, 'speed_trials_weather')
    analyzer2.run_complete_analysis()

    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
