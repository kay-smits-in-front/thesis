# Speed Trials Data Analysis

Comprehensive data analysis package for speed trials datasets.

## Features

- **Outlier Detection**: IQR method and Z-score method with visualizations
- **Anomaly Detection**: Isolation Forest and Local Outlier Factor (LOF)
- **Distribution Analysis**: Histograms, KDE, Q-Q plots, and violin plots
- **Time Series Analysis**: Target feature over time with rolling statistics
- **Correlation Analysis**: Heatmaps and highly correlated feature identification
- **Additional Visualizations**: Missing values, feature ranges, skewness, data density

## Usage

### Run Complete Analysis

From the project root directory:

```bash
python -m analysis.data_analysis
```

This will analyze both `speed_trials` and `speed_trials_weather` datasets and save all visualizations to the `output/` folder.

### Use as a Package

```python
from analysis import DataAnalyzer
from clean_data.clean_speed_trials import SPEED_TRIALS_REGULAR

# Create analyzer instance
analyzer = DataAnalyzer(
    data=SPEED_TRIALS_REGULAR,
    dataset_name='speed_trials',
    target_col='OPC_12_CPP_ENGINE_POWER'
)

# Run complete analysis
analyzer.run_complete_analysis()

# Or run individual analyses
analyzer.detect_outliers_iqr(threshold=1.5)
analyzer.detect_outliers_zscore(threshold=3)
analyzer.detect_anomalies_isolation_forest(contamination=0.1)
analyzer.detect_anomalies_lof(n_neighbors=20)
analyzer.analyze_distributions()
analyzer.plot_target_over_time()
analyzer.create_correlation_heatmap()
analyzer.create_additional_visualizations()
```

## Output

All visualizations are saved to `output/<dataset_name>/` with the following files:

- `outliers_iqr_boxplots.png` - Box plots showing IQR-based outliers
- `outliers_zscore_histograms.png` - Z-score distributions
- `anomalies_isolation_forest.png` - Isolation Forest anomaly detection (2D PCA)
- `anomalies_lof.png` - Local Outlier Factor anomaly detection (2D PCA)
- `distributions_histograms_kde.png` - Feature distributions with KDE
- `distributions_qq_plots.png` - Q-Q plots for normality assessment
- `distributions_violin_plots.png` - Violin plots showing distribution shapes
- `target_time_series.png` - Time series of target feature with rolling stats
- `correlation_heatmap.png` - Feature correlation matrix
- `additional_visualizations.png` - Missing values, ranges, skewness, density

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
```

## Parameters

### DataAnalyzer

- `data` (DataFrame): The dataset to analyze
- `dataset_name` (str): Name for organizing output files
- `target_col` (str): Name of the target/output column (default: 'OPC_12_CPP_ENGINE_POWER')

### Analysis Methods

- `detect_outliers_iqr(threshold=1.5)`: IQR multiplier for outlier bounds
- `detect_outliers_zscore(threshold=3)`: Z-score threshold for outliers
- `detect_anomalies_isolation_forest(contamination=0.1)`: Expected proportion of anomalies
- `detect_anomalies_lof(n_neighbors=20)`: Number of neighbors for LOF algorithm
