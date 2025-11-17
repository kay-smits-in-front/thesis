# Model Tuning & Analysis

Deze directory bevat uitgebreide analyses en hyperparameter tuning voor alle modellen in dit onderzoek.

## Overzicht

Elk analysebestand bevat:
- ✅ Uitgebreide hyperparameter tuning
- ✅ Loss function visualisaties
- ✅ Performance diagnostics
- ✅ Vergelijkingen tussen configuraties
- ✅ Automatische opslag van resultaten en visualisaties

## Bestanden

### 1. `mlp_analysis.py` - Multi-Layer Perceptron Analyse

**Onderzoekt:**
- Verschillende architecturen (shallow → deep)
- Dropout rates (0.0 - 0.5)
- Learning rates (0.0001 - 0.01)
- Batch sizes (16 - 256)
- Batch normalization impact

**Visualisaties:**
- Learning curves (loss & MAE)
- Prediction scatter plots
- Residual analysis
- Model vergelijkingen

**Uitvoeren:**
```bash
python tuning_models/mlp_analysis.py
```

**Output:**
- `visualisations/mlp_learning_curves_*.png`
- `visualisations/mlp_comparison_*.png`
- `visualisations/mlp_predictions_*.png`
- `visualisations/mlp_results_*.json`

---

### 2. `randomforest_analysis.py` - Random Forest Analyse

**Onderzoekt:**
- Aantal trees (n_estimators: 10 - 500)
- Tree depth (max_depth: 5 - None)
- Splitting criteria (min_samples_split: 2 - 50)
- Feature sampling (max_features: sqrt, log2, 0.5, 1.0)
- Overfitting analyse

**Visualisaties:**
- Feature importance rankings
- Train vs test performance (overfitting check)
- Prediction accuracy plots
- Hyperparameter impact

**Uitvoeren:**
```bash
python tuning_models/randomforest_analysis.py
```

**Output:**
- `visualisations/rf_feature_importance_*.png`
- `visualisations/rf_overfitting_analysis_*.png`
- `visualisations/rf_comparison_*.png`
- `visualisations/rf_predictions_*.png`
- `visualisations/rf_results_*.json`

---

### 3. `lstm_analysis.py` - LSTM Analyse

**Geoptimaliseerde parameters (al getest):**
- ✅ Dropout: 0.2
- ✅ Learning rate: 0.001
- ✅ Batch sizes: 16, 32, 64

**Onderzoekt:**
- Architecturen (1-layer → 5-layer)
- LSTM units per layer (32 - 128)
- Sequence lengths/timesteps (10 - 100)
- Convergentie snelheid
- Time series performance

**Visualisaties:**
- Loss curves met log scale
- MAE progressie
- Time series predictions
- Residual distributions
- Convergence speed analysis

**Uitvoeren:**
```bash
python tuning_models/lstm_analysis.py
```

**Output:**
- `visualisations/lstm_learning_curves_*.png`
- `visualisations/lstm_convergence_*.png`
- `visualisations/lstm_comparison_*.png`
- `visualisations/lstm_predictions_*.png`
- `visualisations/lstm_results_*.json`

---

### 4. `dnn_analysis.py` - Deep Neural Network Analyse

**Onderzoekt:**
- Network depths (3 - 10 layers)
- Network widths (64 - 1024 units)
- Activation functions (ReLU, tanh, sigmoid, ELU)
- Regularization strategieën:
  - Dropout (0.0 - 0.5)
  - L2 regularization
  - Batch normalization
  - Combinaties
- Optimizers (Adam, SGD, RMSprop)

**Visualisaties:**
- Deep learning curves
- Activation function impact
- Regularization effectiveness
- Q-Q plots (normality checks)
- Comprehensive comparisons

**Uitvoeren:**
```bash
python tuning_models/dnn_analysis.py
```

**Output:**
- `visualisations/dnn_learning_curves_*.png`
- `visualisations/dnn_comparison_*.png`
- `visualisations/dnn_predictions_*.png`
- `visualisations/dnn_results_*.json`

---

## Datasets

Alle analyses worden uitgevoerd op:
1. **Speed Trials data** - Basis scheepvaart data
2. **Speed Trials + Weather data** - Uitgebreid met weergegevens

Dit maakt het mogelijk om de impact van weather features te evalueren.

---

## Resultaten Structuur

### Visualisaties
Alle plots worden opgeslagen in `tuning_models/visualisations/`:
- PNG formaat (300 DPI) voor publicatie kwaliteit
- Beschrijvende bestandsnamen
- Duidelijke labels en legends

### JSON Resultaten
Elke analyse slaat gestructureerde resultaten op met:
- Model configuratie
- Performance metrics (R², RMSE, MAE, MSE)
- Training details (epochs, loss values)
- Timestamp

---

## Gebruik

### Alles draaien
```bash
# Run alle analyses (kan lang duren!)
python tuning_models/mlp_analysis.py
python tuning_models/randomforest_analysis.py
python tuning_models/lstm_analysis.py
python tuning_models/dnn_analysis.py
```

### Individuele analyses
Elk script kan onafhankelijk gedraaid worden en produceert zijn eigen visualisaties.

---

## Interpretatie Richtlijnen

### Loss Curves
- **Convergerend**: Model leert goed
- **Diverging train/val**: Overfitting
- **Plateau**: Mogelijk underfitting of local minimum
- **Noisy**: Mogelijk te hoge learning rate

### Residual Plots
- **Random scatter rond 0**: Goed model
- **Patronen zichtbaar**: Model mist informatie
- **Trechter vorm**: Heteroskedasticity (variantie verandert)

### Feature Importance (Random Forest)
- Identificeert belangrijkste predictors
- Helpt feature selectie
- Inzicht in domein kennis

### Overfitting Indicators
- Grote gap tussen train en test R²
- Perfecte train score, slechte test score
- Loss curves die divergeren

---

## Performance Metrics

### R² (R-squared)
- Bereik: 0 tot 1 (1 = perfect)
- Meet proportie verklaarde variantie
- **Target**: > 0.9 voor goede fit

### RMSE (Root Mean Squared Error)
- In dezelfde units als target (kW)
- Straft grote fouten zwaarder
- **Target**: < 5% van gemiddelde power

### MAE (Mean Absolute Error)
- Gemiddelde absolute fout
- Makkelijker te interpreteren dan RMSE
- **Target**: < 3% van gemiddelde power

---

## Vereisten

Installeer dependencies:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn scipy
```

---

## Tijdsinschatting

**Minder tijdrovend (eerst uitvoeren):**
- MLP analysis: ~10-20 minuten
- Random Forest analysis: ~5-15 minuten

**Meer tijdrovend:**
- LSTM analysis: ~30-60 minuten (sequenties + meerdere timesteps)
- DNN analysis: ~20-40 minuten (veel configuraties)

**Totaal**: ~1-2 uur voor complete analyse suite

---

## Aanbevelingen

1. **Start met Random Forest** - Snelst, geeft feature importance insights
2. **Daarna MLP** - Baseline neural network performance
3. **Dan DNN** - Deep learning zonder sequences
4. **Laatste LSTM** - Meest complex, time-consuming

5. **Analyseer tussentijds** - Bekijk visualisaties na elk model
6. **Vergelijk resultaten** - Gebruik JSON files voor cross-model vergelijking

---

## Troubleshooting

### Out of Memory
- Verlaag batch sizes
- Reduceer model complexity
- Train op kleinere dataset subset

### Slow Training
- Gebruik GPU indien beschikbaar
- Verlaag epochs
- Reduceer aantal configuraties

### Poor Performance
- Check data preprocessing
- Verify feature scaling
- Review hyperparameter ranges

---

## Contact & Vragen

Voor vragen over de analyses, zie de individuele Python bestanden voor gedetailleerde documentatie en comments.

---

**Laatste update**: 2025-11-17
**Versie**: 1.0
