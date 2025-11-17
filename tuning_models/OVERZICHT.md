# Model Tuning & Analyse - Overzicht

## 📋 Wat is er toegevoegd?

Er is een complete analyse suite gemaakt in de `tuning_models/` directory met uitgebreide tests en visualisaties voor alle modellen.

## 📁 Bestandsstructuur

```
tuning_models/
├── README.md                    # Uitgebreide documentatie
├── OVERZICHT.md                 # Dit bestand
├── __init__.py                  # Module initialisatie
│
├── mlp_analysis.py              # MLP analyse & tuning
├── randomforest_analysis.py     # Random Forest analyse
├── lstm_analysis.py             # LSTM analyse & visualisatie
├── dnn_analysis.py              # DNN analyse & tuning
│
├── quick_analysis.py            # Snelle analyse (15-20 min)
├── run_all_analyses.py          # Volledige analyse suite (1-2 uur)
│
└── visualisations/              # Alle gegenereerde visualisaties komen hier
```

## 🚀 Hoe te gebruiken

### Optie 1: Snelle Analyse (AANBEVOLEN OM TE STARTEN)

Voor een snel overzicht van model performance:

```bash
python tuning_models/quick_analysis.py
```

**Wat doet dit?**
- Traint één configuratie per model type
- Genereert belangrijkste visualisaties
- Analyseert weather impact
- Duur: ~15-20 minuten

**Output:**
- Quick comparison van alle modellen
- Loss curves voor neural networks
- Feature importance (Random Forest)
- Weather impact analyse

### Optie 2: Volledige Analyse

Voor uitgebreide hyperparameter tuning:

```bash
python tuning_models/run_all_analyses.py
```

**Wat doet dit?**
- Uitgebreide tests van alle hyperparameters
- Alle visualisaties en vergelijkingen
- JSON files met gedetailleerde resultaten
- Duur: ~1-2 uur

### Optie 3: Individuele Analyses

Voor specifieke modellen:

```bash
# Random Forest
python tuning_models/randomforest_analysis.py

# MLP
python tuning_models/mlp_analysis.py

# LSTM
python tuning_models/lstm_analysis.py

# DNN
python tuning_models/dnn_analysis.py
```

## 📊 Wat wordt geanalyseerd?

### 1. MLP (Multi-Layer Perceptron)
- ✅ Architectuur vergelijking (2-5 layers)
- ✅ Dropout rates (0.0 - 0.5)
- ✅ Learning rates (0.0001 - 0.01)
- ✅ Batch sizes (16 - 256)
- ✅ Batch normalization impact

**Visualisaties:**
- Learning curves (loss & MAE)
- Model vergelijkingen
- Prediction scatter plots
- Residual analysis

### 2. Random Forest
- ✅ Aantal trees (10 - 500)
- ✅ Tree depth (5 - unlimited)
- ✅ Splitting criteria
- ✅ Feature sampling strategieën
- ✅ Overfitting analyse

**Visualisaties:**
- Feature importance rankings
- Train vs test performance
- Overfitting curves
- Prediction analysis

### 3. LSTM (Long Short-Term Memory)

**Note:** Dropout=0.2, LR=0.001, Batch sizes 16/32/64 zijn al geoptimaliseerd (zoals aangegeven)

Onderzoekt daarom:
- ✅ Architecturen (1-5 layers)
- ✅ Units per layer (32 - 128)
- ✅ Sequence lengths (10 - 100 timesteps)
- ✅ Convergentie snelheid
- ✅ Time series performance

**Visualisaties:**
- Loss curves met log scale
- MAE progressie
- Time series predictions
- Residual distributions
- Convergence analysis

### 4. DNN (Deep Neural Network)
- ✅ Network depths (3 - 10 layers)
- ✅ Network widths (64 - 1024 units)
- ✅ Activation functions (ReLU, tanh, sigmoid, ELU)
- ✅ Regularization (dropout, L2, batch norm)
- ✅ Optimizers (Adam, SGD, RMSprop)

**Visualisaties:**
- Deep learning curves
- Activation comparisons
- Regularization effectiveness
- Q-Q plots voor normality checks

## 🎯 Belangrijkste Features

### Visualisaties
Alle plots worden opgeslagen als:
- ✅ Hoge kwaliteit PNG (300 DPI)
- ✅ Beschrijvende bestandsnamen
- ✅ Duidelijke labels en legends
- ✅ Professionele styling

### Metrics
Elk model wordt geëvalueerd op:
- **R²** - Verklaarde variantie (hoger = beter, target > 0.9)
- **RMSE** - Root Mean Squared Error (lager = beter)
- **MAE** - Mean Absolute Error (lager = beter)
- **Training tijd** - Epochs tot convergentie

### Diagnostics
- Loss curves (train & validation)
- Overfitting detection
- Residual analysis
- Prediction quality
- Feature importance (waar relevant)

## 📈 Te verwachten output

### Voor elke analyse:
1. **PNG visualisaties** in `tuning_models/visualisations/`
2. **JSON resultaten** met alle metrics en configuraties
3. **Console output** met progressie en samenvatting

### Voorbeeld visualisaties:
- `mlp_learning_curves_speed_trials.png`
- `rf_feature_importance_weather.png`
- `lstm_convergence_speed_trials.png`
- `dnn_comparison_weather.png`
- En vele meer...

## 🔍 Analysepunten (zoals besproken)

Het systeem onderzoekt automatisch:

1. **Loss function gedrag**
   - Convergentie snelheid
   - Overfitting detectie
   - Optimale stopping point

2. **Model complexiteit**
   - Impact van depth/width
   - Regularization effectiviteit
   - Architecture choices

3. **Hyperparameter impact**
   - Learning rate gevoeligheid
   - Batch size effecten
   - Dropout optimalisatie

4. **Data impact**
   - Weather features contributie
   - Feature importance rankings
   - Sequence length impact (LSTM)

5. **Performance verschillen**
   - Waarom presteren modellen zoals ze doen?
   - Vergelijkingen tussen architecturen
   - Trade-offs tussen complexiteit en performance

## 📝 Tips voor gebruik

### Start met quick analysis
```bash
python tuning_models/quick_analysis.py
```
Dit geeft je in 15-20 minuten een goed overzicht.

### Bekijk visualisaties tussen runs
Na elke analyse, bekijk de PNG files om te zien wat er gebeurt.

### JSON files voor vergelijking
De JSON resultaten zijn handig voor cross-model vergelijking en rapportage.

### Pas aan indien nodig
Alle Python scripts zijn goed gedocumenteerd en makkelijk aan te passen.

## ⚙️ Dependencies

Zorg dat je hebt:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn scipy
```

(Deze staan waarschijnlijk al in je requirements.txt)

## 🎓 Voor je thesis

Deze analyses helpen met:
- ✅ Begrijpen waarom modellen presteren zoals ze doen
- ✅ Visualisaties voor in je thesis
- ✅ Gedetailleerde vergelijkingen
- ✅ Reproduceerbare resultaten
- ✅ Wetenschappelijke onderbouwing van model keuzes

## ❓ Veelgestelde vragen

**Q: Hoe lang duurt alles?**
A: Quick analysis ~15-20 min, volledige suite ~1-2 uur

**Q: Kan ik GPU gebruiken?**
A: Ja, TensorFlow gebruikt automatisch GPU indien beschikbaar

**Q: Wat als ik memory errors krijg?**
A: Verlaag batch sizes of train op kleinere data subset

**Q: Kan ik de analyses aanpassen?**
A: Ja! Alle scripts zijn goed gedocumenteerd en modulair

**Q: Welke analyses zijn het belangrijkst?**
A: Start met quick_analysis.py voor overzicht, dan specifieke modellen die je interessant vindt

## 📞 Volgende stappen

1. **Test het systeem:**
   ```bash
   python tuning_models/quick_analysis.py
   ```

2. **Bekijk de visualisaties** in `tuning_models/visualisations/`

3. **Voor uitgebreide analyse:**
   ```bash
   python tuning_models/run_all_analyses.py
   ```

4. **Gebruik resultaten** voor je thesis rapportage

## 🎉 Klaar!

Je hebt nu een compleet analyse systeem voor:
- ✅ MLP
- ✅ Random Forest
- ✅ LSTM
- ✅ DNN

Met:
- ✅ Uitgebreide visualisaties
- ✅ Loss function analyse
- ✅ Hyperparameter tuning
- ✅ Performance diagnostics
- ✅ Automatische rapportage

**Veel succes met je analyses!**
