# Model Training Documentation

**Last Updated**: October 23, 2025 | **Status**: 2/7 models trained

---

## Models (7 Total)

**Basic Models**: Linear Regression ✅ (R²=-0.002), Decision Tree ✅ (R²=0.001), Random Forest 🔄, SVM ⏳  
**Advanced Models**: KNN ⏳, ARIMA ⏳, Autoencoder ⏳

---

## Storage Structure

```
backend/models/
├── {model_name}/{model_name}_model.pkl    # Classical ML (joblib)
├── autoencoder/                           # Deep learning (3 files)
│   ├── autoencoder_model.pkl_autoencoder.h5
│   ├── autoencoder_model.pkl_encoder.h5
│   └── autoencoder_model.pkl_metadata.pkl
└── model_status.json
```

---

## Training Architecture

```
backend/training/
├── basic_models/{model}/           # Linear Regression, Decision Tree, Random Forest, SVM
│   ├── config.py                   # Hyperparameters
│   └── trainer.py                  # Standalone trainer
├── advanced_models/{model}/        # KNN, ARIMA, Autoencoder
│   ├── config.py
│   └── trainer.py
├── common_trainer_utils.py         # Shared utilities
├── display_manager.py              # Progress tracking
└── validation_stocks.json          # Validation data
```

---

## Training Data

**Period**: 2020-2024 (5 years) | **Stocks**: 1,001 (501 US + 500 Indian)  
**Samples**: ~1,200 rows/stock → ~1,057 after feature engineering  
**Features**: 38 | **Total**: ~1M training samples

**Memory**: X=365MB, y=10MB, Peak=2-8GB depending on model

---

## Features (38 Total)

**Price (2)**: price_change, price_change_abs  
**Moving Averages (10)**: ma_5/10/20/50/200 + ratios  
**Volatility (1)**: volatility  
**Momentum (1)**: rsi  
**Intraday (2)**: hl_ratio, oc_ratio  
**Position (1)**: price_position  
**Lagged (5)**: close_lag_1/2/3/5/10  
**Rolling Stats (9)**: close_std/min/max for 5/10/20 days  
**Time (3)**: day_of_week, month, quarter  
**Raw OHLC (3)**: open, high, low

**Note**: Volume exists but is NOT USED in calculations.

---

## Training Commands

```bash
# Basic models
python backend/training/basic_models/linear_regression/trainer.py
python backend/training/basic_models/decision_tree/trainer.py
python backend/training/basic_models/random_forest/trainer.py
python backend/training/basic_models/svm/trainer.py

# Advanced models
python backend/training/advanced_models/knn/trainer.py
python backend/training/advanced_models/arima/trainer.py
python backend/training/advanced_models/autoencoder/trainer.py

# Options: --max-stocks 100, --force-retrain
```

---

## Expected Performance

| Model | Time | Size | Memory | Target R² | Current R² | Status |
|-------|------|------|--------|-----------|------------|--------|
| Linear Regression | 2-3m | 3MB | 2-3GB | 0.85 | -0.002 | ✅ Needs fix |
| Decision Tree | 3-5m | 150MB | 3-4GB | 0.85 | 0.001 | ✅ Needs fix |
| Random Forest | 8-12m | 34MB | 6-8GB | 0.90+ | - | 🔄 Next |
| SVM | 15-25m | 10MB | 4-6GB | 0.80 | - | ⏳ Pending |
| KNN | 5-8m | 10MB | 4-5GB | 0.80 | - | ⏳ Pending |
| ARIMA | 25-40m | 170MB | 3-5GB | 0.70 | - | ⏳ Pending |
| Autoencoder | 18-30m | 1MB | 4-6GB | 0.75 | - | ⏳ Pending |

**Requirements**: 8GB RAM min, 16GB recommended | 15GB disk | 4+ cores

---

## Prediction Method

Models predict **percentage change** → System converts to price

```
Current: $100 | Model: +2.5% | Final: $100 × 1.025 = $102.50
```

---

## Recent Fixes (Oct 23, 2025)

- **Decision Tree**: Removed unnecessary StandardScaler (tree models don't need scaling)
- **Review Workflow**: Code review before training catches bugs early
- **Validation**: Using separate validation stocks for unbiased testing

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Low R² (~0.02) | Review feature engineering, check hyperparameters |
| Out of Memory | Close apps, ensure 8GB+ free, train one at a time |
| Training Hangs | Check logs in `backend/logs/`, verify data files exist |
| Model Not Found | Run trainer: `python backend/training/.../trainer.py` |

---

## Logs & Validation

**Logs**: `backend/logs/{model}_training_{timestamp}.log`  
**Validation**: 40 stocks (20 US + 20 Indian) defined in `validation_stocks.json`  
**Metrics**: R² Score, RMSE, MAE, Average % error

---

## See Also

[Complete Training Guide](TRAINING_GUIDE.md) | [Backend README](../backend/README.md) | [Training Module](../backend/training/README.md)
