# MODEL_TRAINING.md

# Models

- ANN
- ARIMA
- Autoencoder
- CNN
- Decision Tree
- KNN
- Linear Regression
- Random Forest
- SVM

---
# Model Storage Structure

```
backend/models/
├── linear_regression/
│   └── linear_regression_model.pkl
├── random_forest/
│   └── random_forest_model.pkl
├── decision_tree/
│   └── decision_tree_model.pkl
├── svm/
│   └── svm_model.pkl
├── knn/
│   └── knn_model.pkl
├── ann/
│   ├── ann_model.pkl        # scikit-learn metadata/wrapper
│   └── ann_model_model.h5   # Keras deep learning weights
│   # (ANN may have additional logs/config if needed)
├── cnn/
│   ├── cnn_model.pkl        # joblib wrapper/metadata
│   ├── cnn_model_main.h5    # Keras model (main)
│   ├── cnn_model_encoder.h5 # (optional) encoder, if present
│   └── cnn_model_extra.h5   # (optional) other .h5 files (aux models)
├── arima/
│   └── arima_model.pkl
├── autoencoder/
│   ├── autoencoder_model.pkl_autoencoder.h5   # full autoencoder (Keras/Tensorflow)
│   ├── autoencoder_model.pkl_encoder.h5       # encoder part
│   └── autoencoder_model.pkl_metadata.pkl     # pickle: scaler, regressor, config
│   # All 3 files are needed to reconstruct the autoencoder
└── model_status.json        # Global status, tracks all model updates

```

**Summary:**  
- Each model saves in its own folder under `backend/models/`.
- `.pkl`: classical ML metadata; `.h5`: deep learning weights.
- ANN, CNN, Autoencoder use both `.pkl` (metadata) and `.h5` (weights).
- Autoencoder uses 3 files: `autoencoder.h5`, `encoder.h5`, `metadata.pkl`.
- `model_status.json` tracks model/training status.


---
# Training Data Math

**Time Period**
- Start: 01-01-2020
- End: 31-12-2024
- Calendar Days: 1,825 days
- Trading Days: ~1,260 days (252/year × 5)
- Actual Rows: ~1,200/stock (holidays/weekends excluded)

**Stock Coverage**
- US Stocks: 501
- Indian Stocks: 500
- **Total: 1,001 stocks**

**Per Stock**
- Rows: ~1,200
- Features: 37
- Data Points: 1,200 × 37 = **44,400 numbers**

**Full Dataset**
- Total Rows: 1,001 × 1,200 = **1,201,200 rows**
- Total Features: 1,201,200 × 37 = **44,444,400 data points**
- Target Values: 1,201,200 close prices

**Memory Calculation**
- Float64: 8 bytes/number
- X (features): 44.4M × 8 = **355 MB**
- y (targets): 1.2M × 8 = **10 MB**
- With overhead (2x): **~730 MB minimum**
- Per batch (100 stocks): **~73 MB**

**Training Split**
- Train: 80% (~960,960 rows)
- Validation: 20% (~240,240 rows)

---
# Training Resources

| Model | Time | Size | Memory |
|-------|------|------|--------|
| Linear Regression | 5-10 min | 3 MB | 2-3 GB |
| Decision Tree | 5-10 min | 150 MB | 3-4 GB |
| Random Forest | 10-15 min | 9.5 GB | 6-8 GB |
| KNN | 15-25 min | 10 MB | 4-5 GB |
| SVM | 20-30 min | 10 MB | 4-6 GB |
| ANN | 30-45 min | 256 KB | 4-6 GB |
| Autoencoder | 40-60 min | 1 MB | 4-6 GB |
| CNN | 45-75 min | 5 MB | 5-7 GB |
| ARIMA | 90-180 min | 170 MB | 3-5 GB |

**Minimum**: 8 GB RAM, 12 GB disk, 4 cores  
**Recommended**: 16 GB RAM, 20 GB disk, 8 cores, SSD

**Training Order**: Linear → Decision Tree → Random Forest → KNN → SVM → ANN → Autoencoder → CNN → ARIMA  
**Total Time**: 4-6 hours sequential

---
# Features (37 Total)

**Price (2)**: `price_change`, `price_change_abs`

**Moving Averages (10)**: `ma_5`, `ma_5_ratio`, `ma_10`, `ma_10_ratio`, `ma_20`, `ma_20_ratio`, `ma_50`, `ma_50_ratio`, `ma_200`, `ma_200_ratio`

**Volatility (1)**: `volatility`

**Momentum (1)**: `rsi` (14-period)

**Intraday Ratios (2)**: `hl_ratio`, `oc_ratio`

**Position (1)**: `price_position`

**Lagged (5)**: `close_lag_1`, `close_lag_2`, `close_lag_3`, `close_lag_5`, `close_lag_10`

**Rolling Stats (9)**: `close_std_5/10/20`, `close_min_5/10/20`, `close_max_5/10/20`

**Time (3)**: `day_of_week`, `month`, `quarter`

**Raw OHLC (3)**: `open`, `high`, `low`

---
# Training Data

**Source**: `permanent/{us_stocks|ind_stocks}/individual_files/`  
**Period**: 5 years (2020-2025)  
**Stocks**: ~1,000 (501 US + 500 Indian)  
**Samples**: ~1.2M data points  
**Target**: Next day's closing price

**Volume**: Exists in files but **NOT USED** in calculations

---
# Batch Strategies

| Model | Strategy | Method |
|-------|----------|--------|
| Linear Regression | Incremental | `partial_fit()` per batch |
| Decision Tree | Accumulate | Load all, train once |
| Random Forest | Accumulate | Load all, train once |
| SVM | Subsample | Train on 50% sample |
| KNN | Subsample | Train on 50% sample |
| ANN | Keras Batch | Mini-batch (32 samples) |
| CNN | Keras Batch | Mini-batch (8 samples) |
| ARIMA | Subsample | Train on sample |
| Autoencoder | Keras Batch | Mini-batch (32 samples) |

**Default**: 100 stocks/batch, 50K rows/sub-batch

---
# Directory Structure

```
data/
├── past/                              # Training data
│   ├── ind_stocks/individual_files/   # 500 stocks
│   └── us_stocks/individual_files/    # 501 stocks
├── latest/                            # Real-time data
│   ├── ind_stocks/individual_files/
│   └── us_stocks/individual_files/
├── future/                            # ML predictions
│   ├── ind_stocks/individual_files/
│   └── us_stocks/individual_files/
├── index_ind_stocks_dynamic.csv
└── index_us_stocks_dynamic.csv
```

---
# Commands

**Train Model**
```bash
python backend/training/train_full_dataset.py --model MODEL_NAME
```

**Force Retrain**
```bash
python backend/training/train_full_dataset.py --model MODEL_NAME --force-retrain
```

**Check Status**
```bash
python status.py
```

**Custom Batch**
```bash
python backend/training/train_full_dataset.py \
    --model MODEL_NAME \
    --stock-batch-size 50 \
    --subsample-percent 30
```

---
# Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| ANN: R² < -1000 | Gradient explosion | `--force-retrain` (fixed) |
| SVM: R² < 0 | Too much data | `--force-retrain` (subsampling) |
| KNN: R² < 0 | k=5 too small | `--force-retrain` (k=15) |
| Autoencoder: R² < -100K | Sigmoid output | `--force-retrain` (linear) |
| CNN: Out of Memory | Batch too large | `--force-retrain` (batch=8) |
| Linear Reg: Stuck | SGD disabled | `--force-retrain` (enabled) |
| ARIMA: Too Slow | No timeout | Reduced search space |

**Out of Memory**: Reduce batch size in `backend/prediction/config.py`
```python
self.STOCK_BATCH_SIZE = 50
```

**Training Hangs**: Auto-timeout 1 hour/batch

---
# Expected Performance

| Model | Target R² |
|-------|-----------|
| Linear Regression | > 0.85 |
| Random Forest | > 0.90 |
| Decision Tree | > 0.80 |
| SVM | > 0.80 |
| KNN | > 0.75 |
| ANN | > 0.75 |
| CNN | > 0.75 |
| ARIMA | > 0.70 |
| Autoencoder | > 0.75 |

---
# Verification

```bash
# Check status
python status.py

# Start backend
cd backend && python main.py

# Start frontend
cd frontend && npm run dev

# Test all horizons: 1D/1W/1M/1Y/5Y
```

---
