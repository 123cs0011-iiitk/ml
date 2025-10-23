# Complete Model Training Guide

**Last Updated**: October 23, 2025 | **Status**: 2/7 trained - Random Forest next

---

## Current Status

✅ **Trained**: Linear Regression (R²=-0.002), Decision Tree (R²=0.001)  
🔄 **Next**: Random Forest (expected R²>0.90)  
⏳ **Pending**: SVM, KNN, ARIMA, Autoencoder

---

## Prerequisites

**Hardware**: 8GB RAM min (16GB rec), 4+ cores, 15GB disk  
**Software**: Python 3.8+, Node.js 16+, Git

```bash
cd backend
python -m venv venv
venv\Scripts\activate              # Windows
source venv/bin/activate           # Linux/Mac
pip install -r requirements.txt
```

**Data**: 1,001 stocks in `permanent/us_stocks/` and `permanent/ind_stocks/`

---

## Training Order (by speed & importance)

1. **Random Forest** (8-12m) - Best expected performer 🔄
2. **SVM** (15-25m) - Support vector regression
3. **KNN** (5-8m) - Instance-based learning
4. **Autoencoder** (18-30m) - Feature extraction
5. **ARIMA** (25-40m) - Time series (slowest)

---

## Quick Reference

### Commands

```bash
# Train models
python backend/training/basic_models/random_forest/trainer.py
python backend/training/basic_models/svm/trainer.py
python backend/training/advanced_models/knn/trainer.py
python backend/training/advanced_models/arima/trainer.py
python backend/training/advanced_models/autoencoder/trainer.py

# Options
--max-stocks 100        # Test with limited data
--force-retrain         # Ignore existing model

# Status & servers
python status.py
cd backend && python main.py          # Backend (port 5000)
cd frontend && npm run dev            # Frontend (port 5173)
```

### Model Files

**Models**: `backend/models/{model_name}/{model_name}_model.pkl`  
**Logs**: `backend/logs/{model}_training_{timestamp}.log`  
**Config**: `backend/training/{basic|advanced}_models/{model}/config.py`

---

## Model Training Details

### Random Forest 🔄 NEXT

**Time**: 8-12m | **Size**: 34MB | **Memory**: 6-8GB | **Target R²**: >0.90

**Review Files** (in `ignore.md`):
- `backend/training/basic_models/random_forest/trainer.py`
- `backend/algorithms/optimised/random_forest/random_forest.py`
- `backend/training/basic_models/random_forest/config.py`

**Config**: N_ESTIMATORS=100, MAX_DEPTH=15, MIN_SAMPLES_SPLIT=10, MIN_SAMPLES_LEAF=5

```bash
python backend/training/basic_models/random_forest/trainer.py
```

**Success**: R²>0.90, OOB score>0.85, feature importance calculated

---

### SVM ⏳

**Time**: 15-25m | **Size**: 10MB | **Memory**: 4-6GB | **Target R²**: >0.80

Uses LinearSVR, may subsample to 50K-100K samples for efficiency.

```bash
python backend/training/basic_models/svm/trainer.py
```

---

### KNN ⏳

**Time**: 5-8m | **Size**: 10MB | **Memory**: 4-5GB | **Target R²**: >0.75

Instance-based, fast training, may subsample to 50K samples.

```bash
python backend/training/advanced_models/knn/trainer.py
```

---

### Autoencoder ⏳

**Time**: 18-30m | **Size**: 1MB (3 files) | **Memory**: 4-6GB | **Target R²**: >0.70

Two-stage: (1) Autoencoder for features, (2) Regressor on encoded features. Uses TensorFlow/Keras.

```bash
python backend/training/advanced_models/autoencoder/trainer.py
```

**Files**: `autoencoder.h5`, `encoder.h5`, `metadata.pkl`

---

### ARIMA ⏳

**Time**: 25-40m | **Size**: 170MB | **Memory**: 3-5GB | **Target R²**: >0.65

Time series with hyperparameter search (p,d,q). Slowest model.

```bash
python backend/training/advanced_models/arima/trainer.py
```

---

### Linear Regression ✅ TRAINED

**Status**: R²=-0.002 (essentially zero - needs investigation)  
**Issue**: Model learns nothing despite training successfully

---

### Decision Tree ✅ TRAINED

**Status**: R²=0.001 (essentially zero - scaling removed Oct 23)  
**Fix**: Removed unnecessary StandardScaler, but R² still near zero

---

## Testing

### 1. Check Status
```bash
python status.py
```

### 2. Start Servers
```bash
cd backend && python main.py         # Terminal 1
cd frontend && npm run dev           # Terminal 2
```

### 3. Test Predictions

**Browser**: http://localhost:5173  
**Test Stocks**: AAPL, MSFT, TSLA (US) | RELIANCE, TCS, INFY (Indian)  
**Horizons**: 1D, 1W, 1M, 1Y, 5Y

**Verify**: Reasonable predictions (±20% for 1D/1W), confidence shown, no errors

---

## Monitoring

### Progress
```bash
# Windows PowerShell
Get-Content backend\logs\random_forest_training_*.log -Tail 50 -Wait

# Linux/Mac
tail -f backend/logs/random_forest_training_*.log
```

### Memory
Watch Task Manager (Windows), htop (Linux), Activity Monitor (Mac). Keep usage <80% of total RAM.

---

## Expected Performance

| Model | Min R² | Target R² | Status |
|-------|--------|-----------|--------|
| Random Forest | 0.85 | 0.90+ | 🔄 Next |
| SVM | 0.75 | 0.80 | ⏳ |
| KNN | 0.70 | 0.80 | ⏳ |
| Autoencoder | 0.65 | 0.75 | ⏳ |
| ARIMA | 0.60 | 0.70 | ⏳ |
| Linear Reg | 0.75 | 0.85 | ⚠️ -0.002 |
| Decision Tree | 0.75 | 0.85 | ⚠️ 0.001 |

---

## Troubleshooting

**Out of Memory**: Close apps, ensure 8GB+ free, train one model at a time, try `--max-stocks 100`  
**Poor R²**: Check features, verify data, review hyperparameters, try `--force-retrain`  
**Training Hangs**: Check logs, verify data exists, check RAM, restart if needed  
**No Predictions**: Verify backend running (`:5000/health`), check model trained (`status.py`), check browser console

---

## Best Practices

1. **Review First**: Read `trainer.py`, model class, `config.py` before training (catches bugs early)
2. **Train in Order**: Random Forest → SVM → KNN → Autoencoder → ARIMA
3. **Test Each**: Run `status.py` and test frontend after each model
4. **Monitor Resources**: Keep Task Manager open, watch memory <80%
5. **Save Logs**: Located in `backend/logs/` for debugging

---

## Configuration

Edit hyperparameters in `backend/training/{basic|advanced}_models/{model}/config.py`

**Example**: `random_forest/config.py`
```python
N_ESTIMATORS = 100      # Trees
MAX_DEPTH = 15          # Depth
MIN_SAMPLES_SPLIT = 10  # Split threshold
```

After editing, run trainer with `--force-retrain`.

**Features**: 38 total, defined in `backend/algorithms/stock_indicators.py`. Volume NOT used.

---

## Next Steps

**Immediate**: Review Random Forest code → Train → Validate  
**Short Term**: Train SVM, KNN, Autoencoder, ARIMA  
**Medium Term**: Fix Linear Regression & Decision Tree low R² scores  
**Long Term**: Optimize hyperparameters, automated retraining, add models

---

**Remember**: Quality over speed. Review-first workflow saves time.
