# Complete Model Training Guide

This guide provides step-by-step instructions for training all 9 ML models, with expected times, testing procedures, and troubleshooting tips.

## Training Approach

**All models use single-pass training:**
- Load complete dataset (~1000 stocks, 5 years of historical data) into memory
- Train each model with a single `model.fit(X, y)` call
- Progress tracking during data loading phase

This approach ensures optimal model performance and simplifies the training process.

---

## Prerequisites

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: Multi-core processor (4+ cores recommended)
- **Disk Space**: 10GB free space
- **OS**: Windows 10+, Linux, or macOS
- **Python**: 3.8 or higher

### Software Requirements
```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt

# Verify installation
python -c "import flask, pandas, numpy, sklearn, tensorflow; print('All packages installed successfully')"
```

### Data Requirements
- Historical data in `permanent/` directory (5 years, 1001 stocks)
- Index files: `data/index_us_stocks_dynamic.csv`, `data/index_ind_stocks_dynamic.csv`
- 37 technical indicators (OHLC-based, no volume)

---

## Training All Models

### Recommended Training Order

Train models in this order (fastest to slowest):

1. **Linear Regression** (5-10 min) - Good baseline
2. **Decision Tree** (5-10 min) - Fast, interpretable
3. **Random Forest** (10-15 min) - Best performer
4. **KNN** (15-25 min) - Instance-based learning
5. **SVM** (20-30 min) - Support vector regression
6. **Autoencoder** (40-60 min) - Feature extraction
7. **ANN** (30-45 min) - Neural network
8. **CNN** (45-75 min) - Convolutional network
9. **ARIMA** (90-180 min) - Time series (slowest)

### Training Commands

```bash
# Change to project root
cd C:\Users\ankit\Desktop\PROJECTS\CS301\ml

# Train models one by one
python backend/training/train_full_dataset.py --model linear_regression
python backend/training/train_full_dataset.py --model decision_tree --force-retrain
python backend/training/train_full_dataset.py --model random_forest --force-retrain
python backend/training/train_full_dataset.py --model knn --force-retrain
python backend/training/train_full_dataset.py --model svm --force-retrain
python backend/training/train_full_dataset.py --model autoencoder --force-retrain
python backend/training/train_full_dataset.py --model ann --force-retrain
python backend/training/train_full_dataset.py --model cnn
python backend/training/train_full_dataset.py --model arima

# Check status after each training
python status.py
```

---

## Model-by-Model Training Guide

### 1. Linear Regression

**Expected Time**: 5-10 minutes
**Model Size**: ~3 MB
**Memory Usage**: 2-3 GB peak

**Command**:
```bash
python backend/training/train_full_dataset.py --model linear_regression
```

**What to Expect**:
- Data loading: 1-2 minutes (1001 stocks)
- Training: 3-5 minutes
- Validation: 1-2 minutes
- Expected R²: > 0.85

**Success Indicators**:
- ✅ "Training completed successfully"
- ✅ R² > 0.85 in `python status.py`
- ✅ Model saved to `backend/models/linear_regression/linear_regression_model.pkl`

**If Training Fails**:
- Check logs in `backend/logs/training.log`
- Ensure SGD is enabled (default: True)
- Try with `--force-retrain` flag

---

### 2. Decision Tree

**Expected Time**: 5-10 minutes
**Model Size**: ~150 MB
**Memory Usage**: 3-4 GB peak

**Command**:
```bash
python backend/training/train_full_dataset.py --model decision_tree --force-retrain
```

**What to Expect**:
- Fast training on full dataset
- Expected R²: > 0.80
- Good interpretability

**Success Indicators**:
- ✅ R² ≈ 0.85 (from previous training)
- ✅ Fast training completion
- ✅ Predictions are stable

---

### 3. Random Forest

**Expected Time**: 10-15 minutes
**Model Size**: ~9.5 GB
**Memory Usage**: 6-8 GB peak

**Command**:
```bash
python backend/training/train_full_dataset.py --model random_forest --force-retrain
```

**What to Expect**:
- Longer training due to ensemble
- Expected R²: > 0.90 (best performer)
- Multiple trees being trained

**Success Indicators**:
- ✅ R² ≈ 0.994 (excellent)
- ✅ Consistent validation performance
- ✅ Primary model for ensemble

---

### 4. K-Nearest Neighbors (KNN)

**Expected Time**: 15-25 minutes
**Model Size**: ~10 MB
**Memory Usage**: 4-5 GB peak

**Command**:
```bash
python backend/training/train_full_dataset.py --model knn --force-retrain
```

**What to Expect**:
- Subsampling to 50K samples (from 1M+)
- Distance-weighted neighbors (k=15)
- Ball tree algorithm for efficiency
- Expected R²: > 0.75

**Success Indicators**:
- ✅ Subsampling message in logs
- ✅ Training completes in < 30 minutes
- ✅ R² > 0.75 (improved from -27.9)

**Previous Issue**: k=5 too small, no subsampling → Fixed

---

### 5. Support Vector Regression (SVM)

**Expected Time**: 20-30 minutes
**Model Size**: ~10 MB
**Memory Usage**: 4-6 GB peak

**Command**:
```bash
python backend/training/train_full_dataset.py --model svm --force-retrain
```

**What to Expect**:
- Subsampling to 10K samples
- LinearSVR for large datasets
- Better hyperparameters (C=100, epsilon=0.01)
- Expected R²: > 0.80

**Success Indicators**:
- ✅ "Using LinearSVR for large dataset" message
- ✅ Training completes in < 40 minutes
- ✅ R² > 0.80 (improved from -26.2)

**Previous Issue**: No subsampling, poor hyperparameters → Fixed

---

### 6. Autoencoder

**Expected Time**: 40-60 minutes
**Model Size**: ~1 MB (3 files: 2x .h5 + 1x .pkl)
**Memory Usage**: 4-6 GB peak

**Command**:
```bash
python backend/training/train_full_dataset.py --model autoencoder --force-retrain
```

**What to Expect**:
- Two-stage training (encoder + regressor)
- Linear output activation (not sigmoid)
- StandardScaler (not MinMaxScaler)
- Expected R²: > 0.75

**Success Indicators**:
- ✅ Both encoder and regressor train
- ✅ Trains on all stocks (not just 17)
- ✅ R² > 0.75 (improved from -136K)

**Previous Issue**: Wrong activation, wrong scaler → Fixed

---

### 7. Artificial Neural Network (ANN)

**Expected Time**: 30-45 minutes
**Model Size**: ~256 KB (2 files: .pkl + .h5)
**Memory Usage**: 4-6 GB peak

**Command**:
```bash
python backend/training/train_full_dataset.py --model ann --force-retrain
```

**What to Expect**:
- Architecture: [128, 64, 32]
- Dropout: 0.2 (reduced from 0.5)
- Gradient clipping active
- Early stopping with patience=30
- Expected R²: > 0.75

**Success Indicators**:
- ✅ Validation loss decreases steadily
- ✅ No gradient explosion
- ✅ R² > 0.75 (improved from -9.7M)
- ✅ Training converges smoothly

**Previous Issue**: Catastrophic gradient explosion → Fixed

---

### 8. Convolutional Neural Network (CNN)

**Expected Time**: 45-75 minutes
**Model Size**: ~5 MB
**Memory Usage**: 5-7 GB peak

**Command**:
```bash
python backend/training/train_full_dataset.py --model cnn
```

**What to Expect**:
- Sequence length: 20 (reduced from 30)
- Filters: [32, 16] (simplified)
- Single-pass training: loads all data, trains with one model.fit() call
- Memory monitoring active
- Expected R²: > 0.75

**Success Indicators**:
- ✅ No memory errors
- ✅ Training completes successfully
- ✅ Memory optimization messages in logs
- ✅ R² > 0.75

**Previous Issue**: Memory allocation failure → Fixed

---

### 9. ARIMA (Slowest)

**Expected Time**: 90-180 minutes (1.5-3 hours)
**Model Size**: ~170 MB
**Memory Usage**: 3-5 GB peak

**Command**:
```bash
python backend/training/train_full_dataset.py --model arima
```

**What to Expect**:
- Hyperparameter search (p, d, q)
- Reduced search space (max_p=3, max_q=3)
- 5-minute timeout per search
- Expected R²: > 0.70

**Success Indicators**:
- ✅ Parameter search completes
- ✅ Optimal (p,d,q) found
- ✅ Training completes in < 3 hours
- ✅ R² > 0.70

**Note**: ARIMA is the slowest model. Consider running overnight or limiting to specific stocks for testing.

---

## Monitoring Training Progress

### Real-Time Status

```bash
# Check current status
python status.py

# Watch status continuously (Linux/Mac)
watch -n 30 python status.py

# Check logs
tail -f backend/logs/training.log
```

### Expected Output During Training

```
[MODEL_NAME] Data Loading: 450/1001 stocks (44.9%) - Current: AAPL
[MODEL_NAME] Training linear_regression on 452,103 samples from 987 stocks...
[MODEL_NAME] Training in progress - single-pass training on full dataset...
[MODEL_NAME] linear_regression model training completed in 4.2 minutes
```

### Memory Monitoring

```bash
# Windows
taskmgr

# Linux
htop

# Mac
Activity Monitor
```

**Recommended**: Keep memory usage below 80% for stable training.

---

## Testing After Training

### 1. Check Model Status

```bash
python status.py
```

**Expected Output**:
```
╔══════════════╦════════╦═══════════════╦═════════╗
║ Model        ║ Status ║ Stocks        ║ R² Score║
╠══════════════╬════════╬═══════════════╬═════════╣
║ Random Forest║   ✅   ║ 913           ║  0.994  ║
║ Linear Reg   ║   ✅   ║ 913           ║  0.894  ║
║ ...          ║  ...   ║ ...           ║   ...   ║
╚══════════════╩════════╩═══════════════╩═════════╝
```

### 2. Start Backend Server

```bash
cd backend
python main.py
```

**Expected Output**:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### 3. Start Frontend

```bash
# In a new terminal
cd frontend
npm run dev
```

**Expected Output**:
```
  VITE v4.x.x  ready in 500 ms
  ➜  Local:   http://localhost:5173/
```

### 4. Test Predictions on Frontend

1. **Open Browser**: Navigate to `http://localhost:5173`

2. **Search for Stock**: 
   - US Stock: "AAPL" or "MSFT"
   - Indian Stock: "RELIANCE" or "TCS"

3. **Test All Horizons**:
   - Click **1D** button → Should show price prediction for tomorrow
   - Click **1W** button → Should show price prediction for next week
   - Click **1M** button → Should show price prediction for next month
   - Click **1Y** button → Should show price prediction for next year
   - Click **5Y** button → Should show price prediction for 5 years ahead

4. **Verify Predictions**:
   - ✅ Predicted price is reasonable (within ±50% of current price for 1D/1W)
   - ✅ Confidence intervals are shown
   - ✅ Model name is displayed (e.g., "Random Forest")
   - ✅ No error messages

### 5. Test Multiple Stocks

Test at least 3-5 stocks to ensure consistency:
- **US**: AAPL, MSFT, GOOGL, TSLA, AMZN
- **Indian**: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK

---

## Expected Performance Metrics

### Target R² Scores (Expected After Training)

| Model | Minimum R² | Target R² | Status |
|-------|-----------|-----------|--------|
| Linear Regression | 0.85 | 0.89 | ⏳ Not Trained |
| Random Forest | 0.90 | 0.99 | ⏳ Not Trained |
| Decision Tree | 0.80 | 0.85 | ⏳ Not Trained |
| SVM | 0.80 | 0.85 | ⏳ Not Trained |
| KNN | 0.75 | 0.80 | ⏳ Not Trained |
| ANN | 0.75 | 0.85 | ⏳ Not Trained |
| CNN | 0.75 | 0.80 | ⏳ Not Trained |
| ARIMA | 0.70 | 0.75 | ⏳ Not Trained |
| Autoencoder | 0.75 | 0.80 | ⏳ Not Trained |

### Training Time Benchmarks

| Model | Minimum | Average | Maximum |
|-------|---------|---------|---------|
| Linear Regression | 5 min | 7 min | 10 min |
| Decision Tree | 5 min | 7 min | 10 min |
| Random Forest | 10 min | 12 min | 15 min |
| KNN | 15 min | 20 min | 25 min |
| SVM | 20 min | 25 min | 30 min |
| ANN | 30 min | 37 min | 45 min |
| CNN | 45 min | 60 min | 75 min |
| Autoencoder | 40 min | 50 min | 60 min |
| ARIMA | 90 min | 135 min | 180 min |

---

## Troubleshooting Common Issues

### Training Hangs

**Solution**:
1. Check error logs in `backend/logs/training.log`
2. Fix data quality issues if needed
3. Use `--force-retrain` to retry
4. Check `backend/logs/training.log` for errors

### Out of Memory

**Solution**:
1. Close other applications
2. Ensure sufficient RAM (8GB minimum, 16GB recommended)
3. Train models sequentially (not in parallel)

### Poor R² Score

**Solution**:
1. Retrain with `--force-retrain`
2. Check data quality in `permanent/` directory
3. Verify 37 features are calculated correctly
4. See troubleshooting in `MODEL_TRAINING.md`

### Frontend Doesn't Show Predictions

**Solution**:
1. Check backend is running: `http://localhost:5000/health`
2. Check model is trained: `python status.py`
3. Check browser console for errors (F12)
4. Verify CORS is enabled in backend

---

## Training Tips & Best Practices

### 1. Train in Recommended Order
- Start with fast models (Linear Regression, Decision Tree)
- Test predictions before training slow models
- Save ARIMA for last (takes 1.5-3 hours)

### 2. Monitor Resources
- Keep Task Manager/Activity Monitor open
- Watch memory usage during training
- Close unnecessary applications

### 3. Test After Each Model
- Run `python status.py` after each training
- Test predictions on frontend
- Verify R² is within expected range

### 4. Save Logs
- Logs are automatically saved to `backend/logs/`
- Keep logs for troubleshooting
- Use `--force-retrain` if retrying failed models

### 5. Single-Pass Training
- All models use single-pass training
- Loads complete dataset (~1000 stocks, 5 years) into memory
- Trains with one model.fit() call per model
- Progress tracking during data loading phase

---

## Quick Reference

### Essential Commands

```bash
# Check status
python status.py

# Train specific model
python backend/training/train_full_dataset.py --model MODEL_NAME

# Force retrain
python backend/training/train_full_dataset.py --model MODEL_NAME --force-retrain

# Train with limited stocks (for testing)
python backend/training/train_full_dataset.py --model MODEL_NAME --max-stocks 100

# Start backend
cd backend && python main.py

# Start frontend
cd frontend && npm run dev

# Check health
curl http://localhost:5000/health
```

### File Locations

- **Models**: `backend/models/{model_name}/` (each model in its own subdirectory)
  - Example: `backend/models/linear_regression/linear_regression_model.pkl`
  - Example: `backend/models/ann/ann_model.pkl` + `ann_model_model.h5`
- **Logs**: `backend/logs/`
- **Data**: `permanent/ind_stocks/` and `permanent/us_stocks/`
- **Config**: `backend/prediction/config.py`
- **Status**: `backend/models/model_status.json`

---

## Next Steps After Training

1. ✅ Verify all 9 models have R² > minimum threshold
2. ✅ Test predictions on frontend for multiple stocks
3. ✅ Update ensemble weights in `backend/prediction/config.py` if needed
4. ✅ Deploy to production (if applicable)
5. ✅ Set up automated retraining schedule (optional)

---

## Support & Resources

- **Troubleshooting**: See `documentation/MODEL_TRAINING.md`
- **API Documentation**: See `backend/README.md`
- **Frontend Guide**: See `frontend/README.md`
- **Data Integration**: See `documentation/UPSTOX_INTEGRATION.md`

---

**Last Updated**: October 22, 2025  
**Version**: 2.1 (Pre-Training State)  
**Status**: All 9 models ready for fresh training - 0/9 trained

