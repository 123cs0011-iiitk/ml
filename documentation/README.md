# Documentation Hub

Navigation for Stock Price Prediction System documentation.

## 📚 Available Guides

### Core Documentation
- **[Main README](../README.md)** - Project overview, quick start, system status
- **[Backend API](../backend/README.md)** - API endpoints, training system, configuration
- **[Offline Mode](OFFLINE_MODE.md)** - Running without API keys using permanent directory

### API Integration
- **[Upstox Integration](UPSTOX_INTEGRATION.md)** - Indian stock market API setup
- **[Quick Auth Guide](QUICK_AUTH_GUIDE.md)** - Daily Upstox authentication (30 seconds)
- **[Currency Conversion](CURRENCY_CONVERSION.md)** - USD/INR conversion implementation

### Training & Development
- **[Model Training](MODEL_TRAINING.md)** - ML training system documentation
- **[Training Guide](TRAINING_GUIDE.md)** - Step-by-step training instructions
- **[Pre-Flight Checklist](PRE_FLIGHT_CHECKLIST.md)** - Ready-to-train verification

### Technical References
- **[Sync Fixes Summary](SYNC_FIXES_SUMMARY.md)** - Import and data format fixes

## 🔍 Quick Commands

```bash
# Check system status
python status.py

# Train individual models (standalone trainers)
python backend/training/basic_models/linear_regression/trainer.py
python backend/training/basic_models/decision_tree/trainer.py
python backend/training/basic_models/random_forest/trainer.py
python backend/training/basic_models/svm/trainer.py
python backend/training/advanced_models/knn/trainer.py
python backend/training/advanced_models/arima/trainer.py
python backend/training/advanced_models/autoencoder/trainer.py

# Start backend server
cd backend && python main.py

# Start frontend
cd frontend && npm run dev

# Verify Indian stock ISINs
python backend/scripts/verify_indian_isins.py --count 25
```

## 📊 Data Information

**Indian Stocks**: All 500 stocks include verified ISINs (12-char format: INExxxxxxxx)
- ✅ 100% verified with live Upstox API
- Required for real-time price fetching

**US Stocks**: 501 stocks use ticker symbols only (NO ISINs)
- Finnhub API doesn't require ISINs

## 🤖 Model Training Status (Oct 24, 2025)

**Progress**: 4/7 models trained
- ✅ Linear Regression (R²=-0.002)
- ✅ Decision Tree (R²=0.001)
- ✅ Random Forest (R²=0.024) - Best performer
- ✅ SVM (R²=-0.0055) - Working via UI
- ⏳ **KNN** - Next to train
- ⏳ ARIMA - Pending
- ⏳ Autoencoder - Pending

## 🆕 Recent Updates (Oct 24, 2025)

- ✅ **Offline Mode**: System works without API keys using permanent directory
- ✅ **Card Synchronization**: Info and prediction cards always use same data source
- ✅ **Visual Indicators**: Amber warnings show when using offline data
- ✅ **SVM Integration**: Fixed predictor to allow explicit model selection

**Architecture**: Standalone trainers in `backend/training/basic_models/` and `advanced_models/`

See individual guides above for detailed information.
