# Documentation Hub

Navigation for Stock Price Prediction System documentation.

## 📚 Available Guides

- **[Main README](../README.md)** - Project overview, quick start, system status
- **[Backend API](../backend/README.md)** - API endpoints, training system, configuration
- **[Upstox Integration](UPSTOX_INTEGRATION.md)** - Indian stock market API setup
- **[Currency Conversion](CURRENCY_CONVERSION.md)** - USD/INR conversion implementation
- **[Model Training](MODEL_TRAINING.md)** - ML training system documentation

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

## 🤖 Model Training Status (Oct 23, 2025)

**Progress**: 2/7 models trained
- ✅ Linear Regression (R²=-0.002 - needs investigation)
- ✅ Decision Tree (R²=0.001 - needs investigation)
- 🔄 Random Forest (next to train - expected R²>0.90)
- ⏳ SVM, KNN, ARIMA, Autoencoder (pending)

**Architecture**: Standalone trainers in `backend/training/basic_models/` and `advanced_models/`

See individual guides above for detailed information.
