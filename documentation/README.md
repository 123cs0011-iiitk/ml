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
python status.py                                          # Check model training status
python main.py                                            # Start backend server
cd frontend && npm run dev                                # Start frontend
python backend/scripts/verify_indian_isins.py --count 25  # Verify Indian stock ISINs
```

## 📊 Data Information

**Indian Stocks**: All 500 stocks include verified ISINs (12-char format: INExxxxxxxx)
- ✅ 100% verified with live Upstox API
- Required for real-time price fetching

**US Stocks**: Use ticker symbols only (NO ISINs)
- Finnhub API doesn't require ISINs

See individual guides above for detailed information.
