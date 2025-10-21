# Documentation

Technical documentation for the Stock Price Prediction System.

## 📚 Available Documentation

### **Core Documentation**
- **[Main README](../README.md)** - Project overview and quick start
- **[Backend README](../backend/README.md)** - Backend API documentation

### **API & Integration Guides**
- **[Backend README](../backend/README.md)** - Complete API documentation with usage examples
- **[Upstox Integration](UPSTOX_INTEGRATION.md)** - Complete Indian stock market integration guide

### **Technical Implementation**
- **[Currency Conversion](CURRENCY_CONVERSION.md)** - Currency conversion implementation and troubleshooting

## 🔧 System Status

### **Model Training Status**
Check current model status with:
```bash
python status.py           # Table format
python status.py --json    # JSON format
python status.py --simple  # Simple format
```

### **Current Model Performance**
- **✅ Working**: Random Forest (R²=0.994), Decision Tree (R²=0.85)
- **⚠️ Poor**: SVM (R²=-26.2), KNN (R²=-27.9), ANN (R²=-9.7M)
- **❌ Failed**: Linear Regression, CNN, ARIMA, Autoencoder

### **System Components**
- **Data Fetching**: ✅ Working (US via Finnhub, Indian via Upstox)
- **Frontend**: ✅ Working (React + TypeScript)
- **Backend API**: ✅ Working (Flask)
- **ML Predictions**: ⚠️ Unreliable (model performance issues)

## 🔑 ISIN Requirements (Critical)

### Indian Stocks - ISINs MANDATORY
**ISINs are absolutely required** for Upstox API integration:
- ✅ All 500 Indian stocks have ISINs (100% coverage)
- ✅ ISINs stored in both permanent and dynamic indexes
- ⚠️ Without ISINs, Upstox returns "wrong ISIN number" errors
- 📍 ISIN format: 12-character alphanumeric (e.g., `INE009A01021` for Infosys)

**Index Files:**
- `permanent/ind_stocks/index_ind_stocks.csv` - Permanent index with ISINs
- `data/index_ind_stocks_dynamic.csv` - Dynamic index with ISINs

**How ISINs were populated:**
- Downloaded from Upstox instruments file (`complete.csv.gz`)
- Extracted ISINs from `instrument_key` column for `NSE_EQ` exchange
- Achieved 100% coverage for all 500 Indian stocks

### US Stocks - ISINs NOT Required
- Finnhub API uses ticker symbols (not ISINs)
- ISINs are optional for US stocks
- System works perfectly without ISINs

### Real-time Data Fetching
**US Stocks:** Finnhub API → Permanent directory (fallback)  
**Indian Stocks:** Upstox API → Permanent directory (fallback)  
**Note:** yfinance removed from real-time fetching (only used for historical data)
