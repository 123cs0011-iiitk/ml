# Offline Mode Guide

**Last Updated:** October 24, 2025

## Overview

The Stock Prediction System is designed to work completely offline without any API keys, using pre-loaded historical data from the `permanent/` directory as a READ-ONLY fallback.

---

## How Offline Mode Works

### Data Loading Hierarchy

The system uses a three-tier fallback strategy:

```
1st Priority: data/past/{category}/individual_files/{SYMBOL}.csv
     ↓ (if not found)
2nd Priority: permanent/{category}/individual_files/{SYMBOL}.csv
     ↓ (if not found)
3rd: Error message with clear diagnostics
```

### Key Principles

1. **Permanent Directory is READ-ONLY**
   - Never written to
   - Contains 1001 pre-loaded stocks (2020-2024 data)
   - Serves as ultimate fallback for offline usage

2. **Data/Past is Primary**
   - Used for training and predictions when available
   - Can be updated with new data
   - Falls back to permanent if empty

3. **Automatic Fallback**
   - No configuration needed
   - System automatically detects missing data
   - Seamless transition between sources

---

## What Works Offline

### ✅ Fully Functional (No API Keys Required)

- **Stock Information Cards**
  - Display metadata for 1001 stocks
  - Show company name, sector, market cap
  - Historical prices from permanent directory

- **Historical Charts**
  - Complete 5-year OHLCV data (2020-2024)
  - Interactive Recharts visualization
  - Support for 1W, 1M, 1Y, 5Y time periods

- **ML Predictions**
  - All trained models work with offline data
  - Predictions based on permanent directory
  - Visual indicators show data source and date

- **Search Functionality**
  - Full-text search across 1001 stocks
  - Symbol and company name matching
  - Fast index-based lookup

- **Technical Indicators**
  - 38 indicators calculated from OHLC data
  - Moving averages, RSI, volatility
  - All computed client-side

### ❌ Requires API Keys

- **Live Price Fetching**
  - Finnhub API (US stocks)
  - Upstox API (Indian stocks)
  - Real-time price updates

- **Currency Conversion (Live)**
  - Real-time USD/INR rates
  - Falls back to static rate if API unavailable

---

## Offline Setup (Step by Step)

### For Fresh Clone (No API Keys)

#### 1. Clone Repository
```bash
git clone <repo-url>
cd ml
```

#### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Create Empty .env File (REQUIRED)
```bash
# Windows
type nul > .env

# Mac/Linux
touch .env
```

**Important**: The `.env` file must exist (even if empty) for the system to start properly. This is required by `python-dotenv`.

#### 4. Start Backend
```bash
python main.py
```

Expected output:
```
Loading AAPL from permanent directory
✅ Fetched AAPL from permanent directory (READ-ONLY fallback): $150.25 from 2024-12-31
```

#### 5. Frontend Setup (New Terminal)
```bash
cd frontend
npm install
npm run dev
```

#### 6. Access Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000

---

## Visual Indicators

### Info Card (Offline Data)

When using offline data, you'll see an amber warning box:

```
┌────────────────────────────────────┐
│ 🗄️ Using offline data from 2024-12-31 │
└────────────────────────────────────┘
```

### Prediction Card (Offline Data)

Similarly, predictions show:

```
┌────────────────────────────────────────────────┐
│ 🗄️ Prediction based on offline data from 2024-12-31 │
└────────────────────────────────────────────────┘
```

### Live Data Indicators

When APIs are configured and working:
- Info card shows: "Live data from finnhub/upstox"
- No warning boxes
- Real-time timestamp displayed

---

## Data Synchronization

### Info Card & Prediction Card Sync

Both cards always use the **same data source**:

**Scenario 1: Online (APIs Available)**
- Info Card: Uses live API → Shows current price
- Prediction Card: Uses same price → Prediction based on live data
- ✅ **Both synchronized on live data**

**Scenario 2: Offline (No APIs)**
- Info Card: Falls back to permanent → Shows price from 2024-12-31
- Prediction Card: Uses same permanent data → Prediction from 2024-12-31
- ✅ **Both synchronized on offline data**

**Scenario 3: Partial (Some stocks offline)**
- System automatically detects per-symbol
- Each stock uses best available source
- Visual indicators show which source is used

---

## Transitioning to Live Mode

### Adding API Keys

When ready for live data, create `backend/.env`:

```bash
# US Stocks (Finnhub)
FINNHUB_API_KEY=your_finnhub_key_here

# Indian Stocks (Upstox - requires OAuth)
UPSTOX_API_KEY=your_upstox_key_here
UPSTOX_CLIENT_ID=your_client_id_here
UPSTOX_CLIENT_SECRET=your_client_secret_here
```

### Upstox Daily Authentication

Upstox tokens expire daily at 3:30 AM IST. Run:

```bash
cd backend
python scripts/setup_upstox_oauth.py
```

See [QUICK_AUTH_GUIDE.md](QUICK_AUTH_GUIDE.md) for detailed instructions.

---

## Data Organization

### Directory Structure

```
ml/
├── data/                     # Dynamic data
│   ├── past/                 # Historical data (2020-2024)
│   │   ├── us_stocks/
│   │   │   └── individual_files/
│   │   │       └── AAPL.csv
│   │   └── ind_stocks/
│   │       └── individual_files/
│   │           └── RELIANCE.csv
│   ├── latest/               # Live updates (2025+)
│   └── future/               # Predictions output
│
├── permanent/                # READ-ONLY fallback
│   ├── us_stocks/
│   │   ├── index_us_stocks.csv
│   │   └── individual_files/
│   │       └── [501 CSV files]
│   └── ind_stocks/
│       ├── index_ind_stocks.csv
│       └── individual_files/
│           └── [500 CSV files]
```

### File Format

Each CSV file contains:
```csv
date,open,high,low,close,volume,adjusted_close,currency
2024-12-31,150.00,151.00,149.50,150.25,5000000,,USD
```

---

## Technical Implementation

### Backend Fallback Logic

**File**: `backend/prediction/data_loader.py`

```python
def _load_historical_data(self, symbol: str, category: str):
    # Try data/past first
    file_path = os.path.join(self.config.PAST_DATA_DIR, ...)
    
    if not os.path.exists(file_path):
        # Fallback to permanent directory
        file_path = os.path.join(self.config.PERMANENT_DATA_DIR, ...)
        logger.info(f"Loading {symbol} from permanent directory")
    
    return pd.read_csv(file_path)
```

### Frontend Indicators

**File**: `frontend/src/components/StockInfo.tsx`

```tsx
{!livePriceData.source_reliable && livePriceData.data_date && (
  <div className="bg-amber-50 border border-amber-200">
    <Database className="h-4 w-4" />
    <span>Using offline data from {livePriceData.data_date}</span>
  </div>
)}
```

---

## Troubleshooting

### Issue: "Could not load stock data"

**Cause**: Both `data/past/` and `permanent/` directories are empty or missing.

**Solution**:
1. Verify `permanent/` directory exists
2. Check files: `ls permanent/us_stocks/individual_files/`
3. Re-clone repository if needed

### Issue: Backend won't start without .env

**Cause**: `python-dotenv` requires `.env` file to exist.

**Solution**:
```bash
cd backend
touch .env  # or: type nul > .env (Windows)
python main.py
```

### Issue: Predictions show different date than info card

**Cause**: Should not happen - indicates sync bug.

**Solution**:
1. Check console logs for "data_date" mismatch warnings
2. Report issue with browser console logs
3. This is a bug if dates don't match

### Issue: Charts show no data

**Cause**: Chart endpoint not falling back to permanent.

**Solution**:
1. Check `/historical` endpoint in backend logs
2. Verify permanent files exist
3. Endpoint should show: "Loaded X records from permanent directory"

---

## Best Practices

### For Development

1. **Keep permanent/ intact**
   - Never modify files in permanent/
   - Use as reliable baseline
   - Treat as read-only archive

2. **Use data/past/ for updates**
   - Add new data here
   - System prioritizes over permanent
   - Can be cleared/updated safely

3. **Test offline mode regularly**
   - Remove/rename .env temporarily
   - Verify fallback works
   - Check visual indicators appear

### For Production

1. **Deploy with permanent/**
   - Include in deployment package
   - Ensures offline capability
   - 500MB of historical data

2. **Configure APIs optionally**
   - System works without them
   - Add for live features
   - Graceful degradation if APIs fail

3. **Monitor data sources**
   - Log which source is used
   - Alert if permanent is used unexpectedly
   - Track API availability

---

## Related Documentation

- [UPSTOX_INTEGRATION.md](UPSTOX_INTEGRATION.md) - Indian market API setup
- [QUICK_AUTH_GUIDE.md](QUICK_AUTH_GUIDE.md) - Daily Upstox authentication
- [MODEL_TRAINING.md](MODEL_TRAINING.md) - Training ML models offline
- [CURRENCY_CONVERSION.md](CURRENCY_CONVERSION.md) - USD/INR conversion

---

## Summary

**Offline Mode = Zero Configuration Required**

✅ No API keys needed  
✅ No .env configuration (just empty file)  
✅ No internet required  
✅ 1001 stocks pre-loaded  
✅ 5 years of historical data  
✅ All ML predictions work  
✅ Visual indicators show data source  

**The system is designed to work offline first, with live data as an enhancement.**

