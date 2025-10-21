# MODEL_TRAINING.md

---
# 1 Models

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
# 2 Features Summary

- Open Price
- Close Price
- High Price
- Low Price
- Adjusted Close Price
- Moving Averages (5-day, 10-day, 20-day, etc.)
- Exponential Moving Averages (EMA)
- Relative Strength Index (RSI)
- Bollinger Bands (Upper, Middle, Lower)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Momentum
- Rate of Change (ROC)
- Daily Price Change
- Daily Price Percentage Change
- Cumulative Returns
- Log Returns
- Price Differences
- Standard Deviation (Rolling)
- Variance (Rolling)
- Min/Max Price in Rolling Window
- Trend Indicators
- Volatility Indicators
- Any other derived price-based indicators

---
# 3 Features Detail

#### 1-2. Basic Price Features (2 features)
1. `price_change` - Percentage change in closing price
2. `price_change_abs` - Absolute value of price change

#### 3-12. Moving Averages (MA) (10 features)
3. `ma_5` - 5-day moving average
4. `ma_5_ratio` - Close price / MA_5
5. `ma_10` - 10-day moving average
6. `ma_10_ratio` - Close price / MA_10
7. `ma_20` - 20-day moving average
8. `ma_20_ratio` - Close price / MA_20
9. `ma_50` - 50-day moving average
10. `ma_50_ratio` - Close price / MA_50
11. `ma_200` - 200-day moving average
12. `ma_200_ratio` - Close price / MA_200

#### 13. Volatility (1 feature)
13. `volatility` - Rolling standard deviation of price changes

#### 14. Momentum Indicator (1 feature)
14. `rsi` - Relative Strength Index (14-period)   

#### 15-16. Intraday Price Ratios (2 features)
15. `hl_ratio` - High / Low ratio
16. `oc_ratio` - Open / Close ratio

#### 17. Price Position (1 feature)
17. `price_position` - Position within day's high-low range

#### 18-22. Lagged Price Features (5 features)
18. `close_lag_1` - Close price 1 day ago
19. `close_lag_2` - Close price 2 days ago
20. `close_lag_3` - Close price 3 days ago
21. `close_lag_5` - Close price 5 days ago
22. `close_lag_10` - Close price 10 days ago

#### 23-31. Rolling Statistics (9 features)
23. `close_std_5` - 5-day rolling standard deviation
24. `close_std_10` - 10-day rolling standard deviation
25. `close_std_20` - 20-day rolling standard deviation
26. `close_min_5` - 5-day rolling minimum
27. `close_min_10` - 10-day rolling minimum
28. `close_min_20` - 20-day rolling minimum
29. `close_max_5` - 5-day rolling maximum
30. `close_max_10` - 10-day rolling maximum
31. `close_max_20` - 20-day rolling maximum

#### 32-34. Time-Based Features (3 features)
32. `day_of_week` - Day of the week (0-6)
33. `month` - Month of the year (1-12)
34. `quarter` - Quarter of the year (1-4)

#### 35-37. Raw OHLC Data (3 features)
35. `open` - Opening price
36. `high` - High price
37. `low` - Low price

>[!summary]
> Measures the speed and magnitude of recent price changes to identify overbought or oversold conditions (0-100); a momentum oscillator 
>Formula: RSI = 100 - (100 / (1 + RS)), where RS = Average Gain over 14 periods / Average Loss over 14 periods
---
# Feature Calculation Process

1. Load OHLC data (Open, High, Low, Close)
2. Calculate moving averages (5, 10, 20, 50, 200 days)
3. Calculate technical indicators (RSI, volatility)
4. Generate lagged features (1, 2, 3, 5, 10 days back) for close prices
5. Compute rolling statistics (5, 10, 20 day windows)
6. Add time-based features (day, month, quarter)
7. Normalize and clean data (handle NaN, infinity values)
8. Create feature matrix X and target vector y

### Notes

- Features are standardized across all models for consistency
- NaN values from rolling calculations are handled appropriately
- All infinity values are replaced with median or bounded values
- Data is scaled using StandardScaler for sensitive models (SVM, KNN, ANN, CNN)



**Note:** Volume-based features are completely excluded from ML models because volume data is unavailable or unreliable for all stocks. The volume column exists in raw CSV files for data structure compatibility but is set to NaN during prediction and never used in any feature calculations or model training.

### Training Data Source

- **Training Source**: ONLY `data/past/{category}/individual_files/` (5 years historical data)
- **Target Variable**: `close` (next day's closing price)
- **Stock Coverage**: ~1,000 stocks (500 Indian + 500 US)
- **Total Samples**: ~1,200,000 data points
- **Consistency**: All stocks trained on same 5-year time period (2020-2025)
- **Note**: `data/latest/` is NOT used for training to ensure consistency

### Prediction Data Flow

When making predictions with trained models:
1. User selects stock and time horizon (1D/1W/1M/1Y/5Y)
2. System fetches current live price from data source/API
3. Historical data loaded from `data/past/{category}/individual_files/`
4. Current price appended to historical data
5. 37 features calculated from combined data (historical + current)
6. Trained model generates prediction for selected horizon
7. Prediction returned with confidence intervals

### Volume Data Handling

**Storage vs Usage:**
- Volume column **exists** in raw CSV files (`data/past/`, `data/latest/`) for data structure compatibility
- Volume is **never used** in ML feature calculations, model training, or predictions
- During prediction, volume is set to `np.nan` (not 0 or any numeric value)
- Tests use data without volume to match production reality

**Implementation Details:**
- `backend/prediction/data_loader.py:356` - Volume excluded from training columns
- `backend/prediction/data_loader.py:396` - Volume in exclude list for features
- `backend/prediction/data_loader.py:104` - Volume set to `np.nan` when appending live prices
- `backend/prediction/data_loader.py:504` - Volume removed from data quality validation
- `backend/tests/test_models.py:64` - Test data has no volume column
- All algorithm files explicitly state "Volume is excluded from all calculations"

**Summary:** Volume is purely symbolic in the codebase - it exists in data files but is completely ignored by all ML operations.


---
# 4 Training Dataset Size Estimation


---
# 3. Directory Structure
```
data/
├── past/                           # Historical stock data (training data)
│   ├── ind_stocks/
│   │   └── individual_files/       # 500 Indian stocks (2020-2024)
│   └── us_stocks/
│       └── individual_files/       # 501 US stocks (2020-2024)
│
├── latest/                         # Most recent real-time data
│   ├── ind_stocks/
│   │   └── individual_files/      
│   └── us_stocks/
│       └── individual_files/      
│
├── future/                         # ML-generated predictions
│   ├── ind_stocks/
│   │   └── individual_files/      
│   └── us_stocks/
│       └── individual_files/      
│
├── index_ind_stocks_dynamic.csv   # Index of all Indian stocks
└── index_us_stocks_dynamic.csv    # Index of all US stocks
```

# 4 Index Files Structure 

Indian Stocks Index (`index_ind_stocks_dynamic.csv`)
symbol,company_name,sector,market_cap,headquarters,exchange,currency,isin
```csv
ABBOTINDIA,Abbott India,Healthcare,,India,NSE,INR,INE358A01014
ABCAPITAL,Aditya Birla Capital,Financial Services,,India,NSE,INR,INE674K01013
ABFRL,Aditya Birla Fashion and Retail,Consumer Services,,India,NSE,INR,INE647O01011
ABLBL,Aditya Birla Lifestyle Brands,Consumer Services,,India,NSE,INR,INE14LE01019
```

US Stocks Index (`index_us_stocks_dynamic.csv`)
symbol,company_name,sector,market_cap,headquarters,exchange,currency
```csv
AAPL,Apple,Technology,,"Cupertino, California",NASDAQ,USD
ABBV,AbbVie,Healthcare,,"North Chicago, Illinois",NYSE,USD
ABNB,Airbnb,Consumer Discretionary,,"San Francisco, California",NYSE,USD
```

# 5 Individual Files Structure

date,open,high,low,close,volume,adjusted_close,currency

Latest Prices Files

Indian Latest Prices (`latest/ind_stocks/latest_prices.csv`)
symbol,price,timestamp,source,company_name,currency

US Latest Prices (`latest/us_stocks/latest_prices.csv`)
symbol,price,timestamp,source,company_name,currency,sector,market_cap,headquarters,exchange

Column Descriptions

Stock Data Columns
- `date`: Trading date with timezone
- `open`: Opening price
- `high`: Highest price of the day
- `low`: Lowest price of the day
- `close`: Closing price
- `volume`: Trading volume
- `adjusted_close`: Adjusted closing price (for splits/dividends)
- `currency`: Currency (INR/USD)

Index File Columns
- `symbol`: Stock symbol
- `company_name`: Full company name
- `sector`: Business sector
- `market_cap`: Market capitalization
- `headquarters`: Company headquarters location
- `exchange`: Stock exchange
- `currency`: Trading currency
- `isin`: International Securities Identification Number

Latest Prices Columns
- `symbol`: Stock symbol
- `price`: Current price
- `timestamp`: Price timestamp
- `source`: Data source (Upstox/Finnhub)
- `company_name`: Company name
- `currency`: Currency
- `sector`: Business sector (US only)
- `market_cap`: Market capitalization (US only)
- `headquarters`: Location (US only)
- `exchange`: Exchange (US only)

