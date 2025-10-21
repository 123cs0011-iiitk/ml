# MODEL_TRAINING.md

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

# 2 Features

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

**Note:** Volume-based features are excluded because some stocks do not have volume data.


# 3 data Directory Structure

data/
├── index_ind_stocks_dynamic.csv          # Indian stocks index (~500 companies)
├── index_us_stocks_dynamic.csv           # US stocks index (~500 companies)
│
├── past/                                # Historical data (2020-2024)
│   ├── ind_stocks/individual_files/     # 500 Indian stock CSV files
│   └── us_stocks/individual_files/      # 501 US stock CSV files
│
├── latest/                              # Current/recent data
│   ├── ind_stocks/
│   │   ├── individual_files/            # 33 Indian stock CSV files
│   │   └── latest_prices.csv            # Aggregated latest prices
│   │
│   └── us_stocks/
│       ├── individual_files/            # 8 US stock CSV files
│       └── latest_prices.csv            # Aggregated latest prices
│
└── future/                              # Future/prediction data
    ├── ind_stocks/individual_files/     # 2 Indian stock CSV files
    └── us_stocks/individual_files/      # 3 US stock CSV files


# 4 Index Files Structure 

Indian Stocks Index (`index_ind_stocks_dynamic.csv`)
symbol,company_name,sector,market_cap,headquarters,exchange,currency,isin

US Stocks Index (`index_us_stocks_dynamic.csv`)
symbol,company_name,sector,market_cap,headquarters,exchange,currency,isin

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

