# Stock Prediction Module

This module provides comprehensive stock price prediction capabilities using multiple machine learning algorithms. It generates predictions for multiple time horizons (1D, 1W, 1M, 1Y, 5Y) and stores results in the `data/future/` directory.

## Overview

The prediction system combines historical data from `data/past/` and current data from `data/latest/` to train multiple ML algorithms and generate ensemble predictions. All predictions are saved to CSV files in `data/future/` with the same structure as the historical data.

## Features

- **Multiple Algorithms**: Uses 7 different ML algorithms (Ridge, Lasso, ElasticNet, Random Forest, KNN, SVR, LSTM, ARIMA)
- **Multiple Time Horizons**: Predicts for 1D, 1W, 1M, 1Y, and 5Y timeframes
- **Ensemble Predictions**: Combines predictions from all algorithms with weighted averaging
- **Confidence Intervals**: Provides confidence bounds for each prediction
- **Feature Engineering**: Creates technical indicators (RSI, moving averages, volatility, etc.)
- **Data Validation**: Ensures data quality before training
- **Batch Processing**: Can process all stocks or specific categories

## Directory Structure

```
backend/prediction/
├── __init__.py              # Module initialization
├── config.py                # Configuration parameters
├── data_loader.py           # Data loading and preprocessing
├── predictor.py             # Main prediction orchestrator
├── prediction_saver.py      # Save predictions to CSV
├── run_predictions.py       # Standalone execution script
└── README.md               # This documentation

data/future/
├── us_stocks/
│   └── individual_files/    # One CSV per US stock
└── ind_stocks/
    └── individual_files/    # One CSV per Indian stock
```

## Quick Start

### 1. Check Configuration
```bash
python backend/prediction/run_predictions.py --config-check
```

### 2. Run Quick Test
```bash
python backend/prediction/run_predictions.py --test
```

### 3. Predict Specific Stock
```bash
python backend/prediction/run_predictions.py --symbol AAPL --category us_stocks
```

### 4. Predict All Stocks
```bash
python backend/prediction/run_predictions.py
```

## Usage Examples

### Command Line Options

```bash
# Show help
python backend/prediction/run_predictions.py --help

# Check configuration
python backend/prediction/run_predictions.py --config-check

# Show existing predictions summary
python backend/prediction/run_predictions.py --summary

# Run test with limited stocks
python backend/prediction/run_predictions.py --test

# Predict specific category with limit
python backend/prediction/run_predictions.py --category us_stocks --max-stocks 10

# Predict specific stock
python backend/prediction/run_predictions.py --symbol RELIANCE --category ind_stocks
```

### Programmatic Usage

```python
from prediction.predictor import StockPredictor
from prediction.config import config

# Initialize predictor
predictor = StockPredictor()

# Predict single stock
success = predictor.predict_stock('AAPL', 'us_stocks')

# Predict all stocks in category
results = predictor.predict_all_stocks(category='us_stocks', max_stocks=10)

# Get prediction summary
summary = predictor.get_prediction_summary()
```

## Configuration

The `config.py` file contains all configuration parameters:

### Time Horizons
- **1D**: 1 day ahead
- **1W**: 1 week (7 days) ahead  
- **1M**: 1 month (30 days) ahead
- **1Y**: 1 year (365 days) ahead
- **5Y**: 5 years (1825 days) ahead

### Model Weights
```python
MODEL_WEIGHTS = {
    'ridge': 0.15,
    'lasso': 0.15,
    'elasticnet': 0.10,
    'random_forest': 0.20,
    'knn': 0.10,
    'svr': 0.15,
    'lstm': 0.10,
    'arima': 0.05
}
```

### Data Requirements
- **Minimum Training Days**: 252 (1 year)
- **Lookback Days**: 60 (for feature creation)
- **Test Size**: 20% of data

## Output Format

Each stock's prediction file contains:

```csv
date,horizon,predicted_price,confidence_low,confidence_high,algorithm_used,currency,last_updated,model_accuracy,data_points_used
2025-10-19,1D,250.50,248.20,252.80,"Ridge|Lasso|LSTM|RF|KNN|SVR|ARIMA",USD,2025-10-18T10:30:00,0.8542,1250
2025-10-25,1W,252.30,245.10,259.50,"Ridge|Lasso|LSTM|RF|KNN|SVR|ARIMA",USD,2025-10-18T10:30:00,0.8234,1250
2025-11-18,1M,258.75,240.20,277.30,"Ridge|Lasso|LSTM|RF|KNN|SVR|ARIMA",USD,2025-10-18T10:30:00,0.7891,1250
2026-10-18,1Y,285.60,220.40,350.80,"Ridge|Lasso|LSTM|RF|KNN|SVR|ARIMA",USD,2025-10-18T10:30:00,0.7123,1250
2030-10-18,5Y,420.15,280.50,559.80,"Ridge|Lasso|LSTM|RF|KNN|SVR|ARIMA",USD,2025-10-18T10:30:00,0.6543,1250
```

## Algorithms Used

### 1. Linear Models
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization  
- **ElasticNet**: Combined L1 and L2 regularization

### 2. Tree-Based Models
- **Random Forest**: Ensemble of decision trees

### 3. Instance-Based Models
- **K-Nearest Neighbors**: Pattern matching on similar historical periods

### 4. Support Vector Models
- **Support Vector Regression (SVR)**: RBF kernel with hyperparameter tuning

### 5. Neural Networks
- **LSTM**: Long Short-Term Memory for time series

### 6. Statistical Models
- **ARIMA**: AutoRegressive Integrated Moving Average

## Feature Engineering

The system creates numerous technical indicators:

### Price Features
- Price changes (absolute and percentage)
- Moving averages (5, 10, 20, 50 days)
- Moving average ratios
- High-Low ratios
- Open-Close ratios

### Volatility Features
- Rolling volatility (20-day window)
- Price position within day range

### Technical Indicators
- **RSI**: Relative Strength Index (14-day period)
- Volume moving averages and ratios
- Lagged features (1, 2, 3, 5, 10 days)

### Time Features
- Day of week
- Month
- Quarter

## Data Flow

1. **Data Loading**: Combines historical and latest data
2. **Feature Engineering**: Creates technical indicators
3. **Data Validation**: Ensures quality and completeness
4. **Model Training**: Trains all 7 algorithms
5. **Prediction Generation**: Creates predictions for all time horizons
6. **Ensemble Creation**: Combines predictions with weighted averaging
7. **Confidence Calculation**: Computes confidence intervals
8. **Result Saving**: Saves to CSV files in `data/future/`

## Error Handling

The system includes comprehensive error handling:

- **Data Quality Validation**: Checks for missing data, price ranges, volatility
- **Model Training Errors**: Continues with other models if one fails
- **Prediction Errors**: Logs errors and continues with next stock
- **File I/O Errors**: Handles missing files gracefully

## Performance Considerations

- **Parallel Processing**: Uses multiple workers for batch processing
- **Caching**: Caches loaded data to avoid repeated file reads
- **Chunked Processing**: Processes stocks in chunks to manage memory
- **Model Persistence**: Can save/load trained models (future feature)

## Logging

The system provides detailed logging:

- **INFO**: General progress and results
- **DEBUG**: Detailed model training and prediction steps
- **WARNING**: Data quality issues and model failures
- **ERROR**: Critical errors that stop processing

Logs are written to both console and `prediction.log` file.

## Troubleshooting

### Common Issues

1. **No Data Available**
   - Check if stock files exist in `data/past/` and `data/latest/`
   - Verify index files contain the stock symbol

2. **Model Training Failures**
   - Check data quality (minimum 252 days required)
   - Verify feature engineering produces valid features

3. **Memory Issues**
   - Use `--max-stocks` to limit processing
   - Process categories separately

4. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path includes backend directory

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Future Enhancements

- **Model Persistence**: Save/load trained models
- **Real-time Updates**: Incremental prediction updates
- **Advanced Ensembling**: Dynamic weight adjustment based on performance
- **More Algorithms**: Additional ML models
- **API Integration**: REST API endpoints for predictions
- **Visualization**: Charts and graphs for predictions

## Dependencies

- pandas
- numpy
- scikit-learn
- tensorflow (for LSTM)
- statsmodels (for ARIMA)
- joblib (for model persistence)

## License

This module is part of the Stock Prediction System project.
