# Stock Prediction Algorithms

This directory contains 16 stock-optimized machine learning algorithms for stock price prediction. All algorithms implement the `ModelInterface` base class and are designed specifically for stock data analysis.

## Overview

All algorithms are optimized for stock price prediction using OHLC (Open, High, Low, Close) data with technical indicators. **Volume data is not used** as some stocks may not have volume data available.

## ModelInterface Pattern

All algorithms follow a consistent interface pattern:

```python
class AlgorithmModel(ModelInterface):
    def __init__(self, **kwargs):
        super().__init__('Algorithm Name', **kwargs)
        # Initialize model-specific parameters
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """Train the model on stock data."""
        # Implementation here
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new stock data."""
        # Implementation here
        return predictions
    
    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        # Implementation here
    
    def load(self, path: str) -> 'ModelInterface':
        """Load a previously saved model from disk."""
        # Implementation here
        return self
```

## Available Algorithms

### Regression Algorithms

#### 1. Linear Regression (`linear_regression/`)
- **Purpose**: Linear regression for stock price prediction
- **Features**: Feature scaling, coefficient analysis, confidence intervals
- **Best for**: Linear relationships, interpretable results

#### 2. Random Forest (`random_forest/`)
- **Purpose**: Ensemble of decision trees for stock prediction
- **Features**: Feature importance, hyperparameter tuning, uncertainty estimation
- **Best for**: Non-linear relationships, feature importance analysis

#### 3. Decision Tree (`decision_tree/`)
- **Purpose**: Single decision tree for interpretable predictions
- **Features**: Tree rules, feature importance, hyperparameter tuning
- **Best for**: Interpretable models, rule-based decisions

#### 4. K-Nearest Neighbors (`knn/`)
- **Purpose**: Instance-based learning algorithm for stock prediction
- **Features**: Distance-based prediction, optimal k selection
- **Best for**: Local patterns, similarity-based prediction

#### 5. Support Vector Regression (`svm/`)
- **Purpose**: SVM for regression with multiple kernels
- **Features**: RBF, polynomial, linear kernels, hyperparameter tuning
- **Best for**: Non-linear relationships, high-dimensional data

### Deep Learning Algorithms

#### 6. Artificial Neural Network (`ann/`)
- **Purpose**: Multi-layer perceptron for stock prediction
- **Features**: Dropout, batch normalization, early stopping
- **Best for**: Complex non-linear relationships

#### 7. 1D Convolutional Neural Network (`cnn/`)
- **Purpose**: Time series CNN for stock prediction
- **Features**: Sequence modeling, LSTM integration, multi-step prediction
- **Best for**: Time series patterns, sequential dependencies

#### 8. ARIMA (`arima/`)
- **Purpose**: Time series forecasting for stock prices
- **Features**: Automatic order selection, confidence intervals, stationarity testing
- **Best for**: Time series forecasting, trend analysis

### Clustering Algorithms (Repurposed for Prediction)

#### 9. K-Means Clustering (`kmeans/`)
- **Purpose**: Market regime detection + prediction
- **Features**: Cluster-based prediction, regime identification
- **Best for**: Market state analysis, regime-based prediction

#### 10. DBSCAN Clustering (`dbscan/`)
- **Purpose**: Anomaly detection + prediction
- **Features**: Outlier detection, anomaly-based prediction
- **Best for**: Anomaly detection, outlier analysis

#### 11. Hierarchical Clustering (`hierarchical_clustering/`)
- **Purpose**: Stock grouping + prediction
- **Features**: Dendrogram analysis, cluster-based prediction
- **Best for**: Stock similarity analysis, group-based prediction

#### 12. General Clustering (`general_clustering/`)
- **Purpose**: Pattern-based prediction
- **Features**: Multiple clustering algorithms, pattern recognition
- **Best for**: Pattern discovery, cluster-based prediction

### Dimensionality Reduction + Prediction

#### 13. Principal Component Analysis (`pca/`)
- **Purpose**: Feature reduction + prediction
- **Features**: Dimensionality reduction, variance analysis
- **Best for**: High-dimensional data, noise reduction

#### 14. Singular Value Decomposition (`svd/`)
- **Purpose**: Feature extraction + prediction
- **Features**: Matrix factorization, feature extraction
- **Best for**: Feature extraction, dimensionality reduction

#### 15. t-SNE (`t_sne/`)
- **Purpose**: Pattern recognition + prediction
- **Features**: Non-linear dimensionality reduction, pattern visualization
- **Best for**: Pattern recognition, visualization

### Advanced Algorithms

#### 16. Autoencoders (`autoencoders/`)
- **Purpose**: Feature extraction + prediction
- **Features**: Unsupervised feature learning, reconstruction
- **Best for**: Feature learning, representation learning

## Technical Indicators

All algorithms use standardized technical indicators calculated from OHLC data:

### Moving Averages
- **SMA**: Simple Moving Average (5, 10, 20, 50, 200 days)
- **EMA**: Exponential Moving Average (12, 26 days)
- **Ratios**: Price to moving average ratios

### Momentum Indicators
- **RSI**: Relative Strength Index (14 periods)
- **MACD**: Moving Average Convergence Divergence (12, 26, 9)
- **Price Momentum**: 1, 5, 10 day returns

### Volatility Indicators
- **Bollinger Bands**: 20-period, 2 standard deviations
- **ATR**: Average True Range (14 periods)
- **Rolling Volatility**: Standard deviation over 5, 10, 20 days

### Price Patterns
- **High/Low Ratios**: Daily price range analysis
- **Open/Close Ratios**: Opening vs closing price analysis
- **Price Position**: Position within daily range

### Lagged Features
- **Price Lags**: 1, 2, 3, 5, 10 day price lags

### Rolling Statistics
- **Min/Max**: Rolling minimum and maximum over 5, 10, 20 days
- **Standard Deviation**: Rolling volatility over 5, 10, 20 days
- **Position**: Price position within rolling range

### Time-based Features
- **Day of Week**: Monday=0, Sunday=6
- **Month**: 1-12
- **Quarter**: 1-4

## Usage Example

```python
from algorithms.optimised.linear_regression.linear_regression import LinearRegressionModel
from algorithms.stock_indicators import StockIndicators

# Create model
model = LinearRegressionModel()

# Load stock data
df = pd.read_csv('stock_data.csv')

# Add technical indicators
df_with_features = model._create_technical_indicators(df)

# Prepare training data
X, y = StockIndicators.prepare_training_data(df_with_features)

# Train model
model.fit(X, y)

# Make predictions
predictions = model.predict(X[-10:])

# Save model
model.save('linear_model.pkl')

# Load model
loaded_model = LinearRegressionModel().load('linear_model.pkl')
```

## Model Persistence

All models support save/load functionality:

```python
# Save model
model.save('model.pkl')

# Load model
loaded_model = ModelClass().load('model.pkl')
```

## Performance Metrics

All models calculate standard regression metrics:

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error

## Integration with Prediction System

Models are automatically integrated into the prediction system via the `StockPredictor` class in `backend/prediction/predictor.py`.

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- tensorflow (for neural networks)
- statsmodels (for ARIMA)
- joblib (for model persistence)

## Notes

- **OHLC Data**: All algorithms use Open, High, Low, Close data (volume not used as some stocks may not have volume data)
- **Technical Indicators**: Standardized across all algorithms
- **ModelInterface**: Consistent interface for all algorithms
- **Stock-Specific**: All algorithms are optimized for stock price prediction