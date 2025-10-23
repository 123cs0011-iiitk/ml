# Detailed Model Training Guide

## Overview

This document explains how the machine learning model training system works from start to finish. It covers everything from loading raw CSV files to saving trained models that can predict stock prices.

**What happens during training:**
1. Load historical stock price data (5 years: 2020-2024)
2. Calculate 37 technical features from raw OHLC data
3. Split data into training/validation sets
4. Train 9 different ML models using batch processing
5. Save trained models for making predictions

**Training vs Prediction:**
- **Training**: Uses `data/past/` directory with 5 years of historical data to teach models patterns
- **Prediction**: Uses trained models + current price data to forecast future prices

---

## Complete Data Flow: Raw CSV → Trained Model

### Step 1: Load Raw CSV Data 📁

```python
# File: data/past/us_stocks/individual_files/AAPL.csv
df = pd.read_csv('AAPL.csv')
```

**Raw CSV Structure:**
```
date,open,high,low,close,volume
01-01-2020,296.24,300.60,295.05,300.35,135480400
02-01-2020,297.15,300.58,296.50,297.43,146322800
03-01-2020,295.81,299.68,295.00,297.43,143117600
...
```

**Result:** DataFrame with ~1,200 rows × 6 columns (date, OHLC, volume)

**Note:** Volume exists in files but is **NOT USED** for training (some stocks lack volume data)

---

### Step 2: Create Features 🔧

```python
df_with_features = data_loader.create_features(df)
```

This transforms raw OHLC data into 37 technical features that help models learn patterns:

#### 2.1 Moving Averages (10 features)
```python
df['ma_5'] = df['close'].rolling(5).mean()
df['ma_5_ratio'] = df['close'] / df['ma_5']
df['ma_10'] = df['close'].rolling(10).mean()
df['ma_10_ratio'] = df['close'] / df['ma_10']
# Same for: ma_20, ma_50, ma_200 + their ratios
```

**Purpose:** Show price trends over different time windows

#### 2.2 Price Changes (2 features)
```python
df['price_change'] = df['close'].pct_change()
df['price_change_abs'] = df['price_change'].abs()
```

**Purpose:** Capture daily price movement magnitude and direction

#### 2.3 Technical Indicators (3 features)
```python
df['rsi'] = calculate_rsi(df['close'], 14)  # Momentum indicator (0-100)
df['volatility'] = df['close'].rolling(20).std()  # Price stability
df['price_position'] = (close - min) / (max - min)  # Position in range
```

**Purpose:** Measure momentum, volatility, and relative positioning

#### 2.4 Intraday Ratios (2 features)
```python
df['hl_ratio'] = df['high'] / df['low']  # Daily range
df['oc_ratio'] = df['open'] / df['close']  # Open-close relationship
```

**Purpose:** Capture intraday price behavior

#### 2.5 Lagged Features (5 features)
```python
df['close_lag_1'] = df['close'].shift(1)  # Yesterday's price
df['close_lag_2'] = df['close'].shift(2)  # 2 days ago
df['close_lag_3'] = df['close'].shift(3)  # 3 days ago
df['close_lag_5'] = df['close'].shift(5)  # 5 days ago
df['close_lag_10'] = df['close'].shift(10)  # 10 days ago
```

**Purpose:** Let models see recent price history

#### 2.6 Rolling Statistics (9 features)
```python
# For windows: 5, 10, 20 days
df['close_std_5'] = df['close'].rolling(5).std()  # Volatility
df['close_min_5'] = df['close'].rolling(5).min()  # Recent low
df['close_max_5'] = df['close'].rolling(5).max()  # Recent high
# Same for windows: 10, 20
```

**Purpose:** Provide statistical context for recent price movements

#### 2.7 Time Features (3 features)
```python
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['month'] = df['date'].dt.month  # 1-12
df['quarter'] = df['date'].dt.quarter  # 1-4
```

**Purpose:** Capture seasonal and weekly patterns

#### 2.8 Raw OHLC (3 features)
```python
# Keep original: open, high, low
# (close is the prediction target, not a feature)
```

**Result:** DataFrame with ~1,200 rows × **43 columns** (6 original + 37 features)

---

### Step 3: Prepare Training Data 🎯

```python
X, y = data_loader.prepare_training_data(df_with_features)
```

**What Happens:**

```python
# 1. Select 37 feature columns (exclude date, volume, close)
feature_columns = [
    'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_200',
    'ma_5_ratio', 'ma_10_ratio', ...,
    'open', 'high', 'low', ...
]  # 37 total
X = df[feature_columns].values  # Shape: (1200, 37)

# 2. Create target = next day's closing price
y = df['close'].shift(-1).values  # Shift up by 1 day

# 3. Remove last row (no future price available)
X = X[:-1]  # Shape: (1199, 37)
y = y[:-1]  # Shape: (1199,)

# 4. Clean: replace NaN/Inf with valid numbers
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
y = np.nan_to_num(y, nan=mean_price)
```

**Result:**
- **X**: (1199, 37) → 1,199 rows, 37 features (inputs)
- **y**: (1199,) → 1,199 target prices (outputs)

**Training Goal:** Learn function: `y = f(X)` where f predicts tomorrow's price from today's features

---

### Step 4: Batch Loading 📦

```python
# Load 100 stocks at once to avoid memory overflow
for stock_batch in batches_of_100:
    all_X = []
    all_y = []
    
    for stock in stock_batch:
        df = load_stock_data(stock)           # Step 1: Load CSV
        df_feat = create_features(df)         # Step 2: Create 37 features
        X, y = prepare_training_data(df_feat)  # Step 3: Prepare arrays
        all_X.append(X)
        all_y.append(y)
    
    # Combine 100 stocks into one batch
    X_batch = np.vstack(all_X)  # Stack vertically
    y_batch = np.concatenate(all_y)  # Concatenate
```

**Result:**
- **1 batch** = 100 stocks
- **X_batch**: (119,900, 37) = 100 stocks × 1,199 rows each
- **y_batch**: (119,900,) = 100 stocks × 1,199 targets
- **Memory**: ~73 MB per batch

---

### Step 5: Train Model 🚀

Different models use different training strategies:

```python
# A) Incremental Strategy (Linear Regression)
for batch in all_batches:
    model.partial_fit(X_batch, y_batch)  # Update model incrementally
    # Model learns from this batch, keeps knowledge from previous batches

# B) Accumulate Strategy (Random Forest, Decision Tree)
accumulated_X = []
accumulated_y = []
for batch in all_batches:
    accumulated_X.append(X_batch)  # Just collect
    accumulated_y.append(y_batch)
# After all batches collected:
X_all = np.vstack(accumulated_X)
y_all = np.concatenate(accumulated_y)
model.fit(X_all, y_all)  # Train once on everything

# C) Subsample Strategy (SVM, KNN)
X_sample = X_batch[::2]  # Take every 2nd row = 50% of data
y_sample = y_batch[::2]
model.fit(X_sample, y_sample)  # Train on sample (faster, less memory)

# D) Keras Batch Strategy (ANN, CNN, Autoencoder)
model.fit(X_batch, y_batch, batch_size=32, epochs=50)
# Keras internally splits X_batch into mini-batches of 32 samples
# Trains for 50 epochs (50 passes through the data)
```

---

## Visual Flow Diagram

```
CSV File (1 stock: AAPL.csv)
  ↓ load_stock_data()
OHLC Data (1,200 rows × 6 cols: date, open, high, low, close, volume)
  ↓ create_features()
Feature Data (1,200 rows × 43 cols: OHLC + 37 technical features)
  ↓ prepare_training_data()
Training Arrays: X (1,199 × 37), y (1,199)
  ↓ ×100 stocks
Stock Batch: X_batch (119,900 × 37), y_batch (119,900)
  ↓ ×11 batches
Full Dataset: X_full (1,201,200 × 37), y_full (1,201,200)
  ↓ model.fit() or partial_fit()
Trained Model (.pkl or .h5 files)
```

---

## Numbers Recap

| Level | Stocks | Rows/Stock | Features | Total Rows | Memory |
|-------|--------|------------|----------|------------|--------|
| 1 Stock | 1 | 1,199 | 37 | 1,199 | ~0.73 MB |
| 1 Batch | 100 | 1,199 | 37 | 119,900 | ~73 MB |
| Full Dataset | 1,001 | 1,199 | 37 | 1,201,200 | ~730 MB |

**Why Batching?**
- **Without batching:** Load all 1,001 stocks at once = 730 MB → 🔥 Memory overflow on most machines
- **With batching:** Load 100 stocks at a time = 73 MB → ✅ Manageable, fits in RAM comfortably

---

## Batch Loading System

### Why Batching is Needed

The full dataset is massive:
- 1,001 stocks × 1,200 rows × 37 features = 44.4 million numbers
- At 8 bytes per float64: 355 MB just for features
- Add targets, overhead, intermediate arrays: **~730 MB minimum**
- Many models need 2-3x more memory during training → **2-3 GB just for data**

**Solution:** Process stocks in smaller batches

### Three Levels of Batching

#### 1. Stock-Level Batching (Primary Approach)

```python
# Split 1,001 stocks into batches of 100
batch_1: stocks 1-100
batch_2: stocks 101-200
...
batch_11: stocks 1001-1001
```

**Memory per batch:** 100 stocks × 1,199 rows × 37 features × 8 bytes = **~73 MB**

**Process:**
1. Load 100 CSV files
2. Create features for each
3. Combine into single X_batch, y_batch
4. Train/accumulate based on strategy
5. Clear memory, move to next batch

#### 2. Row-Level Sub-Batching (For Large Batches)

If a stock batch has too many rows (e.g., stocks with more data points), split further:

```python
if len(X_batch) > 50000:  # Too many rows
    # Split into sub-batches of 50K rows
    sub_batch_1 = X_batch[0:50000]
    sub_batch_2 = X_batch[50000:100000]
    # Train on each sub-batch
```

**Default:** 50,000 rows per sub-batch

#### 3. Mini-Batch for Neural Networks

Neural networks (ANN, CNN, Autoencoder) use Keras' built-in mini-batching:

```python
# Even within a stock batch, Keras splits further
model.fit(X_batch, y_batch, batch_size=32)
# Keras processes 32 samples at a time internally
```

**Mini-batch sizes:**
- **ANN/Autoencoder:** 32 samples
- **CNN:** 8 samples (needs more memory per sample due to sequences)

---

## Training Strategies by Model

| Model | Strategy | How It Works |
|-------|----------|--------------|
| Linear Regression | Incremental | `partial_fit()` on each batch, updates weights incrementally |
| Decision Tree | Accumulate | Collect all batches → train once on full dataset |
| Random Forest | Accumulate | Collect all batches → train once on full dataset |
| SVM | Subsample | Train on 50% random sample to handle large dataset |
| KNN | Subsample | Train on 50% random sample (KNN stores training data) |
| ANN | Keras Batch | Mini-batch training (32 samples), 50 epochs |
| CNN | Keras Batch | Mini-batch training (8 samples), 100 epochs |
| ARIMA | Subsample | Train on sample with timeout (very slow otherwise) |
| Autoencoder | Keras Batch | Mini-batch training (32 samples), 100 epochs |

**Default Settings:**
- Stock batch size: 100 stocks/batch
- Row sub-batch size: 50,000 rows/sub-batch
- Subsample percent: 50%

---

## Common Training Steps (All Models)

Every model training follows these steps:

### 1. Initialize
```python
trainer = EnhancedModelTrainer(model_name='linear_regression')
model = trainer.initialize_model()
data_loader = DataLoader()
```

### 2. Get Stock Symbols
```python
# Load from index files
us_stocks = get_symbols('us_stocks')  # 501 stocks
ind_stocks = get_symbols('ind_stocks')  # 500 stocks
all_symbols = us_stocks + ind_stocks  # 1,001 total
```

### 3. Create Batches
```python
# Split into batches of 100 stocks
batches = []
for i in range(0, len(all_symbols), 100):
    batch = all_symbols[i:i+100]
    batches.append(batch)
# Result: 11 batches (10 × 100 + 1 × 1)
```

### 4. Process Each Batch
```python
for batch_num, stock_batch in enumerate(batches):
    print(f"Processing batch {batch_num+1}/11...")
    
    # 4a. Load data for all stocks in batch
    all_X = []
    all_y = []
    for stock_symbol in stock_batch:
        # Load CSV
        df = data_loader.load_stock_data(stock_symbol, category)
        
        # Validate (skip if bad data)
        if df is None or len(df) < 252:  # Need 1 year minimum
            continue
        
        # Create features
        df_features = data_loader.create_features(df)
        
        # Prepare arrays
        X, y = data_loader.prepare_training_data(df_features)
        
        # Add to batch
        all_X.append(X)
        all_y.append(y)
    
    # 4b. Combine into batch
    X_batch = np.vstack(all_X)  # Stack vertically
    y_batch = np.concatenate(all_y)  # Concatenate
    
    # 4c. Train based on strategy
    if strategy == 'incremental':
        model.partial_fit(X_batch, y_batch)
    elif strategy == 'accumulate':
        accumulated_data.append((X_batch, y_batch))
    elif strategy == 'subsample':
        accumulated_data.append((X_batch, y_batch))
```

### 5. Final Training (for Accumulate/Subsample)
```python
if strategy in ['accumulate', 'subsample']:
    # Combine all batches
    X_all = np.vstack([x for x, y in accumulated_data])
    y_all = np.concatenate([y for x, y in accumulated_data])
    
    # Subsample if needed
    if strategy == 'subsample':
        sample_size = len(X_all) // 2
        indices = np.random.choice(len(X_all), sample_size, replace=False)
        X_all = X_all[indices]
        y_all = y_all[indices]
    
    # Train once
    model.fit(X_all, y_all)
```

### 6. Save Model
```python
model_dir = f'backend/models/{model_name}/'
os.makedirs(model_dir, exist_ok=True)

# Save model file
joblib.dump(model, f'{model_dir}/{model_name}_model.pkl')

# For neural networks, also save weights
if model_type in ['ann', 'cnn', 'autoencoder']:
    model.keras_model.save(f'{model_dir}/{model_name}_model.h5')
```

### 7. Update Status
```python
status = {
    'model': model_name,
    'trained': True,
    'last_updated': datetime.now().isoformat(),
    'r2_score': r2_score,
    'data_points': len(y_all)
}
with open('backend/models/model_status.json', 'w') as f:
    json.dump(status, f)
```

---

## Model-Specific Training Details

### Linear Regression
**Strategy:** Incremental  
**Details:**
- Uses Stochastic Gradient Descent (SGD) for incremental learning
- Calls `partial_fit()` on each batch
- Model weights update after each batch
- Very fast: 5-10 minutes total
- Model size: ~3 MB

```python
from sklearn.linear_model import SGDRegressor
model = SGDRegressor(max_iter=1000, tol=1e-3)
for X_batch, y_batch in batches:
    model.partial_fit(X_batch, y_batch)
```

---

### Decision Tree
**Strategy:** Accumulate  
**Details:**
- Collects all batches into memory
- Trains once on full dataset
- No incremental learning support
- Fast: 5-10 minutes
- Model size: ~150 MB (stores tree structure)

```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=20, min_samples_split=10)
# Collect all batches
X_all, y_all = accumulate_all_batches()
# Train once
model.fit(X_all, y_all)
```

---

### Random Forest
**Strategy:** Accumulate  
**Details:**
- Ensemble of 100 decision trees
- Collects all batches, trains once
- Most accurate but largest model
- Training time: 10-15 minutes
- Model size: ~9.5 GB (100 trees × 95 MB each)

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1)
X_all, y_all = accumulate_all_batches()
model.fit(X_all, y_all)
```

---

### SVM (Support Vector Machine)
**Strategy:** Subsample  
**Details:**
- Cannot handle 1.2M data points efficiently
- Trains on 50% random sample (600K points)
- Uses RBF kernel for non-linear patterns
- Training time: 20-30 minutes
- Model size: ~10 MB

```python
from sklearn.svm import SVR
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
X_all, y_all = accumulate_all_batches()
# Subsample 50%
X_sample = X_all[::2]
y_sample = y_all[::2]
model.fit(X_sample, y_sample)
```

---

### KNN (K-Nearest Neighbors)
**Strategy:** Subsample  
**Details:**
- Stores training data (no actual "training")
- k=15 neighbors for predictions
- Subsamples to 50% to reduce storage/query time
- Training time: 15-25 minutes
- Model size: ~10 MB

```python
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=15, weights='distance')
X_all, y_all = accumulate_all_batches()
X_sample = X_all[::2]
y_sample = y_all[::2]
model.fit(X_sample, y_sample)
```

---

### ANN (Artificial Neural Network)
**Strategy:** Keras Batch  
**Details:**
- 3 hidden layers: [128, 64, 32] neurons
- Mini-batch size: 32 samples
- Epochs: 50
- Uses gradient clipping to prevent explosion
- Training time: 30-45 minutes
- Model size: ~256 KB

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(37,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output: predicted price
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, batch_size=32, epochs=50, 
          validation_split=0.2, verbose=1)
```

---

### CNN (Convolutional Neural Network)
**Strategy:** Keras Batch  
**Details:**
- 1D convolutions over time sequences
- Sequence length: 20 days
- Filters: [32, 16] with kernel size 3
- Mini-batch size: 8 (larger memory per sample)
- Epochs: 100
- Training time: 45-75 minutes
- Model size: ~5 MB

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

# Create sequences of 20 days
X_sequences = create_sequences(X, seq_length=20)

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(20, 37)),
    MaxPooling1D(pool_size=2),
    Conv1D(16, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_sequences, y, batch_size=8, epochs=100)
```

---

### ARIMA (AutoRegressive Integrated Moving Average)
**Strategy:** Subsample  
**Details:**
- Traditional time series model
- Fits per-stock ARIMA, then averages
- Order: (5,1,0) - 5 lags, 1 differencing, 0 MA terms
- Uses timeout to prevent hanging
- Training time: 90-180 minutes (slowest)
- Model size: ~170 MB

```python
from statsmodels.tsa.arima.model import ARIMA

# For each stock (subsampled)
for stock_data in sampled_stocks:
    model = ARIMA(stock_data, order=(5, 1, 0))
    fitted = model.fit()
    # Store fitted model
```

---

### Autoencoder
**Strategy:** Keras Batch  
**Details:**
- Encoder-decoder architecture
- Encoder: [64, 32] → latent space (16 dims)
- Decoder: [32, 64, 37] → reconstruction
- Uses encoded features for regression
- Mini-batch size: 32
- Epochs: 100
- Training time: 40-60 minutes
- Model size: ~1 MB (3 files: autoencoder.h5, encoder.h5, metadata.pkl)

```python
# Encoder
encoder = Sequential([
    Dense(64, activation='relu', input_shape=(37,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu')  # Latent space
])

# Decoder
decoder = Sequential([
    Dense(32, activation='relu', input_shape=(16,)),
    Dense(64, activation='relu'),
    Dense(37, activation='linear')  # Reconstruct input
])

# Full autoencoder
autoencoder = Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, batch_size=32, epochs=100)

# Use encoded features for prediction
X_encoded = encoder.predict(X_train)
regressor = RandomForestRegressor()
regressor.fit(X_encoded, y_train)
```

---

## Confidence Calculation Method

The system calculates confidence scores (0-100%) for predictions based on multiple factors.

### For Ensemble Predictions (Multiple Models)

When using multiple models together:

```python
confidence = (
    40% × Model_Accuracy_Score +
    20% × Prediction_Variance_Score +
    15% × Stock_Volatility_Score +
    25% × Time_Decay_Score
)
```

#### Factor 1: Model Accuracy (40% weight)
```python
# Average R² scores with complexity multipliers
accuracy_scores = []
for model_name, r2_score in model_accuracies.items():
    multiplier = MODEL_COMPLEXITY_MULTIPLIERS[model_name]
    accuracy_scores.append(r2_score * multiplier * 100)

accuracy_score = mean(accuracy_scores)
```

**Model Complexity Multipliers:**
- **ANN, CNN, Autoencoder, ARIMA:** 1.05× (complex models, deep analysis)
- **Random Forest:** 1.03× (ensemble model)
- **Linear Regression, Decision Tree, KNN, SVM:** 1.0× (simpler models)

**Why multipliers?** Complex models can capture non-linear patterns better, so even with same R² score, they deserve slightly higher confidence.

#### Factor 2: Prediction Variance (20% weight)
```python
# How much do models agree?
predictions = [model1_pred, model2_pred, model3_pred, ...]
std_dev = std(predictions)
mean_pred = mean(predictions)
coefficient_of_variation = std_dev / mean_pred

# Lower variation = higher confidence
variance_score = 100 * (1 - min(coefficient_of_variation, 1.0))
```

**Example:**
- If all models predict $100 ± $2 → CV = 0.02 → Score = 98
- If models predict $100 ± $20 → CV = 0.20 → Score = 80

#### Factor 3: Stock Volatility (15% weight)
```python
# How stable is the stock historically?
recent_prices = last_30_days_prices
volatility = std(recent_prices) / mean(recent_prices)

# Lower volatility = higher confidence
volatility_score = 100 * (1 - min(volatility * 2, 1.0))
```

**Example:**
- Stable stock (volatility = 0.02) → Score = 96
- Volatile stock (volatility = 0.50) → Score = 0

#### Factor 4: Time Decay (25% weight)
```python
# Longer prediction horizon = lower confidence
days_ahead = time_horizon_days
decay_factor = 1.0 / (1.0 + days_ahead / 365)
time_decay_score = decay_factor * 100
```

**Example:**
- 1 day ahead → decay = 1.0 / 1.003 = 0.997 → Score = 99.7
- 30 days ahead → decay = 1.0 / 1.082 = 0.924 → Score = 92.4
- 365 days ahead → decay = 1.0 / 2.0 = 0.50 → Score = 50.0
- 1825 days (5Y) → decay = 1.0 / 6.0 = 0.167 → Score = 16.7

---

### For Single Model Predictions

When using just one model:

```python
confidence = (
    50% × Model_Accuracy_Score +
    25% × Stock_Volatility_Score +
    25% × Time_Decay_Score
)
```

**Differences from ensemble:**
- No variance score (only 1 model, no agreement to measure)
- Accuracy weighted higher (50% vs 40%)
- Volatility and time decay split the remaining 50%

---

### Example Confidence Calculation

**Scenario:** Predicting AAPL price 30 days ahead with ensemble

**Inputs:**
- Model R² scores: ANN=0.85, Random Forest=0.92, Linear Reg=0.87
- Model predictions: $175, $178, $176 (mean=$176.33, std=$1.53)
- Historical volatility: 0.03 (3%)
- Time horizon: 30 days

**Calculation:**

```python
# 1. Model Accuracy (40%)
ann_score = 0.85 * 1.05 * 100 = 89.25
rf_score = 0.92 * 1.03 * 100 = 94.76
lr_score = 0.87 * 1.0 * 100 = 87.0
accuracy_score = (89.25 + 94.76 + 87.0) / 3 = 90.34
weighted_accuracy = 0.40 * 90.34 = 36.14

# 2. Prediction Variance (20%)
cv = 1.53 / 176.33 = 0.0087
variance_score = 100 * (1 - 0.0087) = 99.13
weighted_variance = 0.20 * 99.13 = 19.83

# 3. Stock Volatility (15%)
volatility_score = 100 * (1 - 0.03 * 2) = 94.0
weighted_volatility = 0.15 * 94.0 = 14.10

# 4. Time Decay (25%)
decay = 1.0 / (1.0 + 30/365) = 0.924
time_decay_score = 0.924 * 100 = 92.4
weighted_time = 0.25 * 92.4 = 23.10

# Final Confidence
confidence = 36.14 + 19.83 + 14.10 + 23.10 = 93.17%
```

**Result:** 93.2% confidence (very high due to good model accuracy, agreement, low volatility, short horizon)

---

## Training Commands

### Basic Training
```bash
# Train a specific model
python backend/training/train_full_dataset.py --model linear_regression
python backend/training/train_full_dataset.py --model random_forest
python backend/training/train_full_dataset.py --model ann

# Available models: linear_regression, decision_tree, random_forest, 
#                   svm, knn, ann, cnn, arima, autoencoder
```

### Force Retrain
```bash
# Retrain even if model already exists
python backend/training/train_full_dataset.py --model knn --force-retrain
```

### Custom Batch Size
```bash
# Use smaller batches (50 stocks instead of 100)
python backend/training/train_full_dataset.py --model svm --stock-batch-size 50

# Use different subsample percentage
python backend/training/train_full_dataset.py --model svm --subsample-percent 30
```

### Check Training Status
```bash
# See which models are trained
python status.py
```

### Train All Models
```bash
# Train all 9 models sequentially (4-6 hours)
python backend/training/train_full_dataset.py --model linear_regression
python backend/training/train_full_dataset.py --model decision_tree
python backend/training/train_full_dataset.py --model random_forest
python backend/training/train_full_dataset.py --model knn
python backend/training/train_full_dataset.py --model svm
python backend/training/train_full_dataset.py --model ann
python backend/training/train_full_dataset.py --model autoencoder
python backend/training/train_full_dataset.py --model cnn
python backend/training/train_full_dataset.py --model arima
```

---

## Key File Locations

### Training Data
```
data/past/
├── us_stocks/individual_files/
│   ├── AAPL.csv (501 files)
│   ├── GOOGL.csv
│   └── ...
└── ind_stocks/individual_files/
    ├── RELIANCE.csv (500 files)
    ├── TCS.csv
    └── ...
```

### Trained Models
```
backend/models/
├── linear_regression/
│   └── linear_regression_model.pkl
├── random_forest/
│   └── random_forest_model.pkl
├── ann/
│   ├── ann_model.pkl
│   └── ann_model_model.h5
├── cnn/
│   ├── cnn_model.pkl
│   └── cnn_model_main.h5
├── autoencoder/
│   ├── autoencoder_model.pkl_autoencoder.h5
│   ├── autoencoder_model.pkl_encoder.h5
│   └── autoencoder_model.pkl_metadata.pkl
└── model_status.json
```

### Training Code
- **Main Trainer:** `backend/training/enhanced_model_trainer.py`
- **Batch Iterator:** `backend/training/batch_iterator.py`
- **Training Strategies:** `backend/training/batch_strategies.py`
- **Training Script:** `backend/training/train_full_dataset.py`
- **Data Loader:** `backend/prediction/data_loader.py`
- **Confidence Calculator:** `backend/prediction/confidence_calculator.py`

### Model Implementations
```
backend/algorithms/optimised/
├── linear_regression/linear_regression.py
├── decision_tree/decision_tree.py
├── random_forest/random_forest.py
├── svm/svm.py
├── knn/knn.py
├── ann/ann.py
├── cnn/cnn.py
├── arima/arima.py
└── autoencoders/autoencoder.py
```

---

## Memory and Performance

### Dataset Size
- **Stocks:** 1,001 (501 US + 500 Indian)
- **Rows per stock:** ~1,200
- **Total rows:** 1,201,200
- **Features:** 37
- **Total data points:** 44,444,400

### Memory Requirements
- **Per stock:** 1,199 rows × 37 features × 8 bytes = **0.73 MB**
- **Per batch:** 100 stocks × 0.73 MB = **73 MB**
- **Full dataset:** 1,001 stocks × 0.73 MB = **730 MB** (minimum)
- **With overhead (2x):** ~1.5 GB
- **Peak during training:** 2-8 GB depending on model

### Training Time
| Model | Time | Reason |
|-------|------|--------|
| Linear Regression | 5-10 min | Incremental, simple math |
| Decision Tree | 5-10 min | Single tree, fast |
| Random Forest | 10-15 min | 100 trees, parallelized |
| KNN | 15-25 min | Subsample + distance calculations |
| SVM | 20-30 min | Subsample + kernel computations |
| ANN | 30-45 min | 50 epochs × forward/backward pass |
| Autoencoder | 40-60 min | 100 epochs, larger architecture |
| CNN | 45-75 min | 100 epochs, sequences, convolutions |
| ARIMA | 90-180 min | Per-stock fitting, iterative |

**Total time for all models:** 4-6 hours sequential

### Hardware Recommendations
- **Minimum:** 8 GB RAM, 12 GB disk, 4 cores
- **Recommended:** 16 GB RAM, 20 GB disk, 8 cores, SSD
- **Optimal:** 32 GB RAM, 50 GB disk, 16 cores, NVMe SSD

### Model File Sizes
| Model | Size | Storage |
|-------|------|---------|
| Linear Regression | 3 MB | Model coefficients |
| Decision Tree | 150 MB | Tree structure |
| Random Forest | 9.5 GB | 100 trees |
| SVM | 10 MB | Support vectors |
| KNN | 10 MB | Training data subset |
| ANN | 256 KB | Neural network weights |
| CNN | 5 MB | Convolutional weights |
| ARIMA | 170 MB | Per-stock models |
| Autoencoder | 1 MB | Encoder + decoder + metadata |

**Total:** ~10 GB for all models

---

## Summary

**Training Pipeline:**
1. Load 1,001 CSV files with 5 years of OHLC data
2. Calculate 37 technical features per stock
3. Prepare X (features) and y (next day price) arrays
4. Process in batches of 100 stocks (73 MB each)
5. Train using strategy appropriate to each model
6. Save trained models to `backend/models/`
7. Use models for predictions with confidence scores

**Key Insights:**
- **Batching prevents memory overflow** (730 MB → 73 MB chunks)
- **Different strategies optimize for different model types** (incremental vs accumulate vs subsample)
- **37 features capture price patterns** better than raw OHLC alone
- **Confidence depends on 4 factors:** model accuracy, prediction agreement, stock volatility, time horizon
- **Complex models get confidence boost** via multipliers (ANN/CNN/Autoencoder: 1.05×)
- **Training takes 4-6 hours** for all 9 models

**Next Steps:**
- See `documentation/MODEL_TRAINING.md` for quick reference
- Run `python status.py` to check trained models
- Use `python backend/training/train_full_dataset.py --model MODEL_NAME` to train
- Models are automatically used by prediction system once trained

