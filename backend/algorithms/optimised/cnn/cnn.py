"""
1D Convolutional Neural Network for Stock Price Prediction

Optimized implementation using TensorFlow/Keras for stock price prediction
based on OHLCV data and technical indicators. Uses 1D convolutions for time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model_interface import ModelInterface
from stock_indicators import StockIndicators


class CNNModel(ModelInterface):
    """
    1D Convolutional Neural Network model for stock price prediction.
    
    Uses technical indicators calculated from OHLC data to predict
    future stock prices. Volume is excluded from all calculations.
    """
    
    def __init__(self, sequence_length: int = 60, filters: List[int] = [64, 32, 16],
                 kernel_size: int = 3, dropout_rate: float = 0.2,
                 learning_rate: float = 0.001, batch_size: int = 32, 
                 epochs: int = 100, use_lstm: bool = False, **kwargs):
        super().__init__('1D Convolutional Neural Network', **kwargs)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.sequence_length = sequence_length
        
        # Model parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_lstm = use_lstm
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build the 1D CNN architecture.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Input layer
        model.add(Conv1D(self.filters[0], self.kernel_size, activation='relu', 
                        input_shape=input_shape,
                        kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(self.dropout_rate))
        
        # Additional Conv1D layers
        for filters in self.filters[1:]:
            model.add(Conv1D(filters, self.kernel_size, activation='relu',
                           kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))
            model.add(Dropout(self.dropout_rate))
        
        # Optional LSTM layer
        if self.use_lstm:
            model.add(Bidirectional(LSTM(32, return_sequences=True)))
            model.add(Bidirectional(LSTM(16)))
        else:
            model.add(GlobalAveragePooling1D())
        
        # Dense layers
        model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the CNN model on stock data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - stock prices
            
        Returns:
            self: Returns self for method chaining
        """
        self.validate_input(X, y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {self.sequence_length} samples.")
        
        # Build model
        self.model = self._build_model((self.sequence_length, X_scaled.shape[1]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Calculate training metrics
        y_pred = self.model.predict(X_seq, verbose=0).flatten()
        mse = mean_squared_error(y_seq, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_seq, y_pred)
        mae = mean_absolute_error(y_seq, y_pred)
        
        self.set_training_metrics({
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'training_loss': history.history['loss'][-1],
            'validation_loss': history.history['val_loss'][-1]
        })
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new stock data.
        
        Args:
            X: Features to predict on (n_samples, n_features)
            
        Returns:
            predictions: Predicted stock prices (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.validate_input(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences for prediction
        if len(X_scaled) < self.sequence_length:
            # Pad with zeros if not enough data
            padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_scaled = np.vstack([padding, X_scaled])
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        if len(X_seq) == 0:
            # If no sequences can be created, use the last sequence_length points
            X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        
        return predictions
    
    def predict_sequence(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Make multi-step predictions.
        
        Args:
            X: Features to predict on
            steps: Number of steps to predict ahead
            
        Returns:
            predictions: Predicted stock prices for each step
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        predictions = []
        current_X = X.copy()
        
        for _ in range(steps):
            pred = self.predict(current_X[-1:])  # Predict next step
            predictions.append(pred[0])
            
            # Update features for next prediction (simplified)
            # In practice, you'd need to update technical indicators
            current_X = np.vstack([current_X, current_X[-1:]])
        
        return np.array(predictions)
    
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: File path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Save Keras model
        model_path = path.replace('.pkl', '_model.h5')
        self.model.save(model_path)
        
        # Save other components
        joblib.dump({
            'scaler': self.scaler,
            'metrics': self.training_metrics,
            'params': self.model_params,
            'feature_columns': self.feature_columns,
            'model_path': model_path,
            'sequence_length': self.sequence_length
        }, path)
    
    def load(self, path: str) -> 'ModelInterface':
        """
        Load a previously saved model from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            self: Returns self for method chaining
        """
        from tensorflow.keras.models import load_model
        
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.training_metrics = data['metrics']
        self.model_params = data['params']
        self.feature_columns = data.get('feature_columns')
        self.sequence_length = data.get('sequence_length', 60)
        
        # Load Keras model
        model_path = data.get('model_path', path.replace('.pkl', '_model.h5'))
        self.model = load_model(model_path)
        
        self.is_trained = True
        return self


# Example usage and testing
if __name__ == "__main__":
    # Create sample stock data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Generate synthetic OHLC data
    base_price = 100
    returns = np.random.normal(0, 0.02, 200)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices
    })
    
    # Ensure high >= low and high >= close >= low
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    # Create model
    model = CNNModel(sequence_length=30, filters=[32, 16], epochs=50)
    
    # Add technical indicators
    df_with_features = model._create_technical_indicators(df)
    
    # Prepare training data
    X, y = StockIndicators.prepare_training_data(df_with_features)
    
    if len(X) > 0:
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X[-10:])  # Predict last 10 days
        
        print(f"1D CNN Model Results:")
        print(f"Training R²: {model.training_metrics['r2_score']:.4f}")
        print(f"Training RMSE: {model.training_metrics['rmse']:.4f}")
        print(f"Sample predictions: {predictions[:5]}")
        
        # Test save/load
        model.save('test_cnn_model.pkl')
        loaded_model = CNNModel().load('test_cnn_model.pkl')
        print(f"Model loaded successfully: {loaded_model.is_trained}")
        
        # Clean up
        os.remove('test_cnn_model.pkl')
        os.remove('test_cnn_model_model.h5')
    else:
        print("Insufficient data for training")
