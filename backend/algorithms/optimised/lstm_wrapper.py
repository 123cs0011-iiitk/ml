"""
LSTM Wrapper for Stock Price Prediction

This module implements a production-ready LSTM model for time series prediction
using TensorFlow/Keras with proper scaling and early stopping.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..model_interface import ModelInterface
from ..utils import calculate_metrics

logger = logging.getLogger(__name__)


class LSTMWrapper(ModelInterface):
    """
    LSTM model wrapper for stock price prediction.
    
    Implements a 3-layer LSTM architecture with dropout and early stopping.
    Uses MinMaxScaler for data normalization.
    """
    
    def __init__(self, lookback: int = 60, lstm_units: Tuple[int, int, int] = (128, 64, 32),
                 dropout_rate: float = 0.2, learning_rate: float = 0.001, 
                 patience: int = 20, **kwargs):
        """
        Initialize LSTM model.
        
        Args:
            lookback: Number of time steps to look back
            lstm_units: Number of units in each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            **kwargs: Additional parameters
        """
        super().__init__("LSTM", **kwargs)
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.patience = patience
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_fitted = False
        
    def _build_model(self, n_features: int) -> Sequential:
        """
        Build the LSTM model architecture.
        
        Args:
            n_features: Number of input features
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=(self.lookback, n_features)
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(
            units=self.lstm_units[1],
            return_sequences=True
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Third LSTM layer
        model.add(LSTM(
            units=self.lstm_units[2],
            return_sequences=False
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSTMWrapper':
        """
        Train the LSTM model.
        
        Args:
            X: Training features (n_samples, lookback, n_features)
            y: Training targets (n_samples,)
            
        Returns:
            self
        """
        self.validate_input(X, y)
        
        logger.info(f"Training LSTM model with {X.shape[0]} samples")
        
        # Scale features
        n_samples, lookback, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.feature_scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, lookback, n_features)
        
        # Scale targets
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Build model
        self.model = self._build_model(n_features)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate metrics
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        
        metrics = calculate_metrics(y_true, y_pred)
        self.set_training_metrics(metrics)
        self.is_fitted = True
        
        logger.info(f"LSTM training completed. RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on (n_samples, lookback, n_features)
            
        Returns:
            predictions: Predicted values (n_samples,)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.validate_input(X)
        
        # Scale features
        n_samples, lookback, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.feature_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, lookback, n_features)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def predict_sequence(self, X: np.ndarray, steps: int) -> np.ndarray:
        """
        Predict multiple steps ahead using iterative forecasting.
        
        Args:
            X: Initial sequence (1, lookback, n_features)
            steps: Number of steps to predict ahead
            
        Returns:
            predictions: Predicted sequence (steps,)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        current_sequence = X.copy()
        
        for _ in range(steps):
            # Make prediction for next step
            pred_scaled = self.model.predict(current_sequence, verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            # Shift sequence and add prediction as new feature
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred_scaled[0, 0]  # Use scaled prediction for consistency
        
        return np.array(predictions)
    
    def save(self, path: str) -> None:
        """
        Save the trained model and scalers.
        
        Args:
            path: Directory path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, 'lstm_model.h5')
        self.model.save(model_path)
        
        # Save scalers
        import joblib
        scaler_path = os.path.join(path, 'target_scaler.joblib')
        feature_scaler_path = os.path.join(path, 'feature_scaler.joblib')
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_scaler, feature_scaler_path)
        
        # Save metadata
        metadata = {
            'lookback': self.lookback,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"LSTM model saved to {path}")
    
    def load(self, path: str) -> 'LSTMWrapper':
        """
        Load a previously saved model.
        
        Args:
            path: Directory path to load the model from
            
        Returns:
            self
        """
        # Load model
        model_path = os.path.join(path, 'lstm_model.h5')
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scalers
        import joblib
        scaler_path = os.path.join(path, 'target_scaler.joblib')
        feature_scaler_path = os.path.join(path, 'feature_scaler.joblib')
        
        self.scaler = joblib.load(scaler_path)
        self.feature_scaler = joblib.load(feature_scaler_path)
        
        # Load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.lookback = metadata.get('lookback', self.lookback)
            self.lstm_units = tuple(metadata.get('lstm_units', self.lstm_units))
            self.dropout_rate = metadata.get('dropout_rate', self.dropout_rate)
            self.learning_rate = metadata.get('learning_rate', self.learning_rate)
            self.patience = metadata.get('patience', self.patience)
            self.training_metrics = metadata.get('training_metrics', {})
            self.is_fitted = True
        
        logger.info(f"LSTM model loaded from {path}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            'lookback': self.lookback,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'is_fitted': self.is_fitted
        })
        return info
