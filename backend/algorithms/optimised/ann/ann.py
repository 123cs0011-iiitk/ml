"""
Artificial Neural Network for Stock Price Prediction

Optimized implementation using TensorFlow/Keras for stock price prediction
based on OHLCV data and technical indicators.
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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model_interface import ModelInterface
from stock_indicators import StockIndicators


class ANNModel(ModelInterface):
    """
    Artificial Neural Network model for stock price prediction.
    
    Uses technical indicators calculated from OHLC data to predict
    future stock prices. Volume is excluded from all calculations.
    """
    
    def __init__(self, hidden_layers: List[int] = [64, 32, 16], 
                 activation: str = 'relu', dropout_rate: float = 0.2,
                 learning_rate: float = 0.001, optimizer: str = 'adam',
                 batch_size: int = 32, epochs: int = 100, **kwargs):
        super().__init__('Artificial Neural Network', **kwargs)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Model parameters
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _build_model(self, input_dim: int) -> Sequential:
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], activation=self.activation, 
                       input_shape=(input_dim,),
                       kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation=self.activation,
                           kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        if self.optimizer == 'adam':
            opt = Adam(learning_rate=self.learning_rate)
        else:
            opt = SGD(learning_rate=self.learning_rate, momentum=0.9)
        
        model.compile(
            optimizer=opt,
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the ANN model on stock data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) - stock prices
            
        Returns:
            self: Returns self for method chaining
        """
        self.validate_input(X, y)
        
        # Scale features (important for neural networks)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.model = self._build_model(X_scaled.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train model
        history = self.model.fit(
            X_scaled, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled, verbose=0).flatten()
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
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
        
        # Make predictions
        predictions = self.model.predict(X_scaled, verbose=0).flatten()
        
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation using dropout.
        
        Args:
            X: Features to predict on
            n_samples: Number of samples for Monte Carlo dropout
            
        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.validate_input(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Enable dropout for uncertainty estimation
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X_scaled, training=True).numpy().flatten()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_predictions, uncertainty
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            String representation of model architecture
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            self.model.summary()
        return f.getvalue()
    
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
            'model_path': model_path
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
    model = ANNModel(hidden_layers=[32, 16], dropout_rate=0.2, epochs=50)
    
    # Add technical indicators
    df_with_features = model._create_technical_indicators(df)
    
    # Prepare training data
    X, y = StockIndicators.prepare_training_data(df_with_features)
    
    if len(X) > 0:
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X[-10:])  # Predict last 10 days
        
        print(f"ANN Model Results:")
        print(f"Training R²: {model.training_metrics['r2_score']:.4f}")
        print(f"Training RMSE: {model.training_metrics['rmse']:.4f}")
        print(f"Sample predictions: {predictions[:5]}")
        
        # Test save/load
        model.save('test_ann_model.pkl')
        loaded_model = ANNModel().load('test_ann_model.pkl')
        print(f"Model loaded successfully: {loaded_model.is_trained}")
        
        # Clean up
        os.remove('test_ann_model.pkl')
        os.remove('test_ann_model_model.h5')
    else:
        print("Insufficient data for training")
