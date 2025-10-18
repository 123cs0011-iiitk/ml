"""
Support Vector Regression (SVR) Wrapper for Stock Price Prediction

This module implements a production-ready SVR model with RBF kernel
and hyperparameter tuning using GridSearchCV.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from ..model_interface import ModelInterface
from ..utils import calculate_metrics

logger = logging.getLogger(__name__)


class SVRWrapper(ModelInterface):
    """
    Support Vector Regression model wrapper for stock price prediction.
    
    Uses RBF kernel with hyperparameter tuning via GridSearchCV.
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C_values: List[float] = [0.1, 1, 10, 100],
                 gamma_values: List[float] = ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                 epsilon_values: List[float] = [0.01, 0.1, 0.2, 0.5],
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize SVR model.
        
        Args:
            kernel: Kernel type for SVR
            C_values: List of C parameter values to try
            gamma_values: List of gamma parameter values to try
            epsilon_values: List of epsilon parameter values to try
            random_state: Random state for reproducibility
            n_jobs: Number of jobs to run in parallel
            **kwargs: Additional parameters
        """
        super().__init__("SVR", **kwargs)
        self.kernel = kernel
        self.C_values = C_values
        self.gamma_values = gamma_values
        self.epsilon_values = epsilon_values
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.scaler = StandardScaler()
        self.best_params_ = None
        self.is_fitted = False
        
    def _prepare_data_for_svr(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare data for SVR (flatten time series).
        
        Args:
            X: Input data (n_samples, lookback, n_features)
            
        Returns:
            Flattened data (n_samples, lookback * n_features)
        """
        if X.ndim == 3:
            # Flatten time series for SVR
            n_samples, lookback, n_features = X.shape
            return X.reshape(n_samples, lookback * n_features)
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRWrapper':
        """
        Train the SVR model with GridSearchCV.
        
        Args:
            X: Training features (n_samples, lookback, n_features) or (n_samples, n_features)
            y: Training targets (n_samples,)
            
        Returns:
            self
        """
        self.validate_input(X, y)
        
        logger.info(f"Training SVR model with {X.shape[0]} samples")
        
        # Prepare data for SVR
        X_flat = self._prepare_data_for_svr(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Define parameter grid
        param_grid = {
            'C': self.C_values,
            'gamma': self.gamma_values,
            'epsilon': self.epsilon_values
        }
        
        # Create base model
        base_model = SVR(kernel=self.kernel)
        
        # Use TimeSeriesSplit for time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # Fit the model
        grid_search.fit(X_scaled, y)
        
        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        metrics = calculate_metrics(y, y_pred)
        self.set_training_metrics(metrics)
        self.is_fitted = True
        
        logger.info(f"SVR training completed. Best params: {self.best_params_}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on (n_samples, lookback, n_features) or (n_samples, n_features)
            
        Returns:
            predictions: Predicted values (n_samples,)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.validate_input(X)
        
        # Prepare data for SVR
        X_flat = self._prepare_data_for_svr(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_flat)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        return y_pred
    
    def save(self, path: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            path: Directory path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, 'svr_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(path, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'kernel': self.kernel,
            'best_params': self.best_params_,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"SVR model saved to {path}")
    
    def load(self, path: str) -> 'SVRWrapper':
        """
        Load a previously saved model.
        
        Args:
            path: Directory path to load the model from
            
        Returns:
            self
        """
        # Load model
        model_path = os.path.join(path, 'svr_model.joblib')
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(path, 'scaler.joblib')
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.kernel = metadata.get('kernel', self.kernel)
            self.best_params_ = metadata.get('best_params', {})
            self.training_metrics = metadata.get('training_metrics', {})
            self.is_fitted = True
        
        logger.info(f"SVR model loaded from {path}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            'kernel': self.kernel,
            'best_params': self.best_params_,
            'is_fitted': self.is_fitted
        })
        return info
