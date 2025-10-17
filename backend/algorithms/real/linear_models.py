"""
Linear Models Wrapper for Stock Price Prediction

This module implements production-ready linear models including
Ridge and Lasso regression with hyperparameter tuning.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from ..model_interface import ModelInterface
from ..utils import calculate_metrics

logger = logging.getLogger(__name__)


class LinearModelsWrapper(ModelInterface):
    """
    Linear models wrapper for stock price prediction.
    
    Supports Ridge, Lasso, ElasticNet, and Linear Regression.
    """
    
    def __init__(self, 
                 model_type: str = 'ridge',
                 alpha_values: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0],
                 l1_ratio_values: List[float] = [0.1, 0.5, 0.7, 0.9],
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize Linear model.
        
        Args:
            model_type: Type of linear model ('ridge', 'lasso', 'elasticnet', 'linear')
            alpha_values: List of alpha parameter values to try
            l1_ratio_values: List of l1_ratio values for ElasticNet
            random_state: Random state for reproducibility
            n_jobs: Number of jobs to run in parallel
            **kwargs: Additional parameters
        """
        super().__init__(f"Linear_{model_type.title()}", **kwargs)
        self.model_type = model_type.lower()
        self.alpha_values = alpha_values
        self.l1_ratio_values = l1_ratio_values
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.scaler = StandardScaler()
        self.best_params_ = None
        self.is_fitted = False
        
        # Validate model type
        valid_types = ['ridge', 'lasso', 'elasticnet', 'linear']
        if self.model_type not in valid_types:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {valid_types}")
    
    def _prepare_data_for_linear(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare data for linear models (flatten time series).
        
        Args:
            X: Input data (n_samples, lookback, n_features)
            
        Returns:
            Flattened data (n_samples, lookback * n_features)
        """
        if X.ndim == 3:
            # Flatten time series for linear models
            n_samples, lookback, n_features = X.shape
            return X.reshape(n_samples, lookback * n_features)
        return X
    
    def _get_base_model(self):
        """Get the base model based on model_type."""
        if self.model_type == 'ridge':
            return Ridge(random_state=self.random_state)
        elif self.model_type == 'lasso':
            return Lasso(random_state=self.random_state, max_iter=2000)
        elif self.model_type == 'elasticnet':
            return ElasticNet(random_state=self.random_state, max_iter=2000)
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_param_grid(self):
        """Get parameter grid based on model type."""
        if self.model_type == 'linear':
            return {}  # No hyperparameters for LinearRegression
        elif self.model_type == 'elasticnet':
            return {
                'alpha': self.alpha_values,
                'l1_ratio': self.l1_ratio_values
            }
        else:  # ridge or lasso
            return {
                'alpha': self.alpha_values
            }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearModelsWrapper':
        """
        Train the linear model with GridSearchCV (if applicable).
        
        Args:
            X: Training features (n_samples, lookback, n_features) or (n_samples, n_features)
            y: Training targets (n_samples,)
            
        Returns:
            self
        """
        self.validate_input(X, y)
        
        logger.info(f"Training {self.model_type} model with {X.shape[0]} samples")
        
        # Prepare data for linear models
        X_flat = self._prepare_data_for_linear(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Get base model and parameter grid
        base_model = self._get_base_model()
        param_grid = self._get_param_grid()
        
        if param_grid:  # If there are hyperparameters to tune
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
        else:  # No hyperparameters to tune
            self.model = base_model
            self.model.fit(X_scaled, y)
            self.best_params_ = {}
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        metrics = calculate_metrics(y, y_pred)
        self.set_training_metrics(metrics)
        self.is_fitted = True
        
        logger.info(f"{self.model_type.title()} training completed. Best params: {self.best_params_}")
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
        
        # Prepare data for linear models
        X_flat = self._prepare_data_for_linear(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_flat)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        return y_pred
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Get model coefficients (for linear models).
        
        Returns:
            Coefficients array or None if not available
        """
        if not self.is_fitted or self.model is None:
            return None
        
        if hasattr(self.model, 'coef_'):
            return self.model.coef_
        return None
    
    def get_intercept(self) -> Optional[float]:
        """
        Get model intercept (for linear models).
        
        Returns:
            Intercept value or None if not available
        """
        if not self.is_fitted or self.model is None:
            return None
        
        if hasattr(self.model, 'intercept_'):
            return self.model.intercept_
        return None
    
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
        model_path = os.path.join(path, f'{self.model_type}_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(path, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'best_params': self.best_params_,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"{self.model_type.title()} model saved to {path}")
    
    def load(self, path: str) -> 'LinearModelsWrapper':
        """
        Load a previously saved model.
        
        Args:
            path: Directory path to load the model from
            
        Returns:
            self
        """
        # Load model
        model_path = os.path.join(path, f'{self.model_type}_model.joblib')
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
            
            self.model_type = metadata.get('model_type', self.model_type)
            self.best_params_ = metadata.get('best_params', {})
            self.training_metrics = metadata.get('training_metrics', {})
            self.is_fitted = True
        
        logger.info(f"{self.model_type.title()} model loaded from {path}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            'model_type': self.model_type,
            'best_params': self.best_params_,
            'is_fitted': self.is_fitted,
            'has_coefficients': self.get_coefficients() is not None,
            'has_intercept': self.get_intercept() is not None
        })
        return info
