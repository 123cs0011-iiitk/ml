"""
K-Nearest Neighbors (KNN) Wrapper for Stock Price Prediction

This module implements a production-ready KNN regression model
with hyperparameter tuning using GridSearchCV.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from ..model_interface import ModelInterface
from ..utils import calculate_metrics

logger = logging.getLogger(__name__)


class KNNWrapper(ModelInterface):
    """
    K-Nearest Neighbors regression model wrapper for stock price prediction.
    
    Uses GridSearchCV for hyperparameter tuning.
    """
    
    def __init__(self, 
                 n_neighbors_values: List[int] = [3, 5, 7, 9, 11, 15, 20],
                 weights: List[str] = ['uniform', 'distance'],
                 algorithm: str = 'auto',
                 leaf_size: int = 30,
                 p_values: List[int] = [1, 2],  # 1 for Manhattan, 2 for Euclidean
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize KNN model.
        
        Args:
            n_neighbors_values: List of k values to try
            weights: List of weight functions to try
            algorithm: Algorithm used to compute nearest neighbors
            leaf_size: Leaf size for tree-based algorithms
            p_values: List of p values for Minkowski distance
            n_jobs: Number of jobs to run in parallel
            **kwargs: Additional parameters
        """
        super().__init__("KNN", **kwargs)
        self.n_neighbors_values = n_neighbors_values
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p_values = p_values
        self.n_jobs = n_jobs
        
        self.model = None
        self.scaler = StandardScaler()
        self.best_params_ = None
        self.is_fitted = False
        
    def _prepare_data_for_knn(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare data for KNN (flatten time series).
        
        Args:
            X: Input data (n_samples, lookback, n_features)
            
        Returns:
            Flattened data (n_samples, lookback * n_features)
        """
        if X.ndim == 3:
            # Flatten time series for KNN
            n_samples, lookback, n_features = X.shape
            return X.reshape(n_samples, lookback * n_features)
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNWrapper':
        """
        Train the KNN model with GridSearchCV.
        
        Args:
            X: Training features (n_samples, lookback, n_features) or (n_samples, n_features)
            y: Training targets (n_samples,)
            
        Returns:
            self
        """
        self.validate_input(X, y)
        
        logger.info(f"Training KNN model with {X.shape[0]} samples")
        
        # Prepare data for KNN
        X_flat = self._prepare_data_for_knn(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': self.n_neighbors_values,
            'weights': self.weights,
            'p': self.p_values
        }
        
        # Create base model
        base_model = KNeighborsRegressor(
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs
        )
        
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
        
        logger.info(f"KNN training completed. Best params: {self.best_params_}")
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
        
        # Prepare data for KNN
        X_flat = self._prepare_data_for_knn(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_flat)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        return y_pred
    
    def predict_with_distances(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions and return distances to neighbors.
        
        Args:
            X: Features to predict on
            
        Returns:
            Dictionary with predictions and distances
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X_flat = self._prepare_data_for_knn(X)
        X_scaled = self.scaler.transform(X_flat)
        
        # Get predictions and distances
        distances, indices = self.model.kneighbors(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        return {
            'predictions': predictions,
            'distances': distances,
            'indices': indices
        }
    
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
        model_path = os.path.join(path, 'knn_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(path, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'best_params': self.best_params_,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"KNN model saved to {path}")
    
    def load(self, path: str) -> 'KNNWrapper':
        """
        Load a previously saved model.
        
        Args:
            path: Directory path to load the model from
            
        Returns:
            self
        """
        # Load model
        model_path = os.path.join(path, 'knn_model.joblib')
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
            
            self.algorithm = metadata.get('algorithm', self.algorithm)
            self.leaf_size = metadata.get('leaf_size', self.leaf_size)
            self.best_params_ = metadata.get('best_params', {})
            self.training_metrics = metadata.get('training_metrics', {})
            self.is_fitted = True
        
        logger.info(f"KNN model loaded from {path}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'best_params': self.best_params_,
            'is_fitted': self.is_fitted
        })
        return info
