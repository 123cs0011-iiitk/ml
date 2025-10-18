"""
Random Forest Wrapper for Stock Price Prediction

This module implements a production-ready Random Forest model with
GridSearchCV for hyperparameter tuning and feature importance analysis.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from ..model_interface import ModelInterface
from ..utils import calculate_metrics

logger = logging.getLogger(__name__)


class RandomForestWrapper(ModelInterface):
    """
    Random Forest model wrapper for stock price prediction.
    
    Uses GridSearchCV for hyperparameter tuning and provides
    feature importance analysis.
    """
    
    def __init__(self, n_estimators: List[int] = [200, 500], 
                 max_depth: List[int] = [10, 20, 30],
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: List of number of trees to try
            max_depth: List of maximum depths to try
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            random_state: Random state for reproducibility
            n_jobs: Number of jobs to run in parallel
            **kwargs: Additional parameters
        """
        super().__init__("RandomForest", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importances_ = None
        self.best_params_ = None
        self.is_fitted = False
        
    def _prepare_data_for_rf(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare data for Random Forest (flatten time series).
        
        Args:
            X: Input data (n_samples, lookback, n_features)
            
        Returns:
            Flattened data (n_samples, lookback * n_features)
        """
        if X.ndim == 3:
            # Flatten time series for Random Forest
            n_samples, lookback, n_features = X.shape
            return X.reshape(n_samples, lookback * n_features)
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestWrapper':
        """
        Train the Random Forest model with GridSearchCV.
        
        Args:
            X: Training features (n_samples, lookback, n_features) or (n_samples, n_features)
            y: Training targets (n_samples,)
            
        Returns:
            self
        """
        self.validate_input(X, y)
        
        logger.info(f"Training Random Forest model with {X.shape[0]} samples")
        
        # Prepare data for Random Forest
        X_flat = self._prepare_data_for_rf(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': [self.min_samples_split],
            'min_samples_leaf': [self.min_samples_leaf]
        }
        
        # Create base model
        base_model = RandomForestRegressor(
            random_state=self.random_state,
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
        self.feature_importances_ = self.model.feature_importances_
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        metrics = calculate_metrics(y, y_pred)
        self.set_training_metrics(metrics)
        self.is_fitted = True
        
        logger.info(f"Random Forest training completed. Best params: {self.best_params_}")
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
        
        # Prepare data for Random Forest
        X_flat = self._prepare_data_for_rf(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_flat)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        return y_pred
    
    def predict_with_uncertainty(self, X: np.ndarray, n_estimates: int = 100) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimation using individual trees.
        
        Args:
            X: Features to predict on
            n_estimates: Number of trees to use for uncertainty estimation
            
        Returns:
            Dictionary with mean predictions and standard deviations
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X_flat = self._prepare_data_for_rf(X)
        X_scaled = self.scaler.transform(X_flat)
        
        # Get predictions from individual trees
        tree_predictions = []
        for tree in self.model.estimators_[:n_estimates]:
            pred = tree.predict(X_scaled)
            tree_predictions.append(pred)
        
        tree_predictions = np.array(tree_predictions)
        
        # Calculate mean and standard deviation
        mean_predictions = np.mean(tree_predictions, axis=0)
        std_predictions = np.std(tree_predictions, axis=0)
        
        return {
            'mean': mean_predictions,
            'std': std_predictions,
            'predictions': tree_predictions
        }
    
    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get feature importances from the trained model.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted or self.feature_importances_ is None:
            raise ValueError("Model must be fitted before getting feature importances")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]
        
        return dict(zip(feature_names, self.feature_importances_))
    
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
        model_path = os.path.join(path, 'random_forest_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(path, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'best_params': self.best_params_,
            'feature_importances': self.feature_importances_.tolist(),
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Random Forest model saved to {path}")
    
    def load(self, path: str) -> 'RandomForestWrapper':
        """
        Load a previously saved model.
        
        Args:
            path: Directory path to load the model from
            
        Returns:
            self
        """
        # Load model
        model_path = os.path.join(path, 'random_forest_model.joblib')
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
            
            self.best_params_ = metadata.get('best_params', {})
            self.feature_importances_ = np.array(metadata.get('feature_importances', []))
            self.training_metrics = metadata.get('training_metrics', {})
            self.is_fitted = True
        
        logger.info(f"Random Forest model loaded from {path}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            'best_params': self.best_params_,
            'feature_importances_available': self.feature_importances_ is not None,
            'is_fitted': self.is_fitted
        })
        return info
