"""
Singular Value Decomposition for Feature Extraction and Stock Prediction

Optimized SVD implementation for matrix decomposition and stock price prediction.
Uses SVD for feature extraction and dimensionality reduction, then applies regression
for stock price prediction based on OHLCV data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model_interface import ModelInterface
from stock_indicators import StockIndicators
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils.extmath import randomized_svd
import joblib
import logging

logger = logging.getLogger(__name__)

class SVDModel(ModelInterface):
    """
    SVD model for feature extraction and stock price prediction.
    
    Uses SVD for matrix decomposition and dimensionality reduction, then applies
    linear regression on the extracted features for stock price prediction.
    """
    
    def __init__(self, n_components: int = None, variance_threshold: float = 0.95, 
                 algorithm: str = 'randomized', **kwargs):
        """
        Initialize SVD model for feature extraction.
        
        Args:
            n_components: Number of components to keep
            variance_threshold: Minimum variance to retain (if n_components not specified)
            algorithm: SVD algorithm ('randomized', 'arpack')
        """
        super().__init__('Singular Value Decomposition', **kwargs)
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.algorithm = algorithm
        
        # Models
        self.svd = None
        self.regressor = None
        self.scaler = None
        self.feature_columns = None
        self.explained_variance_ratio = None
        self.singular_values = None
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _find_optimal_components(self, X: np.ndarray) -> int:
        """
        Find optimal number of components based on variance threshold.
        
        Args:
            X: Input features
            
        Returns:
            Optimal number of components
        """
        # Fit SVD with all components to analyze variance
        temp_svd = TruncatedSVD(n_components=min(X.shape) - 1, algorithm=self.algorithm)
        temp_svd.fit(X)
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(temp_svd.explained_variance_ratio_)
        
        # Find number of components that explain the threshold variance
        n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        
        # Ensure we have at least 2 components
        n_components = max(2, n_components)
        
        logger.info(f"Optimal number of components: {n_components} "
                   f"(explains {cumulative_variance[n_components-1]*100:.2f}% of variance)")
        
        return n_components
    
    def _analyze_components(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Analyze SVD components and their importance.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with component analysis
        """
        # Fit SVD with all components for analysis
        temp_svd = TruncatedSVD(n_components=min(X.shape) - 1, algorithm=self.algorithm)
        temp_svd.fit(X)
        
        # Calculate component statistics
        explained_variance_ratio = temp_svd.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        singular_values = temp_svd.singular_values_
        
        # Find components that explain significant variance
        significant_components = np.where(explained_variance_ratio > 0.01)[0]  # > 1% variance
        
        analysis = {
            'total_components': len(explained_variance_ratio),
            'significant_components': len(significant_components),
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'singular_values': singular_values,
            'first_component_variance': explained_variance_ratio[0] if len(explained_variance_ratio) > 0 else 0,
            'last_component_variance': explained_variance_ratio[-1] if len(explained_variance_ratio) > 0 else 0,
            'condition_number': singular_values[0] / singular_values[-1] if len(singular_values) > 1 else 1
        }
        
        return analysis
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the SVD model for feature extraction and prediction.
        
        Args:
            X: Input features (OHLC data with technical indicators)
            y: Target values (stock prices)
        """
        self.validate_input(X, y)
        
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Find optimal number of components if not specified
            if self.n_components is None:
                self.n_components = self._find_optimal_components(X_scaled)
            
            # Perform SVD
            logger.info(f"Performing SVD with {self.n_components} components...")
            self.svd = TruncatedSVD(n_components=self.n_components, algorithm=self.algorithm)
            X_transformed = self.svd.fit_transform(X_scaled)
            
            # Store variance information
            self.explained_variance_ratio = self.svd.explained_variance_ratio_
            self.singular_values = self.svd.singular_values_
            
            # Train regression model on transformed features
            logger.info("Training regression model on SVD-transformed features...")
            self.regressor = LinearRegression()
            self.regressor.fit(X_transformed, y)
            
            # Calculate training metrics
            y_pred = self.predict(X)
            self.training_metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2_score': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'n_components': self.n_components,
                'explained_variance': np.sum(self.explained_variance_ratio),
                'variance_threshold': self.variance_threshold
            }
            
            self.is_trained = True
            logger.info(f"SVD training completed. R² Score: {self.training_metrics['r2_score']:.4f}, "
                       f"Explained variance: {np.sum(self.explained_variance_ratio)*100:.2f}%")
            
        except Exception as e:
            logger.error(f"Error training SVD model: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using SVD-transformed features.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.validate_input(X)
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Transform features using SVD
            X_transformed = self.svd.transform(X_scaled)
            
            # Make predictions
            predictions = self.regressor.predict(X_transformed)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def get_transformed_features(self, X: np.ndarray) -> np.ndarray:
        """
        Get SVD-transformed features.
        
        Args:
            X: Input features
            
        Returns:
            SVD-transformed features
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.svd.transform(X_scaled)
    
    def get_component_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of SVD components.
        
        Returns:
            Dictionary with component analysis
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return {
            'n_components': self.n_components,
            'explained_variance_ratio': self.explained_variance_ratio,
            'singular_values': self.singular_values,
            'total_variance_explained': np.sum(self.explained_variance_ratio),
            'components': self.svd.components_,
            'algorithm': self.algorithm
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on SVD components.
        
        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Calculate feature importance based on component loadings
        component_weights = self.regressor.coef_
        feature_importance = np.abs(self.svd.components_.T @ component_weights)
        
        return feature_importance
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for each sample.
        
        Args:
            X: Input features
            
        Returns:
            Reconstruction errors
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        X_transformed = self.svd.transform(X_scaled)
        X_reconstructed = self.svd.inverse_transform(X_transformed)
        
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        return reconstruction_error
    
    def get_singular_values(self) -> np.ndarray:
        """
        Get singular values from SVD decomposition.
        
        Returns:
            Singular values
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.singular_values
    
    def get_condition_number(self) -> float:
        """
        Get condition number of the data matrix.
        
        Returns:
            Condition number
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if len(self.singular_values) > 1:
            return self.singular_values[0] / self.singular_values[-1]
        else:
            return 1.0
    
    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            joblib.dump({
                'svd': self.svd,
                'regressor': self.regressor,
                'scaler': self.scaler,
                'explained_variance_ratio': self.explained_variance_ratio,
                'singular_values': self.singular_values,
                'n_components': self.n_components,
                'variance_threshold': self.variance_threshold,
                'algorithm': self.algorithm,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'model_params': self.model_params
            }, path)
            
            logger.info(f"SVD model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving SVD model: {str(e)}")
            raise
    
    def load(self, path: str) -> 'ModelInterface':
        """Load a previously saved model from disk."""
        try:
            data = joblib.load(path)
            self.svd = data['svd']
            self.regressor = data['regressor']
            self.scaler = data['scaler']
            self.explained_variance_ratio = data['explained_variance_ratio']
            self.singular_values = data['singular_values']
            self.n_components = data['n_components']
            self.variance_threshold = data['variance_threshold']
            self.algorithm = data['algorithm']
            self.feature_columns = data['feature_columns']
            self.training_metrics = data['training_metrics']
            self.model_params = data['model_params']
            
            self.is_trained = True
            logger.info(f"SVD model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading SVD model: {str(e)}")
            raise
        
        return self
