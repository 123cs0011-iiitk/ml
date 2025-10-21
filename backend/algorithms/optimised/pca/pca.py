"""
Principal Component Analysis for Feature Reduction and Stock Prediction

Optimized PCA implementation for dimensionality reduction and stock price prediction.
Uses PCA for feature reduction and noise reduction, then applies regression
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging

logger = logging.getLogger(__name__)

class PCAModel(ModelInterface):
    """
    PCA model for feature reduction and stock price prediction.
    
    Uses PCA for dimensionality reduction and noise reduction, then applies
    linear regression on the reduced features for stock price prediction.
    """
    
    def __init__(self, n_components: int = None, variance_threshold: float = 0.95, 
                 **kwargs):
        """
        Initialize PCA model for feature reduction.
        
        Args:
            n_components: Number of principal components to keep
            variance_threshold: Minimum variance to retain (if n_components not specified)
        """
        super().__init__('Principal Component Analysis', **kwargs)
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        
        # Models
        self.pca = None
        self.regressor = None
        self.scaler = None
        self.feature_columns = None
        self.explained_variance_ratio = None
        self.cumulative_variance = None
        
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
        # Fit PCA with all components to analyze variance
        temp_pca = PCA()
        temp_pca.fit(X)
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
        
        # Find number of components that explain the threshold variance
        n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        
        # Ensure we have at least 2 components
        n_components = max(2, n_components)
        
        logger.info(f"Optimal number of components: {n_components} "
                   f"(explains {cumulative_variance[n_components-1]*100:.2f}% of variance)")
        
        return n_components
    
    def _analyze_components(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Analyze principal components and their importance.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with component analysis
        """
        # Fit PCA with all components for analysis
        temp_pca = PCA()
        temp_pca.fit(X)
        
        # Calculate component statistics
        explained_variance_ratio = temp_pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find components that explain significant variance
        significant_components = np.where(explained_variance_ratio > 0.01)[0]  # > 1% variance
        
        analysis = {
            'total_components': len(explained_variance_ratio),
            'significant_components': len(significant_components),
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'first_component_variance': explained_variance_ratio[0] if len(explained_variance_ratio) > 0 else 0,
            'last_component_variance': explained_variance_ratio[-1] if len(explained_variance_ratio) > 0 else 0
        }
        
        return analysis
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the PCA model for feature reduction and prediction.
        
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
            
            # Perform PCA
            logger.info(f"Performing PCA with {self.n_components} components...")
            self.pca = PCA(n_components=self.n_components)
            X_transformed = self.pca.fit_transform(X_scaled)
            
            # Store variance information
            self.explained_variance_ratio = self.pca.explained_variance_ratio_
            self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
            
            # Train regression model on transformed features
            logger.info("Training regression model on PCA-transformed features...")
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
                'explained_variance': self.cumulative_variance[-1],
                'variance_threshold': self.variance_threshold
            }
            
            self.is_trained = True
            logger.info(f"PCA training completed. R² Score: {self.training_metrics['r2_score']:.4f}, "
                       f"Explained variance: {self.cumulative_variance[-1]*100:.2f}%")
            
        except Exception as e:
            logger.error(f"Error training PCA model: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using PCA-transformed features.
        
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
            
            # Transform features using PCA
            X_transformed = self.pca.transform(X_scaled)
            
            # Make predictions
            predictions = self.regressor.predict(X_transformed)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def get_transformed_features(self, X: np.ndarray) -> np.ndarray:
        """
        Get PCA-transformed features.
        
        Args:
            X: Input features
            
        Returns:
            PCA-transformed features
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def get_component_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of principal components.
        
        Returns:
            Dictionary with component analysis
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return {
            'n_components': self.n_components,
            'explained_variance_ratio': self.explained_variance_ratio,
            'cumulative_variance': self.cumulative_variance,
            'total_variance_explained': self.cumulative_variance[-1],
            'components': self.pca.components_,
            'mean': self.pca.mean_,
            'noise_variance': self.pca.noise_variance_
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on PCA components.
        
        Returns:
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Calculate feature importance based on component loadings
        component_weights = self.regressor.coef_
        feature_importance = np.abs(self.pca.components_.T @ component_weights)
        
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
        X_transformed = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        return reconstruction_error
    
    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            joblib.dump({
                'pca': self.pca,
                'regressor': self.regressor,
                'scaler': self.scaler,
                'explained_variance_ratio': self.explained_variance_ratio,
                'cumulative_variance': self.cumulative_variance,
                'n_components': self.n_components,
                'variance_threshold': self.variance_threshold,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'model_params': self.model_params
            }, path)
            
            logger.info(f"PCA model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving PCA model: {str(e)}")
            raise
    
    def load(self, path: str) -> 'ModelInterface':
        """Load a previously saved model from disk."""
        try:
            data = joblib.load(path)
            self.pca = data['pca']
            self.regressor = data['regressor']
            self.scaler = data['scaler']
            self.explained_variance_ratio = data['explained_variance_ratio']
            self.cumulative_variance = data['cumulative_variance']
            self.n_components = data['n_components']
            self.variance_threshold = data['variance_threshold']
            self.feature_columns = data['feature_columns']
            self.training_metrics = data['training_metrics']
            self.model_params = data['model_params']
            
            self.is_trained = True
            logger.info(f"PCA model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading PCA model: {str(e)}")
            raise
        
        return self
