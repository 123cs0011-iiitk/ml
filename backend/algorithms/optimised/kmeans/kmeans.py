"""
K-Means Clustering for Market Regime Detection and Stock Prediction

Repurposed K-Means clustering for market regime detection and stock price prediction.
Uses clustering to identify different market regimes and then applies regime-specific
prediction models for stock price forecasting based on OHLCV data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model_interface import ModelInterface
from stock_indicators import StockIndicators
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import logging

logger = logging.getLogger(__name__)

class KMeansModel(ModelInterface):
    """
    K-Means clustering model for market regime detection and stock prediction.
    
    Uses K-Means to identify different market regimes (bull, bear, sideways) and
    applies regime-specific prediction models for stock price forecasting.
    """
    
    def __init__(self, n_clusters: int = 3, regime_models: str = 'linear', 
                 random_state: int = 42, **kwargs):
        """
        Initialize K-Means model for market regime detection.
        
        Args:
            n_clusters: Number of market regimes to identify
            regime_models: Type of models to use for each regime ('linear', 'ensemble')
            random_state: Random state for reproducibility
        """
        super().__init__('K-Means Market Regime Detection', **kwargs)
        self.n_clusters = n_clusters
        self.regime_models = regime_models
        self.random_state = random_state
        
        # Models
        self.kmeans = None
        self.regime_predictors = {}
        self.scaler = None
        self.feature_columns = None
        self.regime_centers = None
        self.regime_stats = {}
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _identify_regimes(self, X: np.ndarray) -> np.ndarray:
        """
        Identify market regimes using K-Means clustering.
        
        Args:
            X: Input features
            
        Returns:
            Regime labels for each sample
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        regime_labels = self.kmeans.predict(X_scaled)
        return regime_labels
    
    def _get_regime_features(self, X: np.ndarray, regime_labels: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Get features for each regime.
        
        Args:
            X: Input features
            regime_labels: Regime labels for each sample
            
        Returns:
            Dictionary mapping regime to features
        """
        regime_features = {}
        for regime in range(self.n_clusters):
            regime_mask = regime_labels == regime
            if np.any(regime_mask):
                regime_features[regime] = X[regime_mask]
        return regime_features
    
    def _analyze_regimes(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[int, Dict]:
        """
        Analyze characteristics of each market regime.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Returns:
            Dictionary with regime statistics
        """
        regime_stats = {}
        
        for regime in range(self.n_clusters):
            regime_mask = regime_labels == regime
            if np.any(regime_mask):
                regime_prices = y[regime_mask]
                regime_features = X[regime_mask]
                
                regime_stats[regime] = {
                    'count': np.sum(regime_mask),
                    'percentage': np.sum(regime_mask) / len(y) * 100,
                    'mean_price': np.mean(regime_prices),
                    'std_price': np.std(regime_prices),
                    'min_price': np.min(regime_prices),
                    'max_price': np.max(regime_prices),
                    'volatility': np.std(regime_prices) / np.mean(regime_prices) if np.mean(regime_prices) > 0 else 0,
                    'center': self.regime_centers[regime] if self.regime_centers is not None else None
                }
        
        return regime_stats
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the K-Means model for market regime detection and prediction.
        
        Args:
            X: Input features (OHLC data with technical indicators)
            y: Target values (stock prices)
        """
        self.validate_input(X, y)
        
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform K-Means clustering
            logger.info(f"Performing K-Means clustering with {self.n_clusters} clusters...")
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            regime_labels = self.kmeans.fit_predict(X_scaled)
            
            # Store regime centers
            self.regime_centers = self.kmeans.cluster_centers_
            
            # Analyze regimes
            self.regime_stats = self._analyze_regimes(X, y, regime_labels)
            
            # Train regime-specific predictors
            logger.info("Training regime-specific predictors...")
            for regime in range(self.n_clusters):
                regime_mask = regime_labels == regime
                if np.any(regime_mask):
                    regime_X = X[regime_mask]
                    regime_y = y[regime_mask]
                    
                    if self.regime_models == 'linear':
                        predictor = LinearRegression()
                    else:
                        # For ensemble, use multiple models
                        predictor = LinearRegression()  # Simplified for now
                    
                    predictor.fit(regime_X, regime_y)
                    self.regime_predictors[regime] = predictor
                    
                    logger.info(f"Regime {regime}: {np.sum(regime_mask)} samples, "
                              f"Mean price: {np.mean(regime_y):.2f}, "
                              f"Volatility: {self.regime_stats[regime]['volatility']:.4f}")
            
            # Calculate training metrics
            y_pred = self.predict(X)
            self.training_metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2_score': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'inertia': self.kmeans.inertia_,
                'n_clusters': self.n_clusters
            }
            
            self.is_trained = True
            logger.info(f"K-Means training completed. R² Score: {self.training_metrics['r2_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training K-Means model: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using regime-specific models.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.validate_input(X)
        
        try:
            # Identify regimes for new data
            regime_labels = self._identify_regimes(X)
            
            # Make predictions using regime-specific models
            predictions = np.zeros(len(X))
            
            for regime in range(self.n_clusters):
                regime_mask = regime_labels == regime
                if np.any(regime_mask) and regime in self.regime_predictors:
                    regime_X = X[regime_mask]
                    regime_predictions = self.regime_predictors[regime].predict(regime_X)
                    predictions[regime_mask] = regime_predictions
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_with_regime(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with regime information.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, regime_labels)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        regime_labels = self._identify_regimes(X)
        predictions = self.predict(X)
        
        return predictions, regime_labels
    
    def get_regime_analysis(self) -> Dict[int, Dict]:
        """
        Get detailed analysis of each market regime.
        
        Returns:
            Dictionary with regime statistics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.regime_stats
    
    def get_regime_centers(self) -> np.ndarray:
        """
        Get cluster centers for each regime.
        
        Returns:
            Array of cluster centers
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.regime_centers
    
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            X: Input features
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        X_scaled = self.scaler.transform(X)
        
        inertias = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        if len(inertias) > 1:
            # Calculate second derivative to find elbow
            second_deriv = np.diff(inertias, 2)
            elbow_idx = np.argmax(second_deriv) + 2  # +2 because of double diff
            optimal_k = k_range[elbow_idx]
        else:
            optimal_k = 3
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            joblib.dump({
                'kmeans': self.kmeans,
                'regime_predictors': self.regime_predictors,
                'scaler': self.scaler,
                'regime_centers': self.regime_centers,
                'regime_stats': self.regime_stats,
                'n_clusters': self.n_clusters,
                'regime_models': self.regime_models,
                'random_state': self.random_state,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'model_params': self.model_params
            }, path)
            
            logger.info(f"K-Means model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving K-Means model: {str(e)}")
            raise
    
    def load(self, path: str) -> 'ModelInterface':
        """Load a previously saved model from disk."""
        try:
            data = joblib.load(path)
            self.kmeans = data['kmeans']
            self.regime_predictors = data['regime_predictors']
            self.scaler = data['scaler']
            self.regime_centers = data['regime_centers']
            self.regime_stats = data['regime_stats']
            self.n_clusters = data['n_clusters']
            self.regime_models = data['regime_models']
            self.random_state = data['random_state']
            self.feature_columns = data['feature_columns']
            self.training_metrics = data['training_metrics']
            self.model_params = data['model_params']
            
            self.is_trained = True
            logger.info(f"K-Means model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading K-Means model: {str(e)}")
            raise
        
        return self
