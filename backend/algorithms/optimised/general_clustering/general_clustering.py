"""
General Clustering for Pattern-Based Stock Prediction

Repurposed general clustering algorithms for pattern-based stock prediction.
Uses multiple clustering algorithms to identify market patterns and then applies
pattern-specific prediction models for stock price forecasting based on OHLCV data
and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model_interface import ModelInterface
from stock_indicators import StockIndicators
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import logging

logger = logging.getLogger(__name__)

class GeneralClusteringModel(ModelInterface):
    """
    General clustering model for pattern-based stock prediction.
    
    Uses multiple clustering algorithms to identify market patterns and then applies
    pattern-specific prediction models for stock price forecasting.
    """
    
    def __init__(self, algorithm: str = 'kmeans', n_clusters: int = 3, 
                 **kwargs):
        """
        Initialize General clustering model for pattern-based prediction.
        
        Args:
            algorithm: Clustering algorithm ('kmeans', 'hierarchical', 'dbscan', 'gmm')
            n_clusters: Number of clusters/patterns to identify
        """
        super().__init__('General Pattern-Based Clustering', **kwargs)
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        
        # Models
        self.clustering = None
        self.pattern_predictors = {}
        self.scaler = None
        self.feature_columns = None
        self.pattern_stats = {}
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _initialize_clustering(self, X: np.ndarray) -> Any:
        """
        Initialize clustering algorithm based on selected method.
        
        Args:
            X: Input features
            
        Returns:
            Clustering algorithm instance
        """
        if self.algorithm == 'kmeans':
            return KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.algorithm == 'hierarchical':
            return AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
        elif self.algorithm == 'dbscan':
            # Auto-determine eps based on data
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=4)
            nbrs.fit(X)
            distances, indices = nbrs.kneighbors(X)
            eps = np.percentile(distances[:, 3], 90)  # 90th percentile of 4th nearest neighbor
            return DBSCAN(eps=eps, min_samples=5)
        elif self.algorithm == 'gmm':
            return GaussianMixture(n_components=self.n_clusters, random_state=42)
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")
    
    def _analyze_patterns(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray) -> Dict[int, Dict]:
        """
        Analyze characteristics of each market pattern.
        
        Args:
            X: Input features
            y: Target values
            labels: Pattern labels
            
        Returns:
            Dictionary with pattern statistics
        """
        pattern_stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise/Outlier pattern
                pattern_mask = labels == -1
                pattern_name = "Outliers"
            else:
                pattern_mask = labels == label
                pattern_name = f"Pattern_{label}"
            
            if np.any(pattern_mask):
                pattern_prices = y[pattern_mask]
                pattern_features = X[pattern_mask]
                
                pattern_stats[label] = {
                    'name': pattern_name,
                    'count': np.sum(pattern_mask),
                    'percentage': np.sum(pattern_mask) / len(y) * 100,
                    'mean_price': np.mean(pattern_prices),
                    'std_price': np.std(pattern_prices),
                    'min_price': np.min(pattern_prices),
                    'max_price': np.max(pattern_prices),
                    'volatility': np.std(pattern_prices) / np.mean(pattern_prices) if np.mean(pattern_prices) > 0 else 0,
                    'price_range': np.max(pattern_prices) - np.min(pattern_prices),
                    'cv': np.std(pattern_prices) / np.mean(pattern_prices) if np.mean(pattern_prices) > 0 else 0,
                    'is_outlier': label == -1
                }
        
        return pattern_stats
    
    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using multiple methods.
        
        Args:
            X: Input features
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        if self.algorithm == 'kmeans':
            # Use elbow method for K-Means
            inertias = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point
            if len(inertias) > 1:
                second_deriv = np.diff(inertias, 2)
                elbow_idx = np.argmax(second_deriv) + 2
                optimal_k = k_range[elbow_idx]
            else:
                optimal_k = 3
                
        elif self.algorithm == 'gmm':
            # Use BIC for Gaussian Mixture
            bic_scores = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                gmm = GaussianMixture(n_components=k, random_state=42)
                gmm.fit(X)
                bic_scores.append(gmm.bic(X))
            
            optimal_k = k_range[np.argmin(bic_scores)]
            
        else:
            # Default to 3 for other algorithms
            optimal_k = 3
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the general clustering model for pattern-based prediction.
        
        Args:
            X: Input features (OHLC data with technical indicators)
            y: Target values (stock prices)
        """
        self.validate_input(X, y)
        
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Find optimal number of clusters if not specified
            if self.n_clusters is None:
                self.n_clusters = self._find_optimal_clusters(X_scaled)
            
            # Initialize and fit clustering algorithm
            logger.info(f"Performing {self.algorithm} clustering with {self.n_clusters} clusters...")
            self.clustering = self._initialize_clustering(X_scaled)
            labels = self.clustering.fit_predict(X_scaled)
            
            # Analyze patterns
            self.pattern_stats = self._analyze_patterns(X, y, labels)
            
            # Train pattern-specific predictors
            logger.info("Training pattern-specific predictors...")
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                if label == -1:  # Skip outliers
                    continue
                    
                pattern_mask = labels == label
                if np.any(pattern_mask):
                    pattern_X = X[pattern_mask]
                    pattern_y = y[pattern_mask]
                    
                    # Train predictor for this pattern
                    predictor = LinearRegression()
                    predictor.fit(pattern_X, pattern_y)
                    self.pattern_predictors[label] = predictor
                    
                    logger.info(f"Pattern {label}: {np.sum(pattern_mask)} samples, "
                              f"Mean price: {np.mean(pattern_y):.2f}, "
                              f"Volatility: {self.pattern_stats[label]['volatility']:.4f}")
            
            # Calculate training metrics
            y_pred = self.predict(X)
            self.training_metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2_score': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'n_clusters': self.n_clusters,
                'algorithm': self.algorithm
            }
            
            self.is_trained = True
            logger.info(f"General clustering training completed. R² Score: {self.training_metrics['r2_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training general clustering model: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using pattern-specific models.
        
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
            
            # Predict pattern assignments
            labels = self.clustering.predict(X_scaled)
            
            # Make predictions using pattern-specific models
            predictions = np.zeros(len(X))
            
            for label in np.unique(labels):
                if label == -1:  # Skip outliers
                    continue
                    
                pattern_mask = labels == label
                if np.any(pattern_mask) and label in self.pattern_predictors:
                    pattern_X = X[pattern_mask]
                    pattern_predictions = self.pattern_predictors[label].predict(pattern_X)
                    predictions[pattern_mask] = pattern_predictions
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_with_pattern(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with pattern information.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, pattern_labels)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        pattern_labels = self.clustering.predict(X_scaled)
        predictions = self.predict(X)
        
        return predictions, pattern_labels
    
    def get_pattern_analysis(self) -> Dict[int, Dict]:
        """
        Get detailed analysis of each market pattern.
        
        Returns:
            Dictionary with pattern statistics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.pattern_stats
    
    def get_pattern_similarity(self, pattern1: int, pattern2: int) -> float:
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern ID
            pattern2: Second pattern ID
            
        Returns:
            Similarity score between patterns
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if pattern1 not in self.pattern_stats or pattern2 not in self.pattern_stats:
            raise ValueError("Invalid pattern IDs")
        
        # Calculate similarity based on pattern characteristics
        stats1 = self.pattern_stats[pattern1]
        stats2 = self.pattern_stats[pattern2]
        
        # Price similarity
        price_sim = 1 - abs(stats1['mean_price'] - stats2['mean_price']) / max(stats1['mean_price'], stats2['mean_price'])
        
        # Volatility similarity
        vol_sim = 1 - abs(stats1['volatility'] - stats2['volatility']) / max(stats1['volatility'], stats2['volatility'])
        
        # Overall similarity
        similarity = (price_sim + vol_sim) / 2
        
        return similarity
    
    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            joblib.dump({
                'clustering': self.clustering,
                'pattern_predictors': self.pattern_predictors,
                'scaler': self.scaler,
                'pattern_stats': self.pattern_stats,
                'n_clusters': self.n_clusters,
                'algorithm': self.algorithm,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'model_params': self.model_params
            }, path)
            
            logger.info(f"General clustering model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving general clustering model: {str(e)}")
            raise
    
    def load(self, path: str) -> 'ModelInterface':
        """Load a previously saved model from disk."""
        try:
            data = joblib.load(path)
            self.clustering = data['clustering']
            self.pattern_predictors = data['pattern_predictors']
            self.scaler = data['scaler']
            self.pattern_stats = data['pattern_stats']
            self.n_clusters = data['n_clusters']
            self.algorithm = data['algorithm']
            self.feature_columns = data['feature_columns']
            self.training_metrics = data['training_metrics']
            self.model_params = data['model_params']
            
            self.is_trained = True
            logger.info(f"General clustering model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading general clustering model: {str(e)}")
            raise
        
        return self
