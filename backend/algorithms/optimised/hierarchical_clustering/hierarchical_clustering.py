"""
Hierarchical Clustering for Stock Grouping and Prediction

Repurposed hierarchical clustering for stock grouping and stock price prediction.
Uses hierarchical clustering to group similar stocks and then applies group-specific
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import joblib
import logging

logger = logging.getLogger(__name__)

class HierarchicalClusteringModel(ModelInterface):
    """
    Hierarchical clustering model for stock grouping and prediction.
    
    Uses hierarchical clustering to group similar stocks and then applies
    group-specific prediction models for stock price forecasting.
    """
    
    def __init__(self, n_clusters: int = 3, linkage: str = 'ward', 
                 distance_threshold: float = None, **kwargs):
        """
        Initialize Hierarchical clustering model for stock grouping.
        
        Args:
            n_clusters: Number of clusters to form
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            distance_threshold: Distance threshold for cluster formation
        """
        super().__init__('Hierarchical Stock Grouping', **kwargs)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        
        # Models
        self.clustering = None
        self.group_predictors = {}
        self.scaler = None
        self.feature_columns = None
        self.linkage_matrix = None
        self.group_stats = {}
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _compute_linkage_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute linkage matrix for hierarchical clustering.
        
        Args:
            X: Input features
            
        Returns:
            Linkage matrix
        """
        # Compute pairwise distances
        distances = pdist(X, metric='euclidean')
        
        # Compute linkage matrix
        linkage_matrix = linkage(distances, method=self.linkage)
        
        return linkage_matrix
    
    def _find_optimal_clusters(self, linkage_matrix: np.ndarray, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using linkage matrix.
        
        Args:
            linkage_matrix: Linkage matrix from hierarchical clustering
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        # Calculate cophenetic correlation coefficient for different cluster numbers
        best_cophenetic = -1
        optimal_clusters = self.n_clusters
        
        for n_clusters in range(2, max_clusters + 1):
            # Get cluster assignments
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Calculate cophenetic correlation
            from scipy.cluster.hierarchy import cophenet
            cophenetic_distances = cophenet(linkage_matrix)
            original_distances = pdist(self.scaler.transform(self.training_data))
            
            # Calculate correlation
            correlation = np.corrcoef(original_distances, cophenetic_distances)[0, 1]
            
            if correlation > best_cophenetic:
                best_cophenetic = correlation
                optimal_clusters = n_clusters
        
        logger.info(f"Optimal number of clusters: {optimal_clusters} (cophenetic correlation: {best_cophenetic:.4f})")
        return optimal_clusters
    
    def _analyze_groups(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray) -> Dict[int, Dict]:
        """
        Analyze characteristics of each stock group.
        
        Args:
            X: Input features
            y: Target values
            labels: Group labels
            
        Returns:
            Dictionary with group statistics
        """
        group_stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            group_mask = labels == label
            if np.any(group_mask):
                group_prices = y[group_mask]
                group_features = X[group_mask]
                
                group_stats[label] = {
                    'count': np.sum(group_mask),
                    'percentage': np.sum(group_mask) / len(y) * 100,
                    'mean_price': np.mean(group_prices),
                    'std_price': np.std(group_prices),
                    'min_price': np.min(group_prices),
                    'max_price': np.max(group_prices),
                    'volatility': np.std(group_prices) / np.mean(group_prices) if np.mean(group_prices) > 0 else 0,
                    'price_range': np.max(group_prices) - np.min(group_prices),
                    'cv': np.std(group_prices) / np.mean(group_prices) if np.mean(group_prices) > 0 else 0
                }
        
        return group_stats
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the hierarchical clustering model for stock grouping and prediction.
        
        Args:
            X: Input features (OHLC data with technical indicators)
            y: Target values (stock prices)
        """
        self.validate_input(X, y)
        
        try:
            # Store training data for linkage computation
            self.training_data = X.copy()
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Compute linkage matrix
            logger.info(f"Computing linkage matrix with {self.linkage} linkage...")
            self.linkage_matrix = self._compute_linkage_matrix(X_scaled)
            
            # Find optimal number of clusters if not specified
            if self.n_clusters is None:
                self.n_clusters = self._find_optimal_clusters(self.linkage_matrix)
            
            # Perform hierarchical clustering
            logger.info(f"Performing hierarchical clustering with {self.n_clusters} clusters...")
            self.clustering = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                distance_threshold=self.distance_threshold
            )
            labels = self.clustering.fit_predict(X_scaled)
            
            # Analyze groups
            self.group_stats = self._analyze_groups(X, y, labels)
            
            # Train group-specific predictors
            logger.info("Training group-specific predictors...")
            for group in range(self.n_clusters):
                group_mask = labels == group
                if np.any(group_mask):
                    group_X = X[group_mask]
                    group_y = y[group_mask]
                    
                    # Train predictor for this group
                    predictor = LinearRegression()
                    predictor.fit(group_X, group_y)
                    self.group_predictors[group] = predictor
                    
                    logger.info(f"Group {group}: {np.sum(group_mask)} samples, "
                              f"Mean price: {np.mean(group_y):.2f}, "
                              f"Volatility: {self.group_stats[group]['volatility']:.4f}")
            
            # Calculate training metrics
            y_pred = self.predict(X)
            self.training_metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2_score': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'n_clusters': self.n_clusters,
                'linkage': self.linkage
            }
            
            self.is_trained = True
            logger.info(f"Hierarchical clustering training completed. R² Score: {self.training_metrics['r2_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training hierarchical clustering model: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using group-specific models.
        
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
            
            # Predict group assignments
            labels = self.clustering.predict(X_scaled)
            
            # Make predictions using group-specific models
            predictions = np.zeros(len(X))
            
            for group in range(self.n_clusters):
                group_mask = labels == group
                if np.any(group_mask) and group in self.group_predictors:
                    group_X = X[group_mask]
                    group_predictions = self.group_predictors[group].predict(group_X)
                    predictions[group_mask] = group_predictions
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_with_group(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with group information.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, group_labels)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        group_labels = self.clustering.predict(X_scaled)
        predictions = self.predict(X)
        
        return predictions, group_labels
    
    def get_dendrogram_data(self) -> Tuple[np.ndarray, Dict]:
        """
        Get dendrogram data for visualization.
        
        Returns:
            Tuple of (linkage_matrix, dendrogram_data)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.linkage_matrix, {}
    
    def get_group_analysis(self) -> Dict[int, Dict]:
        """
        Get detailed analysis of each stock group.
        
        Returns:
            Dictionary with group statistics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.group_stats
    
    def get_group_similarity(self, group1: int, group2: int) -> float:
        """
        Calculate similarity between two groups.
        
        Args:
            group1: First group ID
            group2: Second group ID
            
        Returns:
            Similarity score between groups
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if group1 not in self.group_stats or group2 not in self.group_stats:
            raise ValueError("Invalid group IDs")
        
        # Calculate similarity based on group characteristics
        stats1 = self.group_stats[group1]
        stats2 = self.group_stats[group2]
        
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
                'group_predictors': self.group_predictors,
                'scaler': self.scaler,
                'linkage_matrix': self.linkage_matrix,
                'group_stats': self.group_stats,
                'n_clusters': self.n_clusters,
                'linkage': self.linkage,
                'distance_threshold': self.distance_threshold,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'model_params': self.model_params
            }, path)
            
            logger.info(f"Hierarchical clustering model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving hierarchical clustering model: {str(e)}")
            raise
    
    def load(self, path: str) -> 'ModelInterface':
        """Load a previously saved model from disk."""
        try:
            data = joblib.load(path)
            self.clustering = data['clustering']
            self.group_predictors = data['group_predictors']
            self.scaler = data['scaler']
            self.linkage_matrix = data['linkage_matrix']
            self.group_stats = data['group_stats']
            self.n_clusters = data['n_clusters']
            self.linkage = data['linkage']
            self.distance_threshold = data['distance_threshold']
            self.feature_columns = data['feature_columns']
            self.training_metrics = data['training_metrics']
            self.model_params = data['model_params']
            
            self.is_trained = True
            logger.info(f"Hierarchical clustering model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading hierarchical clustering model: {str(e)}")
            raise
        
        return self
