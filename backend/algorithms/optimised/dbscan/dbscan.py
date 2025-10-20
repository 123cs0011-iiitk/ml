"""
DBSCAN Clustering for Anomaly Detection and Stock Prediction

Repurposed DBSCAN clustering for anomaly detection and stock price prediction.
Uses density-based clustering to identify market anomalies and outliers, then
applies anomaly-aware prediction models for stock price forecasting based on
OHLCV data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model_interface import ModelInterface
from stock_indicators import StockIndicators
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
import joblib
import logging

logger = logging.getLogger(__name__)

class DBSCANModel(ModelInterface):
    """
    DBSCAN clustering model for anomaly detection and stock prediction.
    
    Uses DBSCAN to identify market anomalies and outliers, then applies
    anomaly-aware prediction models for stock price forecasting.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 anomaly_threshold: float = 0.1, **kwargs):
        """
        Initialize DBSCAN model for anomaly detection.
        
        Args:
            eps: Maximum distance between two samples for one to be considered
                 in the neighborhood of the other
            min_samples: Minimum number of samples in a neighborhood for a point
                         to be considered as a core point
            anomaly_threshold: Threshold for considering a point as anomalous
        """
        super().__init__('DBSCAN Anomaly Detection', **kwargs)
        self.eps = eps
        self.min_samples = min_samples
        self.anomaly_threshold = anomaly_threshold
        
        # Models
        self.dbscan = None
        self.normal_predictor = None
        self.anomaly_predictor = None
        self.scaler = None
        self.feature_columns = None
        self.cluster_stats = {}
        self.anomaly_ratio = 0.0
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _find_optimal_eps(self, X: np.ndarray, k: int = 4) -> float:
        """
        Find optimal eps parameter using k-distance graph.
        
        Args:
            X: Input features
            k: Number of nearest neighbors to consider
            
        Returns:
            Optimal eps value
        """
        # Calculate k-distance for each point
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Sort distances and find elbow point
        k_distances = np.sort(distances[:, k-1])
        
        # Find elbow point (simplified approach)
        if len(k_distances) > 10:
            # Use 90th percentile as a heuristic for eps
            optimal_eps = np.percentile(k_distances, 90)
        else:
            optimal_eps = np.mean(k_distances)
        
        logger.info(f"Optimal eps: {optimal_eps:.4f}")
        return optimal_eps
    
    def _analyze_clusters(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray) -> Dict[int, Dict]:
        """
        Analyze characteristics of each cluster and anomalies.
        
        Args:
            X: Input features
            y: Target values
            labels: Cluster labels (-1 for noise/anomalies)
            
        Returns:
            Dictionary with cluster statistics
        """
        cluster_stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise/Anomaly cluster
                cluster_mask = labels == -1
                cluster_name = "Anomalies"
            else:
                cluster_mask = labels == label
                cluster_name = f"Cluster_{label}"
            
            if np.any(cluster_mask):
                cluster_prices = y[cluster_mask]
                cluster_features = X[cluster_mask]
                
                cluster_stats[label] = {
                    'name': cluster_name,
                    'count': np.sum(cluster_mask),
                    'percentage': np.sum(cluster_mask) / len(y) * 100,
                    'mean_price': np.mean(cluster_prices),
                    'std_price': np.std(cluster_prices),
                    'min_price': np.min(cluster_prices),
                    'max_price': np.max(cluster_prices),
                    'volatility': np.std(cluster_prices) / np.mean(cluster_prices) if np.mean(cluster_prices) > 0 else 0,
                    'is_anomaly': label == -1
                }
        
        return cluster_stats
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the DBSCAN model for anomaly detection and prediction.
        
        Args:
            X: Input features (OHLC data with technical indicators)
            y: Target values (stock prices)
        """
        self.validate_input(X, y)
        
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Find optimal eps if not provided
            if self.eps is None:
                self.eps = self._find_optimal_eps(X_scaled)
            
            # Perform DBSCAN clustering
            logger.info(f"Performing DBSCAN clustering with eps={self.eps}, min_samples={self.min_samples}...")
            self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = self.dbscan.fit_predict(X_scaled)
            
            # Analyze clusters
            self.cluster_stats = self._analyze_clusters(X, y, labels)
            
            # Calculate anomaly ratio
            n_anomalies = np.sum(labels == -1)
            self.anomaly_ratio = n_anomalies / len(labels)
            
            # Train separate predictors for normal and anomalous data
            logger.info("Training anomaly-aware predictors...")
            
            # Normal data predictor (non-anomaly clusters)
            normal_mask = labels != -1
            if np.any(normal_mask):
                normal_X = X[normal_mask]
                normal_y = y[normal_mask]
                self.normal_predictor = LinearRegression()
                self.normal_predictor.fit(normal_X, normal_y)
                logger.info(f"Normal data: {np.sum(normal_mask)} samples")
            
            # Anomaly data predictor (if enough anomalies)
            anomaly_mask = labels == -1
            if np.any(anomaly_mask) and np.sum(anomaly_mask) >= 5:  # Need at least 5 anomalies
                anomaly_X = X[anomaly_mask]
                anomaly_y = y[anomaly_mask]
                self.anomaly_predictor = LinearRegression()
                self.anomaly_predictor.fit(anomaly_X, anomaly_y)
                logger.info(f"Anomaly data: {np.sum(anomaly_mask)} samples")
            
            # Log cluster information
            for label, stats in self.cluster_stats.items():
                logger.info(f"{stats['name']}: {stats['count']} samples ({stats['percentage']:.1f}%), "
                          f"Mean price: {stats['mean_price']:.2f}, "
                          f"Volatility: {stats['volatility']:.4f}")
            
            # Calculate training metrics
            y_pred = self.predict(X)
            self.training_metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2_score': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'n_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0),
                'n_anomalies': n_anomalies,
                'anomaly_ratio': self.anomaly_ratio
            }
            
            self.is_trained = True
            logger.info(f"DBSCAN training completed. R² Score: {self.training_metrics['r2_score']:.4f}, "
                       f"Anomaly ratio: {self.anomaly_ratio:.2%}")
            
        except Exception as e:
            logger.error(f"Error training DBSCAN model: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using anomaly-aware models.
        
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
            
            # Predict cluster assignments
            labels = self.dbscan.fit_predict(X_scaled)
            
            # Make predictions using appropriate model
            predictions = np.zeros(len(X))
            
            # Normal data predictions
            normal_mask = labels != -1
            if np.any(normal_mask) and self.normal_predictor is not None:
                normal_X = X[normal_mask]
                normal_predictions = self.normal_predictor.predict(normal_X)
                predictions[normal_mask] = normal_predictions
            
            # Anomaly data predictions
            anomaly_mask = labels == -1
            if np.any(anomaly_mask):
                if self.anomaly_predictor is not None:
                    # Use anomaly-specific model
                    anomaly_X = X[anomaly_mask]
                    anomaly_predictions = self.anomaly_predictor.predict(anomaly_X)
                    predictions[anomaly_mask] = anomaly_predictions
                elif self.normal_predictor is not None:
                    # Fallback to normal model for anomalies
                    anomaly_X = X[anomaly_mask]
                    anomaly_predictions = self.normal_predictor.predict(anomaly_X)
                    predictions[anomaly_mask] = anomaly_predictions
                else:
                    # Use mean of training data as fallback
                    predictions[anomaly_mask] = np.mean(self.training_metrics.get('mean_target', 0))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_with_anomaly_info(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with anomaly information.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, cluster_labels, anomaly_scores)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        labels = self.dbscan.fit_predict(X_scaled)
        predictions = self.predict(X)
        
        # Calculate anomaly scores (distance to nearest core point)
        anomaly_scores = np.zeros(len(X))
        for i, point in enumerate(X_scaled):
            if labels[i] == -1:  # Anomaly
                # Distance to nearest core point
                core_points = X_scaled[self.dbscan.core_sample_indices_]
                distances = np.sqrt(np.sum((core_points - point) ** 2, axis=1))
                anomaly_scores[i] = np.min(distances)
            else:
                # Distance to cluster center
                cluster_center = np.mean(X_scaled[labels == labels[i]], axis=0)
                anomaly_scores[i] = np.sqrt(np.sum((point - cluster_center) ** 2))
        
        return predictions, labels, anomaly_scores
    
    def get_anomaly_analysis(self) -> Dict[int, Dict]:
        """
        Get detailed analysis of clusters and anomalies.
        
        Returns:
            Dictionary with cluster statistics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.cluster_stats
    
    def get_anomaly_ratio(self) -> float:
        """
        Get the ratio of anomalies in the dataset.
        
        Returns:
            Anomaly ratio
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.anomaly_ratio
    
    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            joblib.dump({
                'dbscan': self.dbscan,
                'normal_predictor': self.normal_predictor,
                'anomaly_predictor': self.anomaly_predictor,
                'scaler': self.scaler,
                'cluster_stats': self.cluster_stats,
                'anomaly_ratio': self.anomaly_ratio,
                'eps': self.eps,
                'min_samples': self.min_samples,
                'anomaly_threshold': self.anomaly_threshold,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'model_params': self.model_params
            }, path)
            
            logger.info(f"DBSCAN model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving DBSCAN model: {str(e)}")
            raise
    
    def load(self, path: str) -> 'ModelInterface':
        """Load a previously saved model from disk."""
        try:
            data = joblib.load(path)
            self.dbscan = data['dbscan']
            self.normal_predictor = data['normal_predictor']
            self.anomaly_predictor = data['anomaly_predictor']
            self.scaler = data['scaler']
            self.cluster_stats = data['cluster_stats']
            self.anomaly_ratio = data['anomaly_ratio']
            self.eps = data['eps']
            self.min_samples = data['min_samples']
            self.anomaly_threshold = data['anomaly_threshold']
            self.feature_columns = data['feature_columns']
            self.training_metrics = data['training_metrics']
            self.model_params = data['model_params']
            
            self.is_trained = True
            logger.info(f"DBSCAN model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading DBSCAN model: {str(e)}")
            raise
        
        return self
