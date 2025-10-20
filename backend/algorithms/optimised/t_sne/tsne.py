"""
t-SNE for Pattern Recognition and Stock Prediction

Optimized t-SNE implementation for pattern recognition and stock price prediction.
Uses t-SNE for non-linear dimensionality reduction and pattern recognition, then
applies regression for stock price prediction based on OHLCV data and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model_interface import ModelInterface
from stock_indicators import StockIndicators
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
import joblib
import logging

logger = logging.getLogger(__name__)

class TSNEModel(ModelInterface):
    """
    t-SNE model for pattern recognition and stock price prediction.
    
    Uses t-SNE for non-linear dimensionality reduction and pattern recognition,
    then applies linear regression on the transformed features for stock price prediction.
    """
    
    def __init__(self, n_components: int = 2, perplexity: float = 30, 
                 learning_rate: float = 200, n_iter: int = 1000, **kwargs):
        """
        Initialize t-SNE model for pattern recognition.
        
        Args:
            n_components: Number of components for t-SNE embedding
            perplexity: Perplexity parameter for t-SNE
            learning_rate: Learning rate for t-SNE optimization
            n_iter: Number of iterations for t-SNE
        """
        super().__init__('t-SNE Pattern Recognition', **kwargs)
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
        # Models
        self.tsne = None
        self.regressor = None
        self.scaler = None
        self.feature_columns = None
        self.embedding = None
        self.kl_divergence = None
        
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from OHLC data (no volume)."""
        return StockIndicators.calculate_all_indicators(df)
    
    def _find_optimal_perplexity(self, X: np.ndarray) -> float:
        """
        Find optimal perplexity parameter for t-SNE.
        
        Args:
            X: Input features
            
        Returns:
            Optimal perplexity value
        """
        # Perplexity should be between 5 and 50, and less than n_samples/3
        max_perplexity = min(50, X.shape[0] // 3)
        min_perplexity = max(5, X.shape[0] // 100)
        
        # Use a heuristic: perplexity = min(30, n_samples/4)
        optimal_perplexity = min(30, X.shape[0] // 4)
        optimal_perplexity = max(min_perplexity, min(optimal_perplexity, max_perplexity))
        
        logger.info(f"Optimal perplexity: {optimal_perplexity}")
        return optimal_perplexity
    
    def _analyze_embedding(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze t-SNE embedding and patterns.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Dictionary with embedding analysis
        """
        # Calculate embedding statistics
        embedding_stats = {
            'n_components': self.n_components,
            'perplexity': self.perplexity,
            'learning_rate': self.learning_rate,
            'n_iter': self.n_iter,
            'kl_divergence': self.kl_divergence,
            'embedding_shape': self.embedding.shape if self.embedding is not None else None
        }
        
        if self.embedding is not None:
            # Calculate embedding statistics
            embedding_stats.update({
                'embedding_mean': np.mean(self.embedding, axis=0),
                'embedding_std': np.std(self.embedding, axis=0),
                'embedding_range': np.ptp(self.embedding, axis=0),
                'embedding_variance': np.var(self.embedding, axis=0)
            })
            
            # Calculate local neighborhood preservation
            if X.shape[0] > 1:
                # Calculate nearest neighbors in original space
                nbrs_orig = NearestNeighbors(n_neighbors=min(10, X.shape[0]-1))
                nbrs_orig.fit(X)
                distances_orig, indices_orig = nbrs_orig.kneighbors(X)
                
                # Calculate nearest neighbors in embedding space
                nbrs_embed = NearestNeighbors(n_neighbors=min(10, self.embedding.shape[0]-1))
                nbrs_embed.fit(self.embedding)
                distances_embed, indices_embed = nbrs_embed.kneighbors(self.embedding)
                
                # Calculate neighborhood preservation
                preservation_scores = []
                for i in range(len(X)):
                    orig_neighbors = set(indices_orig[i])
                    embed_neighbors = set(indices_embed[i])
                    intersection = len(orig_neighbors.intersection(embed_neighbors))
                    union = len(orig_neighbors.union(embed_neighbors))
                    preservation = intersection / union if union > 0 else 0
                    preservation_scores.append(preservation)
                
                embedding_stats['neighborhood_preservation'] = np.mean(preservation_scores)
        
        return embedding_stats
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """
        Train the t-SNE model for pattern recognition and prediction.
        
        Args:
            X: Input features (OHLC data with technical indicators)
            y: Target values (stock prices)
        """
        self.validate_input(X, y)
        
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Find optimal perplexity if not specified
            if self.perplexity is None:
                self.perplexity = self._find_optimal_perplexity(X_scaled)
            
            # Perform t-SNE
            logger.info(f"Performing t-SNE with {self.n_components} components, "
                       f"perplexity={self.perplexity}...")
            self.tsne = TSNE(
                n_components=self.n_components,
                perplexity=self.perplexity,
                learning_rate=self.learning_rate,
                n_iter=self.n_iter,
                random_state=42
            )
            self.embedding = self.tsne.fit_transform(X_scaled)
            
            # Store KL divergence
            self.kl_divergence = self.tsne.kl_divergence_
            
            # Train regression model on transformed features
            logger.info("Training regression model on t-SNE-transformed features...")
            self.regressor = LinearRegression()
            self.regressor.fit(self.embedding, y)
            
            # Calculate training metrics
            y_pred = self.predict(X)
            self.training_metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2_score': r2_score(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'n_components': self.n_components,
                'perplexity': self.perplexity,
                'kl_divergence': self.kl_divergence
            }
            
            self.is_trained = True
            logger.info(f"t-SNE training completed. R² Score: {self.training_metrics['r2_score']:.4f}, "
                       f"KL Divergence: {self.kl_divergence:.4f}")
            
        except Exception as e:
            logger.error(f"Error training t-SNE model: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using t-SNE-transformed features.
        
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
            
            # Transform features using t-SNE
            X_transformed = self.tsne.fit_transform(X_scaled)
            
            # Make predictions
            predictions = self.regressor.predict(X_transformed)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def get_transformed_features(self, X: np.ndarray) -> np.ndarray:
        """
        Get t-SNE-transformed features.
        
        Args:
            X: Input features
            
        Returns:
            t-SNE-transformed features
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        return self.tsne.fit_transform(X_scaled)
    
    def get_embedding_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of t-SNE embedding.
        
        Returns:
            Dictionary with embedding analysis
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self._analyze_embedding(None, None)
    
    def get_embedding(self) -> np.ndarray:
        """
        Get the t-SNE embedding of the training data.
        
        Returns:
            t-SNE embedding
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.embedding
    
    def get_kl_divergence(self) -> float:
        """
        Get the KL divergence from t-SNE optimization.
        
        Returns:
            KL divergence
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self.kl_divergence
    
    def get_neighborhood_preservation(self, X: np.ndarray) -> float:
        """
        Calculate neighborhood preservation quality.
        
        Args:
            X: Input features
            
        Returns:
            Neighborhood preservation score
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X)
        X_transformed = self.tsne.fit_transform(X_scaled)
        
        # Calculate nearest neighbors in original space
        nbrs_orig = NearestNeighbors(n_neighbors=min(10, X.shape[0]-1))
        nbrs_orig.fit(X_scaled)
        distances_orig, indices_orig = nbrs_orig.kneighbors(X_scaled)
        
        # Calculate nearest neighbors in embedding space
        nbrs_embed = NearestNeighbors(n_neighbors=min(10, X_transformed.shape[0]-1))
        nbrs_embed.fit(X_transformed)
        distances_embed, indices_embed = nbrs_embed.kneighbors(X_transformed)
        
        # Calculate neighborhood preservation
        preservation_scores = []
        for i in range(len(X)):
            orig_neighbors = set(indices_orig[i])
            embed_neighbors = set(indices_embed[i])
            intersection = len(orig_neighbors.intersection(embed_neighbors))
            union = len(orig_neighbors.union(embed_neighbors))
            preservation = intersection / union if union > 0 else 0
            preservation_scores.append(preservation)
        
        return np.mean(preservation_scores)
    
    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            joblib.dump({
                'tsne': self.tsne,
                'regressor': self.regressor,
                'scaler': self.scaler,
                'embedding': self.embedding,
                'kl_divergence': self.kl_divergence,
                'n_components': self.n_components,
                'perplexity': self.perplexity,
                'learning_rate': self.learning_rate,
                'n_iter': self.n_iter,
                'feature_columns': self.feature_columns,
                'training_metrics': self.training_metrics,
                'model_params': self.model_params
            }, path)
            
            logger.info(f"t-SNE model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving t-SNE model: {str(e)}")
            raise
    
    def load(self, path: str) -> 'ModelInterface':
        """Load a previously saved model from disk."""
        try:
            data = joblib.load(path)
            self.tsne = data['tsne']
            self.regressor = data['regressor']
            self.scaler = data['scaler']
            self.embedding = data['embedding']
            self.kl_divergence = data['kl_divergence']
            self.n_components = data['n_components']
            self.perplexity = data['perplexity']
            self.learning_rate = data['learning_rate']
            self.n_iter = data['n_iter']
            self.feature_columns = data['feature_columns']
            self.training_metrics = data['training_metrics']
            self.model_params = data['model_params']
            
            self.is_trained = True
            logger.info(f"t-SNE model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading t-SNE model: {str(e)}")
            raise
        
        return self
