"""
ARIMA Wrapper for Stock Price Prediction

This module implements a production-ready ARIMA model using pmdarima.auto_arima
for automatic parameter selection and prediction intervals.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import pmdarima as pm
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    logger.warning("pmdarima not available, falling back to statsmodels")

try:
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available")

import joblib

from ..model_interface import ModelInterface
from ..utils import calculate_metrics

logger = logging.getLogger(__name__)


class ARIMAWrapper(ModelInterface):
    """
    ARIMA model wrapper for stock price prediction.
    
    Uses pmdarima.auto_arima for automatic parameter selection
    and provides prediction intervals.
    """
    
    def __init__(self, 
                 seasonal: bool = False,
                 max_p: int = 5,
                 max_d: int = 2,
                 max_q: int = 5,
                 max_P: int = 2,
                 max_D: int = 1,
                 max_Q: int = 2,
                 stepwise: bool = True,
                 suppress_warnings: bool = True,
                 error_action: str = 'ignore',
                 **kwargs):
        """
        Initialize ARIMA model.
        
        Args:
            seasonal: Whether to consider seasonal ARIMA
            max_p: Maximum value of p
            max_d: Maximum value of d
            max_q: Maximum value of q
            max_P: Maximum value of P (seasonal)
            max_D: Maximum value of D (seasonal)
            max_Q: Maximum value of Q (seasonal)
            stepwise: Whether to use stepwise selection
            suppress_warnings: Whether to suppress warnings
            error_action: Action to take on error
            **kwargs: Additional parameters
        """
        super().__init__("ARIMA", **kwargs)
        self.seasonal = seasonal
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        self.error_action = error_action
        
        self.model = None
        self.order = None
        self.seasonal_order = None
        self.is_fitted = False
        
    def _check_stationarity(self, series: np.ndarray) -> bool:
        """
        Check if the time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series data
            
        Returns:
            True if stationary, False otherwise
        """
        if not STATSMODELS_AVAILABLE:
            return False
        
        try:
            result = adfuller(series)
            return result[1] <= 0.05  # p-value <= 0.05 means stationary
        except:
            return False
    
    def _make_stationary(self, series: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Make time series stationary by differencing.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (stationary_series, differencing_order)
        """
        d = 0
        current_series = series.copy()
        
        while d < self.max_d and not self._check_stationarity(current_series):
            current_series = np.diff(current_series)
            d += 1
        
        return current_series, d
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ARIMAWrapper':
        """
        Train the ARIMA model.
        
        Args:
            X: Training features (ignored for ARIMA, uses only target)
            y: Training targets (n_samples,)
            
        Returns:
            self
        """
        self.validate_input(X, y)
        
        logger.info(f"Training ARIMA model with {len(y)} samples")
        
        # ARIMA only uses the target variable (y)
        # For time series, we assume y is already in chronological order
        series = y.copy()
        
        # Check if we have enough data
        if len(series) < 10:
            raise ValueError("ARIMA requires at least 10 data points")
        
        try:
            if PMDARIMA_AVAILABLE:
                # Use pmdarima.auto_arima
                self.model = auto_arima(
                    series,
                    seasonal=self.seasonal,
                    max_p=self.max_p,
                    max_d=self.max_d,
                    max_q=self.max_q,
                    max_P=self.max_P,
                    max_D=self.max_D,
                    max_Q=self.max_Q,
                    stepwise=self.stepwise,
                    suppress_warnings=self.suppress_warnings,
                    error_action=self.error_action,
                    trace=False
                )
                
                self.order = self.model.order
                self.seasonal_order = self.model.seasonal_order
                
            elif STATSMODELS_AVAILABLE:
                # Fallback to statsmodels with manual parameter selection
                logger.warning("Using statsmodels ARIMA with manual parameter selection")
                
                # Make series stationary
                stationary_series, d = self._make_stationary(series)
                
                # Simple parameter selection (can be improved)
                best_aic = float('inf')
                best_order = (1, d, 1)
                
                for p in range(0, min(self.max_p + 1, 3)):
                    for q in range(0, min(self.max_q + 1, 3)):
                        try:
                            temp_model = StatsARIMA(series, order=(p, d, q))
                            temp_fit = temp_model.fit()
                            if temp_fit.aic < best_aic:
                                best_aic = temp_fit.aic
                                best_order = (p, d, q)
                                self.model = temp_fit
                        except:
                            continue
                
                self.order = best_order
                self.seasonal_order = None
                
            else:
                raise ImportError("Neither pmdarima nor statsmodels is available")
            
            # Calculate metrics on training data
            y_pred = self.predict(X)
            metrics = calculate_metrics(y, y_pred)
            self.set_training_metrics(metrics)
            self.is_fitted = True
            
            logger.info(f"ARIMA training completed. Order: {self.order}, Seasonal: {self.seasonal_order}")
            logger.info(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray, n_periods: int = 1) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features (ignored for ARIMA)
            n_periods: Number of periods to predict ahead
            
        Returns:
            predictions: Predicted values (n_periods,)
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            if PMDARIMA_AVAILABLE:
                predictions, conf_int = self.model.predict(n_periods=n_periods, return_conf_int=True)
                return predictions
            else:
                # For statsmodels
                predictions = self.model.forecast(steps=n_periods)
                return predictions
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            raise
    
    def predict_with_intervals(self, X: np.ndarray, n_periods: int = 1, 
                              alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Features (ignored for ARIMA)
            n_periods: Number of periods to predict ahead
            alpha: Significance level for confidence intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            if PMDARIMA_AVAILABLE:
                predictions, conf_int = self.model.predict(
                    n_periods=n_periods, 
                    return_conf_int=True,
                    alpha=alpha
                )
                return {
                    'predictions': predictions,
                    'lower_bound': conf_int[:, 0],
                    'upper_bound': conf_int[:, 1],
                    'confidence_level': 1 - alpha
                }
            else:
                # For statsmodels, we can only get point predictions
                predictions = self.model.forecast(steps=n_periods)
                # Estimate confidence intervals using prediction standard error
                pred_se = self.model.get_forecast(steps=n_periods).se_mean
                z_score = 1.96  # For 95% confidence
                
                return {
                    'predictions': predictions,
                    'lower_bound': predictions - z_score * pred_se,
                    'upper_bound': predictions + z_score * pred_se,
                    'confidence_level': 0.95
                }
        except Exception as e:
            logger.error(f"ARIMA prediction with intervals failed: {e}")
            raise
    
    def save(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Directory path to save the model
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, 'arima_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'seasonal': self.seasonal,
            'training_metrics': self.training_metrics,
            'model_params': self.model_params,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"ARIMA model saved to {path}")
    
    def load(self, path: str) -> 'ARIMAWrapper':
        """
        Load a previously saved model.
        
        Args:
            path: Directory path to load the model from
            
        Returns:
            self
        """
        # Load model
        model_path = os.path.join(path, 'arima_model.joblib')
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.order = tuple(metadata.get('order', (1, 1, 1)))
            self.seasonal_order = metadata.get('seasonal_order')
            self.seasonal = metadata.get('seasonal', False)
            self.training_metrics = metadata.get('training_metrics', {})
            self.is_fitted = True
        
        logger.info(f"ARIMA model loaded from {path}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'seasonal': self.seasonal,
            'is_fitted': self.is_fitted,
            'pmdarima_available': PMDARIMA_AVAILABLE,
            'statsmodels_available': STATSMODELS_AVAILABLE
        })
        return info
