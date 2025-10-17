"""
Model Tests for ML Prediction System

This module tests the individual ML models and the ensemble system.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms.model_interface import ModelInterface, PredictionResult
from algorithms.utils import (
    build_features_for_symbol, 
    calculate_metrics, 
    horizon_to_days,
    EnsemblePredictor
)


class TestDataPipeline:
    """Test cases for data pipeline functionality."""
    
    def test_horizon_to_days(self):
        """Test horizon string to days conversion."""
        assert horizon_to_days('1d') == 1
        assert horizon_to_days('1w') == 7
        assert horizon_to_days('1m') == 30
        assert horizon_to_days('1y') == 365
        assert horizon_to_days('5y') == 1825
        
        with pytest.raises(ValueError):
            horizon_to_days('invalid')
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([100, 110, 120, 130, 140])
        y_pred = np.array([105, 115, 125, 135, 145])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert metrics['mae'] == 5.0  # Mean absolute error
        assert metrics['rmse'] == 5.0  # Root mean square error
    
    @patch('algorithms.utils.DataPipeline.load_stock_data')
    def test_build_features_for_symbol(self, mock_load_data):
        """Test feature building for a symbol."""
        # Create mock data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        mock_df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000000, 5000000, 100),
            'currency': 'USD'
        })
        mock_load_data.return_value = mock_df
        
        # Test with demo data
        X_train, X_test, y_train, y_test, scalers, metadata = build_features_for_symbol(
            'DEMO', csv_path='data/demo_sample.csv', lookback=10
        )
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert 'scaler' in scalers
        assert 'n_points' in metadata


class TestEnsemblePredictor:
    """Test cases for ensemble prediction system."""
    
    def test_ensemble_initialization(self):
        """Test ensemble predictor initialization."""
        models = {'model1': MagicMock(), 'model2': MagicMock()}
        ensemble = EnsemblePredictor(models)
        
        assert ensemble.models == models
        assert ensemble.weights == {}
    
    def test_calculate_weights_from_rmse(self):
        """Test weight calculation from RMSE values."""
        models = {'model1': MagicMock(), 'model2': MagicMock()}
        ensemble = EnsemblePredictor(models)
        
        validation_metrics = {
            'model1': {'rmse': 0.1, 'mae': 0.05},
            'model2': {'rmse': 0.2, 'mae': 0.1}
        }
        
        weights = ensemble.calculate_weights_from_rmse(validation_metrics)
        
        # Model1 should have higher weight (lower RMSE)
        assert weights['model1'] > weights['model2']
        assert abs(sum(weights.values()) - 1.0) < 1e-6  # Weights should sum to 1
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        models = {'model1': MagicMock(), 'model2': MagicMock()}
        ensemble = EnsemblePredictor(models)
        
        predictions = {'model1': 100.0, 'model2': 102.0}
        individual_results = {
            'model1': {'prediction': 100.0, 'model_info': {}},
            'model2': {'prediction': 102.0, 'model_info': {}}
        }
        
        confidence = ensemble._calculate_confidence(predictions, individual_results)
        
        assert 0 <= confidence <= 100
        assert isinstance(confidence, float)
    
    def test_calculate_price_range(self):
        """Test price range calculation."""
        models = {'model1': MagicMock(), 'model2': MagicMock()}
        ensemble = EnsemblePredictor(models)
        
        ensemble_prediction = 100.0
        predictions = {'model1': 98.0, 'model2': 102.0}
        
        price_range = ensemble._calculate_price_range(ensemble_prediction, predictions)
        
        assert len(price_range) == 2
        assert price_range[0] < ensemble_prediction  # Lower bound
        assert price_range[1] > ensemble_prediction  # Upper bound


class TestModelInterface:
    """Test cases for model interface."""
    
    def test_prediction_result_creation(self):
        """Test PredictionResult creation."""
        result = PredictionResult(
            predicted_price=150.25,
            confidence=75.5,
            price_range=(145.0, 155.5),
            time_frame_days=1,
            model_info={'algorithm': 'Test'},
            data_points_used=1000,
            last_updated='2025-01-17T10:00:00Z',
            currency='USD'
        )
        
        assert result.predicted_price == 150.25
        assert result.confidence == 75.5
        assert result.price_range == (145.0, 155.5)
        assert result.time_frame_days == 1
        assert result.currency == 'USD'
    
    def test_prediction_result_to_dict(self):
        """Test PredictionResult to dictionary conversion."""
        result = PredictionResult(
            predicted_price=150.25,
            confidence=75.5,
            price_range=(145.0, 155.5),
            time_frame_days=1,
            model_info={'algorithm': 'Test'},
            data_points_used=1000,
            last_updated='2025-01-17T10:00:00Z',
            currency='USD'
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['predicted_price'] == 150.25
        assert result_dict['confidence'] == 75.5
        assert result_dict['price_range'] == [145.0, 155.5]
        assert result_dict['currency'] == 'USD'


class TestModelWrappers:
    """Test cases for individual model wrappers."""
    
    def test_lstm_wrapper_initialization(self):
        """Test LSTM wrapper initialization."""
        from algorithms.real.lstm_wrapper import LSTMWrapper
        
        model = LSTMWrapper(lookback=60, lstm_units=(64, 32, 16))
        
        assert model.model_name == "LSTM"
        assert model.lookback == 60
        assert model.lstm_units == (64, 32, 16)
        assert not model.is_fitted
    
    def test_random_forest_wrapper_initialization(self):
        """Test Random Forest wrapper initialization."""
        from algorithms.real.random_forest import RandomForestWrapper
        
        model = RandomForestWrapper(n_estimators=[100, 200], max_depth=[10, 20])
        
        assert model.model_name == "RandomForest"
        assert model.n_estimators == [100, 200]
        assert model.max_depth == [10, 20]
        assert not model.is_fitted
    
    def test_arima_wrapper_initialization(self):
        """Test ARIMA wrapper initialization."""
        from algorithms.real.arima_wrapper import ARIMAWrapper
        
        model = ARIMAWrapper(seasonal=False, max_p=3, max_d=1, max_q=3)
        
        assert model.model_name == "ARIMA"
        assert model.seasonal == False
        assert model.max_p == 3
        assert not model.is_fitted
    
    def test_svr_wrapper_initialization(self):
        """Test SVR wrapper initialization."""
        from algorithms.real.svr import SVRWrapper
        
        model = SVRWrapper(kernel='rbf', C_values=[1, 10], gamma_values=['scale', 'auto'])
        
        assert model.model_name == "SVR"
        assert model.kernel == 'rbf'
        assert model.C_values == [1, 10]
        assert not model.is_fitted
    
    def test_linear_models_wrapper_initialization(self):
        """Test Linear Models wrapper initialization."""
        from algorithms.real.linear_models import LinearModelsWrapper
        
        model = LinearModelsWrapper(model_type='ridge', alpha_values=[0.1, 1.0])
        
        assert model.model_name == "Linear_Ridge"
        assert model.model_type == 'ridge'
        assert model.alpha_values == [0.1, 1.0]
        assert not model.is_fitted
    
    def test_knn_wrapper_initialization(self):
        """Test KNN wrapper initialization."""
        from algorithms.real.knn import KNNWrapper
        
        model = KNNWrapper(n_neighbors_values=[3, 5, 7], weights=['uniform', 'distance'])
        
        assert model.model_name == "KNN"
        assert model.n_neighbors_values == [3, 5, 7]
        assert model.weights == ['uniform', 'distance']
        assert not model.is_fitted


class TestScratchImplementations:
    """Test cases for scratch implementations."""
    
    def test_linear_regression_scratch(self):
        """Test scratch linear regression implementation."""
        from algorithms.scratch.linear_regression_scratch import LinearRegressionScratch
        
        # Create simple test data
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x
        
        model = LinearRegressionScratch(learning_rate=0.01, epochs=100)
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(np.array([6, 7]))
        assert len(predictions) == 2
        assert abs(predictions[0] - 12) < 1  # Should be close to 12
        assert abs(predictions[1] - 14) < 1  # Should be close to 14
    
    def test_naive_bayes_scratch(self):
        """Test scratch naive bayes implementation."""
        from algorithms.scratch.naive_bayes_scratch import NaiveBayesScratch
        
        # Create simple test data
        X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
        y = np.array([1, 1, 0, 0])
        
        model = NaiveBayesScratch()
        model.fit(X, y)
        
        # Test prediction
        predictions = model.predict(np.array([[1, 1], [0, 0]]))
        assert len(predictions) == 2
        assert predictions[0] in [0, 1]
        assert predictions[1] in [0, 1]


if __name__ == '__main__':
    pytest.main([__file__])
