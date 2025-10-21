"""
Stock Predictor

Main prediction orchestrator that uses all algorithms from the optimised directory
to generate predictions for multiple time horizons.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings

# Add algorithms directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithms'))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

from .config import config
from .data_loader import DataLoader
from .prediction_saver import PredictionSaver

# Import all algorithms from optimised directory (with fallback handling)
try:
    from algorithms.optimised.linear_regression.linear_regression import LinearRegressionModel
except ImportError as e:
    logger.warning(f"Could not import LinearRegressionModel: {e}")
    LinearRegressionModel = None

try:
    from algorithms.optimised.random_forest.random_forest import RandomForestModel
except ImportError as e:
    logger.warning(f"Could not import RandomForestModel: {e}")
    RandomForestModel = None

try:
    from algorithms.optimised.knn.knn import KNNModel
except ImportError as e:
    logger.warning(f"Could not import KNNModel: {e}")
    KNNModel = None

try:
    from algorithms.optimised.svm.svm import SVMModel
except ImportError as e:
    logger.warning(f"Could not import SVMModel: {e}")
    SVMModel = None

try:
    from algorithms.optimised.decision_tree.decision_tree import DecisionTreeModel
except ImportError as e:
    logger.warning(f"Could not import DecisionTreeModel: {e}")
    DecisionTreeModel = None

try:
    from algorithms.optimised.ann.ann import ANNModel
except ImportError as e:
    logger.warning(f"Could not import ANNModel: {e}")
    ANNModel = None

try:
    from algorithms.optimised.cnn.cnn import CNNModel
except ImportError as e:
    logger.warning(f"Could not import CNNModel: {e}")
    CNNModel = None

try:
    from algorithms.optimised.arima.arima import ARIMAModel
except ImportError as e:
    logger.warning(f"Could not import ARIMAModel: {e}")
    ARIMAModel = None

try:
    from algorithms.optimised.autoencoders.autoencoder import AutoencoderModel
except ImportError as e:
    logger.warning(f"Could not import AutoencoderModel: {e}")
    AutoencoderModel = None

# Clustering and dimensionality reduction models removed - not used for direct price prediction


class StockPredictor:
    """
    Main stock prediction orchestrator.
    
    Uses all available algorithms to generate ensemble predictions
    for multiple time horizons.
    """
    
    def __init__(self):
        self.config = config
        self.data_loader = DataLoader()
        self.prediction_saver = PredictionSaver()
        
        # Initialize all models
        self.models = self._initialize_models()
        
        # Track model performance
        self.model_performance = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Load pre-trained models from disk."""
        models = {}
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Model configurations
        model_configs = {
            'linear_regression': LinearRegressionModel,
            'random_forest': RandomForestModel,
            'svm': SVMModel,
            'knn': KNNModel,
            'decision_tree': DecisionTreeModel,
            'ann': ANNModel,
            'cnn': CNNModel,
            'arima': ARIMAModel,
            'autoencoder': AutoencoderModel
        }
        
        for model_name, model_class in model_configs.items():
            if model_class is None:
                logger.warning(f"Model class not available: {model_name}")
                continue
            
            model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
            
            if os.path.exists(model_path):
                try:
                    # Load pre-trained model
                    model = model_class().load(model_path)
                    models[model_name] = model
                    logger.info(f"✅ Loaded pre-trained {model_name}")
                except Exception as e:
                    logger.error(f"❌ Error loading {model_name}: {e}")
            else:
                logger.warning(f"⚠️ Pre-trained model not found: {model_path}")
        
        logger.info(f"Loaded {len(models)} pre-trained models: {list(models.keys())}")
        return models
    
    
    def predict_stock(self, symbol: str, category: str) -> bool:
        """
        Generate predictions for a single stock.
        
        Args:
            symbol: Stock symbol
            category: Stock category ('us_stocks' or 'ind_stocks')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting prediction for {symbol} ({category})")
            
            # Load and prepare data (historical only)
            # TODO: For live predictions, use load_stock_data_with_current_price() 
            # and pass current_price parameter fetched from API
            df = self.data_loader.load_stock_data(symbol, category)
            if df is None:
                logger.warning(f"No data available for {symbol}")
                return False
            
            # Validate data quality
            if not self.data_loader.validate_data_quality(df, symbol):
                logger.warning(f"Data quality issues for {symbol}")
                return False
            
            # Create features
            df_with_features = self.data_loader.create_features(df)
            if df_with_features is None or len(df_with_features) == 0:
                logger.warning(f"Could not create features for {symbol}")
                return False
            
            # Prepare training data
            X, y = self.data_loader.prepare_training_data(df_with_features)
            if len(X) == 0 or len(y) == 0:
                logger.warning(f"Insufficient training data for {symbol}")
                return False
            
            # Generate predictions for all time horizons
            all_predictions = []
            
            for horizon in self.config.TIME_HORIZONS.keys():
                logger.debug(f"Generating {horizon} prediction for {symbol}")
                
                horizon_predictions = self._predict_horizon(
                    symbol, category, X, y, horizon, df_with_features
                )
                
                if horizon_predictions:
                    all_predictions.extend(horizon_predictions)
            
            # Save predictions
            if all_predictions:
                success = self.prediction_saver.save_predictions(
                    symbol, category, all_predictions
                )
                
                if success:
                    logger.info(f"Successfully generated {len(all_predictions)} predictions for {symbol}")
                    return True
                else:
                    logger.error(f"Failed to save predictions for {symbol}")
                    return False
            else:
                logger.warning(f"No predictions generated for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {str(e)}")
            return False
    
    def _predict_horizon(self, symbol: str, category: str, X: np.ndarray, y: np.ndarray, 
                        horizon: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate predictions for a specific time horizon.
        
        Args:
            symbol: Stock symbol
            category: Stock category
            X: Feature matrix
            y: Target vector
            horizon: Time horizon (1D, 1W, 1M, 1Y, 5Y)
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        try:
            # Get number of days for this horizon
            horizon_days = self.config.get_time_horizon_days(horizon)
            
            # Get current price for reference
            current_price = df['close'].iloc[-1]
            
            # Get currency
            currency = 'USD' if category == 'us_stocks' else 'INR'
            
            # Train models and get predictions
            model_predictions = {}
            model_accuracies = {}
            
            for model_name, model in self.models.items():
                try:
                    logger.debug(f"Using pre-trained {model_name} for {symbol} {horizon}")
                    
                    # Get prediction (for next day, then extrapolate for longer horizons)
                    if horizon == '1D':
                        # Direct prediction for next day
                        prediction = model.predict(X[-1:].reshape(1, -1))[0]
                    else:
                        # For longer horizons, use iterative prediction or trend extrapolation
                        prediction = self._extrapolate_prediction(
                            model, X, y, horizon_days, current_price
                        )
                    
                    # Calculate model accuracy (on training data)
                    y_pred_train = model.predict(X)
                    accuracy = self._calculate_accuracy(y, y_pred_train)
                    
                    model_predictions[model_name] = prediction
                    model_accuracies[model_name] = accuracy
                    
                    logger.debug(f"{model_name} prediction for {symbol} {horizon}: {prediction:.2f} (accuracy: {accuracy:.4f})")
                    
                except Exception as e:
                    logger.warning(f"Error with {model_name} for {symbol} {horizon}: {str(e)}")
                    continue
            
            if not model_predictions:
                logger.warning(f"No model predictions for {symbol} {horizon}")
                return []
            
            # Create ensemble prediction
            ensemble_prediction = self._create_ensemble_prediction(
                model_predictions, model_accuracies
            )
            
            # Calculate confidence interval
            confidence_low, confidence_high = self._calculate_confidence_interval(
                model_predictions, ensemble_prediction
            )
            
            # Create prediction dictionary
            prediction_dict = self.prediction_saver.create_prediction_dict(
                horizon=horizon,
                predicted_price=ensemble_prediction,
                confidence_low=confidence_low,
                confidence_high=confidence_high,
                algorithm_used='|'.join(model_predictions.keys()),
                currency=currency,
                model_accuracy=np.mean(list(model_accuracies.values())),
                data_points_used=len(X)
            )
            
            predictions.append(prediction_dict)
            
            # Store individual model predictions for analysis
            for model_name, pred in model_predictions.items():
                individual_pred = self.prediction_saver.create_prediction_dict(
                    horizon=horizon,
                    predicted_price=pred,
                    confidence_low=pred * 0.95,  # 5% range
                    confidence_high=pred * 1.05,
                    algorithm_used=model_name,
                    currency=currency,
                    model_accuracy=model_accuracies[model_name],
                    data_points_used=len(X)
                )
                predictions.append(individual_pred)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting {horizon} for {symbol}: {str(e)}")
            return []
    
    def _extrapolate_prediction(self, model: Any, X: np.ndarray, y: np.ndarray, 
                               horizon_days: int, current_price: float) -> float:
        """
        Extrapolate prediction for longer time horizons.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            horizon_days: Number of days to predict ahead
            current_price: Current stock price
            
        Returns:
            Extrapolated prediction
        """
        try:
            # Simple approach: use trend from recent data
            recent_prices = y[-30:]  # Last 30 days
            if len(recent_prices) < 2:
                return current_price
            
            # Calculate trend
            trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
            
            # Extrapolate based on trend
            trend_prediction = current_price + (trend * horizon_days)
            
            # Also get model's next-day prediction
            next_day_pred = model.predict(X[-1:].reshape(1, -1))[0]
            
            # Combine trend and model prediction
            # Weight decreases with horizon length
            trend_weight = min(0.7, horizon_days / 365)  # More weight to trend for longer horizons
            model_weight = 1 - trend_weight
            
            final_prediction = (trend_weight * trend_prediction + 
                              model_weight * next_day_pred)
            
            return final_prediction
            
        except Exception as e:
            logger.warning(f"Error in extrapolation: {str(e)}")
            # Fallback to current price
            return current_price
    
    def _create_ensemble_prediction(self, model_predictions: Dict[str, float], 
                                   model_accuracies: Dict[str, float]) -> float:
        """
        Create ensemble prediction using weighted average.
        
        Args:
            model_predictions: Dictionary of model predictions
            model_accuracies: Dictionary of model accuracies
            
        Returns:
            Ensemble prediction
        """
        try:
            # Use model weights from config, adjusted by accuracy
            total_weight = 0
            weighted_sum = 0
            
            for model_name, prediction in model_predictions.items():
                base_weight = self.config.get_model_weight(model_name)
                accuracy = model_accuracies.get(model_name, 0.5)
                
                # Adjust weight by accuracy
                adjusted_weight = base_weight * (0.5 + accuracy)  # Boost by accuracy
                weighted_sum += prediction * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                # Fallback to simple average
                return np.mean(list(model_predictions.values()))
                
        except Exception as e:
            logger.warning(f"Error creating ensemble prediction: {str(e)}")
            return np.mean(list(model_predictions.values()))
    
    def _calculate_confidence_interval(self, model_predictions: Dict[str, float], 
                                     ensemble_prediction: float) -> Tuple[float, float]:
        """
        Calculate confidence interval for predictions.
        
        Args:
            model_predictions: Dictionary of model predictions
            ensemble_prediction: Ensemble prediction
            
        Returns:
            Tuple of (confidence_low, confidence_high)
        """
        try:
            predictions = list(model_predictions.values())
            
            if len(predictions) < 2:
                # Single prediction - use 10% range
                range_pct = 0.1
                return (ensemble_prediction * (1 - range_pct), 
                       ensemble_prediction * (1 + range_pct))
            
            # Calculate standard deviation
            std_dev = np.std(predictions)
            
            # Use multiplier from config
            multiplier = self.config.CONFIDENCE_MULTIPLIER
            
            confidence_low = ensemble_prediction - (multiplier * std_dev)
            confidence_high = ensemble_prediction + (multiplier * std_dev)
            
            # Ensure positive values
            confidence_low = max(confidence_low, ensemble_prediction * 0.5)
            confidence_high = max(confidence_high, ensemble_prediction * 1.5)
            
            return confidence_low, confidence_high
            
        except Exception as e:
            logger.warning(f"Error calculating confidence interval: {str(e)}")
            # Fallback to 20% range
            range_pct = 0.2
            return (ensemble_prediction * (1 - range_pct), 
                   ensemble_prediction * (1 + range_pct))
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate model accuracy (R² score).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Accuracy score (0-1)
        """
        try:
            from sklearn.metrics import r2_score
            return max(0, r2_score(y_true, y_pred))
        except:
            # Fallback to simple correlation
            try:
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
                return max(0, correlation ** 2)
            except:
                return 0.5  # Default accuracy
    
    def predict_all_stocks(self, category: str = None, max_stocks: int = None) -> Dict[str, Any]:
        """
        Generate predictions for all stocks in a category.
        
        Args:
            category: Stock category ('us_stocks', 'ind_stocks', or None for both)
            max_stocks: Maximum number of stocks to process (for testing)
            
        Returns:
            Dictionary with prediction results summary
        """
        results = {
            'total_stocks': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'failed_symbols': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
        try:
            categories = [category] if category else ['us_stocks', 'ind_stocks']
            
            for cat in categories:
                logger.info(f"Starting predictions for {cat}")
                
                # Get stock symbols
                symbols = self.data_loader.get_stock_symbols(cat)
                
                if max_stocks:
                    symbols = symbols[:max_stocks]
                    logger.info(f"Processing first {max_stocks} stocks for testing")
                
                results['total_stocks'] += len(symbols)
                
                # Process each stock
                for i, symbol in enumerate(symbols, 1):
                    logger.info(f"Processing {symbol} ({i}/{len(symbols)}) in {cat}")
                    
                    try:
                        success = self.predict_stock(symbol, cat)
                        
                        if success:
                            results['successful_predictions'] += 1
                        else:
                            results['failed_predictions'] += 1
                            results['failed_symbols'].append(f"{symbol} ({cat})")
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        results['failed_predictions'] += 1
                        results['failed_symbols'].append(f"{symbol} ({cat})")
            
            results['end_time'] = datetime.now().isoformat()
            
            logger.info(f"Prediction summary: {results['successful_predictions']}/{results['total_stocks']} successful")
            
            if results['failed_symbols']:
                logger.warning(f"Failed symbols: {results['failed_symbols']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in predict_all_stocks: {str(e)}")
            results['end_time'] = datetime.now().isoformat()
            return results
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of all predictions."""
        summary = {}
        
        for category in ['us_stocks', 'ind_stocks']:
            summary[category] = self.prediction_saver.get_prediction_summary(category)
        
        return summary
