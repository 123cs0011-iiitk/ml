"""
Test script to verify model predictions on normalized data.
"""
import sys
import os
import joblib
import numpy as np

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from prediction.data_loader import DataLoader

def test_model_predictions():
    print("="*80)
    print("TESTING MODEL PREDICTIONS")
    print("="*80)
    
    # Load the trained model
    model_path = "backend/models/linear_regression/linear_regression_model.pkl"
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"Model loaded: {model}")
    print(f"Model has scaler: {hasattr(model, 'scaler') and model.scaler is not None}")
    
    # Load data
    data_loader = DataLoader()
    
    # Test RELIANCE
    print("\n--- Testing RELIANCE ---")
    reliance_df = data_loader.load_stock_data("RELIANCE", "ind_stocks")
    if reliance_df is not None:
        reliance_df['market_type'] = 1
        df_with_features = data_loader.create_features(reliance_df)
        X, y = data_loader.prepare_training_data(df_with_features)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"y stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
        
        # Make predictions
        y_pred = model.predict(X[:100])  # Test on first 100 samples
        print(f"\nPredictions (first 100 samples):")
        print(f"  y_pred stats: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")
        print(f"  Sample predictions: {y_pred[:10].tolist()}")
        print(f"  Sample actuals:     {y[:10].tolist()}")
        
        # Calculate error
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y[:100], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y[:100], y_pred)
        print(f"\nMetrics on first 100 samples:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    # Test AAPL
    print("\n--- Testing AAPL ---")
    aapl_df = data_loader.load_stock_data("AAPL", "us_stocks")
    if aapl_df is not None:
        aapl_df['market_type'] = 0
        df_with_features = data_loader.create_features(aapl_df)
        X, y = data_loader.prepare_training_data(df_with_features)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"y stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
        
        # Make predictions
        y_pred = model.predict(X[:100])
        print(f"\nPredictions (first 100 samples):")
        print(f"  y_pred stats: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")
        print(f"  Sample predictions: {y_pred[:10].tolist()}")
        print(f"  Sample actuals:     {y[:10].tolist()}")
        
        # Calculate error
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y[:100], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y[:100], y_pred)
        print(f"\nMetrics on first 100 samples:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_model_predictions()

