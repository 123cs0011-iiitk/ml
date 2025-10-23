"""
Test script to verify data normalization is working correctly.
"""
import sys
import os

# Add backend to path so we can import modules
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from prediction.data_loader import DataLoader
import numpy as np

def test_normalization():
    print("="*80)
    print("TESTING DATA NORMALIZATION")
    print("="*80)
    
    data_loader = DataLoader()
    
    # Test RELIANCE (Indian stock)
    print("\n--- Testing RELIANCE (Indian stock) ---")
    reliance_df = data_loader.load_stock_data("RELIANCE", "ind_stocks")
    if reliance_df is not None:
        print(f"Shape: {reliance_df.shape}")
        print(f"Close price range: ${reliance_df['close'].min():.2f} to ${reliance_df['close'].max():.2f}")
        print(f"Close price mean: ${reliance_df['close'].mean():.2f}")
        print(f"Sample close prices (first 5): {reliance_df['close'].head().tolist()}")
        
        # Prepare training data
        reliance_df['market_type'] = 1  # Indian stock
        df_with_features = data_loader.create_features(reliance_df)
        X, y = data_loader.prepare_training_data(df_with_features)
        
        print(f"\nTraining data:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  y range: {y.min():.4f} to {y.max():.4f}")
        print(f"  y mean: {y.mean():.4f}")
        print(f"  y std: {y.std():.4f}")
        print(f"  Sample y values (first 10): {y[:10].tolist()}")
    else:
        print("ERROR: Could not load RELIANCE data")
    
    # Test AAPL (US stock)
    print("\n--- Testing AAPL (US stock) ---")
    aapl_df = data_loader.load_stock_data("AAPL", "us_stocks")
    if aapl_df is not None:
        print(f"Shape: {aapl_df.shape}")
        print(f"Close price range: ${aapl_df['close'].min():.2f} to ${aapl_df['close'].max():.2f}")
        print(f"Close price mean: ${aapl_df['close'].mean():.2f}")
        print(f"Sample close prices (first 5): {aapl_df['close'].head().tolist()}")
        
        # Prepare training data
        aapl_df['market_type'] = 0  # US stock
        df_with_features = data_loader.create_features(aapl_df)
        X, y = data_loader.prepare_training_data(df_with_features)
        
        print(f"\nTraining data:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  y range: {y.min():.4f} to {y.max():.4f}")
        print(f"  y mean: {y.mean():.4f}")
        print(f"  y std: {y.std():.4f}")
        print(f"  Sample y values (first 10): {y[:10].tolist()}")
    else:
        print("ERROR: Could not load AAPL data")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_normalization()

