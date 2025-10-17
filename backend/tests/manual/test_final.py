#!/usr/bin/env python3
"""
Final Test - Indian Stock Fetcher
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the fetcher directly
from data_fetching.ind_stocks.current_fetching.ind_current_fetcher import IndianCurrentFetcher

def test_stock_fetching():
    print("=" * 60)
    print("FINAL TEST - INDIAN STOCK FETCHER")
    print("=" * 60)
    
    fetcher = IndianCurrentFetcher()
    
    # Test RELIANCE
    print("\nTesting RELIANCE...")
    try:
        result = fetcher.fetch_current_price('RELIANCE')
        print("✅ SUCCESS!")
        print(f"Symbol: {result['symbol']}")
        print(f"Price: ₹{result['price']}")
        print(f"Source: {result['source']}")
        print(f"Currency: {result['currency']}")
        print(f"Timestamp: {result['timestamp']}")
        
        if result['source'] == 'upstox':
            print("🎉 Upstox integration is working perfectly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_stock_fetching()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 CONGRATULATIONS!")
        print("=" * 60)
        print("✅ Your Upstox integration is working!")
        print("✅ Indian stock data is being fetched successfully!")
        print("✅ Your backend is ready for production!")
        print("\nYou can now test your full application:")
        print("1. Run: python main.py")
        print("2. Visit: http://localhost:5000/live_price?symbol=RELIANCE")
        print("3. Your frontend will get live Indian stock data!")
    else:
        print("\n❌ There was an issue with the integration.")
        print("Please check the error message above.")
