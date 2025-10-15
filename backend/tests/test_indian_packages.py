#!/usr/bin/env python3
"""
Test script for Indian stock data packages
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-fetching'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-fetching', 'ind_stocks', 'current-fetching'))

from ind_current_fetcher import IndianCurrentFetcher
from shared.utilities import categorize_stock

def test_package_availability():
    """Test which packages are available"""
    print("🔍 Testing package availability...")
    
    packages = {
        'yfinance': False,
        'stock-market-india': False,
        'nsepython': False,
        'nselib': False,
        'alpha_vantage': False,
        'india_stocks_api': False
    }
    
    # Test yfinance
    try:
        import yfinance as yf
        packages['yfinance'] = True
        print("✅ yfinance: Available")
    except ImportError:
        print("❌ yfinance: Not available")
    
    # Test stock-market-india
    try:
        from stock_market_india import StockMarketIndia
        packages['stock-market-india'] = True
        print("✅ stock-market-india: Available")
    except ImportError:
        print("❌ stock-market-india: Not available")
    
    # Test nsepython
    try:
        import nsepython
        packages['nsepython'] = True
        print("✅ nsepython: Available")
    except ImportError:
        print("❌ nsepython: Not available")
    
    # Test nselib
    try:
        import nselib
        packages['nselib'] = True
        print("✅ nselib: Available")
    except ImportError:
        print("❌ nselib: Not available")
    
    # Test alpha_vantage
    try:
        from alpha_vantage.timeseries import TimeSeries
        packages['alpha_vantage'] = True
        print("✅ alpha_vantage: Available")
    except ImportError:
        print("❌ alpha_vantage: Not available")
    
    # Test india_stocks_api
    try:
        from india_stocks_api.angelone import AngelOne
        packages['india_stocks_api'] = True
        print("✅ india_stocks_api: Available")
    except ImportError:
        print("❌ india_stocks_api: Not available")
    
    return packages

def test_direct_package_usage():
    """Test direct usage of available packages"""
    print("\n🧪 Testing direct package usage...")
    
    # Test yfinance
    try:
        import yfinance as yf
        print("\n📊 Testing yfinance...")
        tcs = yf.Ticker("TCS.NS")
        hist = tcs.history(period="1d")
        if not hist.empty:
            print(f"✅ yfinance: Got data for TCS.NS (shape: {hist.shape})")
        else:
            print("❌ yfinance: No data returned")
    except Exception as e:
        print(f"❌ yfinance: Error - {e}")
    
    # Test stock-market-india
    try:
        from stock_market_india import StockMarketIndia
        print("\n📊 Testing stock-market-india...")
        smi = StockMarketIndia()
        quote = smi.get_quote('TCS')
        if quote and 'lastPrice' in quote:
            print(f"✅ stock-market-india: Got quote for TCS - ₹{quote['lastPrice']}")
        else:
            print("❌ stock-market-india: No data returned")
    except Exception as e:
        print(f"❌ stock-market-india: Error - {e}")

def test_fallback_chain():
    """Test the complete fallback chain"""
    print("\n🔄 Testing complete fallback chain...")
    
    fetcher = IndianCurrentFetcher()
    test_symbols = ["TCS", "RELIANCE", "INFY", "HDFCBANK"]
    
    for symbol in test_symbols:
        try:
            print(f"\n🔍 Testing {symbol}...")
            result = fetcher.fetch_current_price(symbol)
            print(f"✅ {symbol}: ₹{result['price']} (source: {result['source']})")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")

def test_search_functionality():
    """Test search functionality"""
    print("\n🔍 Testing search functionality...")
    
    # Test categorization
    test_symbols = ["TCS", "AAPL", "RELIANCE", "MSFT"]
    for symbol in test_symbols:
        category = categorize_stock(symbol)
        print(f"📋 {symbol}: {category}")

def main():
    print("🏦 Indian Stock Data Packages Test Suite")
    print("=" * 50)
    
    # Test package availability
    packages = test_package_availability()
    
    # Test direct usage
    test_direct_package_usage()
    
    # Test fallback chain
    test_fallback_chain()
    
    # Test search functionality
    test_search_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Summary:")
    
    available_packages = [name for name, available in packages.items() if available]
    unavailable_packages = [name for name, available in packages.items() if not available]
    
    if available_packages:
        print(f"✅ Available packages: {', '.join(available_packages)}")
    
    if unavailable_packages:
        print(f"❌ Missing packages: {', '.join(unavailable_packages)}")
        print("💡 Install missing packages with:")
        for package in unavailable_packages:
            if package == 'india_stocks_api':
                print(f"   pip install india-stocks-api")
            else:
                print(f"   pip install {package}")
    
    print("\n🚀 Recommendations:")
    print("1. Install stock-market-india for reliable real-time data")
    print("2. Keep yfinance for historical data")
    print("3. Consider nsepython for additional NSE data")
    print("4. Use india-stocks-api if you have broker accounts")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()
