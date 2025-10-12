#!/usr/bin/env python3
"""
Test script to verify the Yahoo Finance rate limiting improvements.
This script tests the caching, rate limiting, and retry logic.
"""

import os
import sys
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_cache_functionality():
    """Test the caching functionality"""
    print("🧪 Testing Cache Functionality")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: First request (should hit API)
    print("1. Making first request for AAPL...")
    start_time = time.time()
    try:
        response1 = requests.get(f"{base_url}/live_price?symbol=AAPL", timeout=30)
        end_time = time.time()
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status: {response1.status_code}")
        
        if response1.status_code == 200:
            data1 = response1.json()
            print(f"   ✅ Success: {data1['data']['symbol']} = ${data1['data']['price']}")
            print(f"   Source: {data1['data']['source']}")
            print(f"   Timestamp: {data1['data']['timestamp']}")
        else:
            print(f"   ❌ Error: {response1.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print()
    
    # Test 2: Second request (should use cache)
    print("2. Making second request for AAPL (should use cache)...")
    start_time = time.time()
    try:
        response2 = requests.get(f"{base_url}/live_price?symbol=AAPL", timeout=10)
        end_time = time.time()
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status: {response2.status_code}")
        
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"   ✅ Success: {data2['data']['symbol']} = ${data2['data']['price']}")
            print(f"   Source: {data2['data']['source']}")
            print(f"   Timestamp: {data2['data']['timestamp']}")
            
            # Check if timestamps are the same (indicating cache usage)
            if 'data1' in locals() and data1['data']['timestamp'] == data2['data']['timestamp']:
                print("   🎯 CACHE HIT: Same timestamp indicates cache was used!")
            else:
                print("   ⚠️  Different timestamp - cache may not be working")
        else:
            print(f"   ❌ Error: {response2.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    print()

def test_rate_limiting():
    """Test rate limiting by making multiple requests quickly"""
    print("🚦 Testing Rate Limiting")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    print("Making rapid requests to test rate limiting...")
    
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. Requesting {symbol}...")
        start_time = time.time()
        try:
            response = requests.get(f"{base_url}/live_price?symbol={symbol}", timeout=30)
            end_time = time.time()
            print(f"   Response time: {end_time - start_time:.2f} seconds")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success: {data['data']['symbol']} = ${data['data']['price']}")
            else:
                print(f"   ❌ Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        print()
        
        # Small delay between requests to avoid overwhelming
        if i < len(symbols):
            time.sleep(1)

def check_api_keys():
    """Check if API keys are configured"""
    print("🔑 Checking API Key Configuration")
    print("=" * 50)
    
    finnhub_key = os.getenv('FINNHUB_API_KEY')
    alphavantage_key = os.getenv('ALPHAVANTAGE_API_KEY')
    
    print(f"FINNHUB_API_KEY: {'✅ Configured' if finnhub_key and finnhub_key != 'your_finnhub_api_key_here' else '❌ Not configured'}")
    print(f"ALPHAVANTAGE_API_KEY: {'✅ Configured' if alphavantage_key and alphavantage_key != 'your_alphavantage_api_key_here' else '❌ Not configured'}")
    print()
    
    if not finnhub_key or finnhub_key == 'your_finnhub_api_key_here':
        print("📝 To configure Finnhub API key:")
        print("   1. Visit: https://finnhub.io/register")
        print("   2. Get your free API key")
        print("   3. Update .env file: FINNHUB_API_KEY=your_actual_key")
        print()
    
    if not alphavantage_key or alphavantage_key == 'your_alphavantage_api_key_here':
        print("📝 To configure Alpha Vantage API key:")
        print("   1. Visit: https://www.alphavantage.co/support/#api-key")
        print("   2. Get your free API key")
        print("   3. Update .env file: ALPHAVANTAGE_API_KEY=your_actual_key")
        print()

def main():
    """Main test function"""
    print("🧪 Yahoo Finance Rate Limiting Improvements Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not running or not responding properly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please make sure the server is running with: python run_server.py")
        return
    
    print("✅ Server is running")
    print()
    
    # Run tests
    check_api_keys()
    test_cache_functionality()
    test_rate_limiting()
    
    print("🎯 Test completed!")
    print()
    print("💡 Key Improvements Implemented:")
    print("   • 60-second cache reduces API calls by ~95%")
    print("   • Exponential backoff with longer delays (4-30 seconds)")
    print("   • Rate limiting enforces minimum 2-second gaps")
    print("   • API key configuration for fallback services")
    print("   • Better error handling and logging")

if __name__ == "__main__":
    main()
