#!/usr/bin/env python3
"""
Test script to debug frontend-backend connection
"""

import requests
import json

def test_search_api():
    """Test the search API directly"""
    print("🔍 Testing Search API")
    print("=" * 30)
    
    base_url = "http://localhost:5000"
    
    # Test different search queries
    test_queries = ["tcs", "TCS", "reliance", "RELIANCE", "aapl", "AAPL"]
    
    for query in test_queries:
        try:
            print(f"\n🔍 Testing query: '{query}'")
            response = requests.get(f"{base_url}/search?q={query}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"✅ Success: {data.get('success')}")
                print(f"✅ Data: {data.get('data')}")
                print(f"✅ Results count: {len(data.get('data', []))}")
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"❌ Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def test_cors():
    """Test CORS headers"""
    print("\n🌐 Testing CORS")
    print("=" * 30)
    
    try:
        response = requests.get(
            "http://localhost:5000/search?q=tcs",
            headers={"Origin": "http://localhost:3000"}
        )
        
        print(f"Status: {response.status_code}")
        print(f"CORS Headers:")
        for header, value in response.headers.items():
            if 'cors' in header.lower() or 'origin' in header.lower():
                print(f"  {header}: {value}")
                
    except Exception as e:
        print(f"❌ CORS Error: {e}")

def test_health():
    """Test health endpoint"""
    print("\n🏥 Testing Health")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:5000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health Error: {e}")

if __name__ == "__main__":
    test_health()
    test_search_api()
    test_cors()
    print("\n✅ All tests completed!")
