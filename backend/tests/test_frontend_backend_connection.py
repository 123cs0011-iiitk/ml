#!/usr/bin/env python3
"""
Test script to verify frontend-backend connection
"""

import requests
import json

def test_cors_connection():
    """Test CORS connection from frontend to backend"""
    print("🌐 Testing CORS Connection")
    print("=" * 30)
    
    # Test search endpoint with CORS headers
    try:
        response = requests.get(
            "http://localhost:5000/search?q=tcs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        print(f"Status: {response.status_code}")
        print(f"CORS Headers:")
        for header, value in response.headers.items():
            if 'cors' in header.lower() or 'origin' in header.lower():
                print(f"  {header}: {value}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Data: {data}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_live_price_connection():
    """Test live price endpoint with CORS headers"""
    print("\n💰 Testing Live Price Connection")
    print("=" * 30)
    
    try:
        response = requests.get(
            "http://localhost:5000/live_price?symbol=TCS",
            headers={
                "Origin": "http://localhost:3000"
            }
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Data: {data}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_health_connection():
    """Test health endpoint"""
    print("\n🏥 Testing Health Connection")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:5000/health")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Data: {data}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("🔗 Frontend-Backend Connection Test")
    print("=" * 40)
    
    health_ok = test_health_connection()
    search_ok = test_cors_connection()
    price_ok = test_live_price_connection()
    
    print("\n" + "=" * 40)
    print("📋 Test Results:")
    print(f"Health Check: {'✅' if health_ok else '❌'}")
    print(f"Search API: {'✅' if search_ok else '❌'}")
    print(f"Live Price API: {'✅' if price_ok else '❌'}")
    
    if all([health_ok, search_ok, price_ok]):
        print("\n🎉 All connections working! Frontend should be able to connect to backend.")
    else:
        print("\n⚠️ Some connections failed. Check the errors above.")
    
    print("\n💡 If the frontend is still not working, check:")
    print("1. Browser developer console for JavaScript errors")
    print("2. Network tab for failed requests")
    print("3. Make sure both servers are running")

if __name__ == "__main__":
    main()
