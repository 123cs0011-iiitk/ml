#!/usr/bin/env python3
"""
Test script to verify Upstox API credentials and connectivity
"""

import os
import sys
from dotenv import load_dotenv
import requests

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

def test_upstox_connection():
    api_key = os.getenv('UPSTOX_API_KEY')
    access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
    
    print("=" * 60)
    print("UPSTOX API CONNECTION TEST")
    print("=" * 60)
    
    # Check if credentials exist
    print(f"\n1. Checking credentials...")
    print(f"   API Key: {'✓ Found' if api_key else '✗ Missing'}")
    print(f"   Access Token: {'✓ Found' if access_token else '✗ Missing'}")
    
    if not access_token:
        print("\n❌ Access token not found in .env file!")
        print("   Please add UPSTOX_ACCESS_TOKEN to your .env file")
        return
    
    # Test API call
    print(f"\n2. Testing Upstox API connection...")
    
    try:
        url = "https://api.upstox.com/v2/market-quote/ltp"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        # Test with RELIANCE stock
        params = {'symbol': 'NSE_EQ|INE002A01018'}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data}")
            if data.get('status') == 'success':
                print("\n✓ Upstox API is working correctly!")
                return True
            else:
                print(f"\n✗ API returned error: {data.get('message')}")
                return False
        elif response.status_code == 401:
            print("\n✗ Authentication failed - Invalid access token")
            return False
        elif response.status_code == 403:
            print("\n✗ Access forbidden - API not enabled in Upstox dashboard")
            return False
        else:
            print(f"\n✗ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n✗ Connection error: {str(e)}")
        return False

if __name__ == '__main__':
    test_upstox_connection()
