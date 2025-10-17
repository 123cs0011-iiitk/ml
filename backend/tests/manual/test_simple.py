#!/usr/bin/env python3
"""
Simple Upstox API test
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv('UPSTOX_ACCESS_TOKEN')

print("=" * 50)
print("SIMPLE UPSTOX API TEST")
print("=" * 50)

print(f"Token: {access_token[:30]}..." if access_token else "No token found")

if not access_token:
    print("❌ No access token found in .env")
    exit(1)

url = 'https://api.upstox.com/v2/market-quote/ltp'
headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
params = {'symbol': 'NSE_EQ|INE002A01018'}  # RELIANCE

try:
    print("Making API call...")
    response = requests.get(url, headers=headers, params=params, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data}")
        
        if data.get('status') == 'success':
            price_data = data['data']['NSE_EQ|INE002A01018']
            price = price_data['last_price']
            print(f"✅ SUCCESS! RELIANCE price: ₹{price}")
        else:
            print(f"❌ API Error: {data.get('message', 'Unknown error')}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        print(f"Response: {response.text[:300]}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("=" * 50)
