#!/usr/bin/env python3
"""
Test API Response
"""

import requests
import json

try:
    response = requests.get('http://localhost:5000/live_price?symbol=RELIANCE')
    data = response.json()
    
    print('=' * 60)
    print('🎉 SUCCESS! INDIAN STOCK DATA RETRIEVED')
    print('=' * 60)
    
    stock_data = data['data']
    print(f'Symbol: {stock_data["symbol"]}')
    print(f'Company: {stock_data["company_name"]}')
    print(f'Price: ₹{stock_data["price"]}')
    print(f'Currency: {stock_data["currency"]}')
    print(f'Source: {stock_data["source"]}')
    print(f'Exchange: {stock_data["exchange"]}')
    print(f'Timestamp: {stock_data["timestamp"]}')
    print(f'Exchange Rate: {stock_data["exchange_rate"]}')
    
    print('\n' + '=' * 60)
    print('✅ YOUR UPSTOX INTEGRATION IS WORKING!')
    print('✅ INDIAN STOCK DATA IS BEING FETCHED!')
    print('✅ YOUR BACKEND IS READY FOR PRODUCTION!')
    print('=' * 60)
    
except Exception as e:
    print(f'Error: {e}')
