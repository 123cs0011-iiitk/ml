#!/usr/bin/env python3
"""
Test script for currency conversion display
"""

import requests
import json

def test_currency_display():
    print('🧪 Testing Complete System with Currency Conversion')
    print('=' * 55)

    # Test TCS (Indian stock)
    print('\n🇮🇳 Testing TCS (Indian stock):')
    response = requests.get('http://localhost:5000/live_price?symbol=TCS')
    if response.status_code == 200:
        data = response.json()
        print('✅ TCS Response:')
        print(f'  Price: ₹{data["data"]["price"]:.2f}')
        print(f'  Currency: {data["data"]["currency"]}')
        if "exchange_rate" in data["data"]:
            print(f'  Exchange Rate: {data["data"]["exchange_rate"]:.4f} USD/INR')
            print(f'  Exchange Source: {data["data"]["exchange_source"]}')
        if "price_usd" in data["data"]:
            print(f'  Price in USD: ${data["data"]["price_usd"]:.2f}')
    else:
        print(f'❌ TCS Error: {response.status_code}')

    # Test AAPL (US stock)
    print('\n🇺🇸 Testing AAPL (US stock):')
    response = requests.get('http://localhost:5000/live_price?symbol=AAPL')
    if response.status_code == 200:
        data = response.json()
        print('✅ AAPL Response:')
        print(f'  Price: ${data["data"]["price"]:.2f}')
        print(f'  Currency: {data["data"]["currency"]}')
        if "exchange_rate" in data["data"]:
            print(f'  Exchange Rate: {data["data"]["exchange_rate"]:.4f} USD/INR')
            print(f'  Exchange Source: {data["data"]["exchange_source"]}')
        if "price_inr" in data["data"]:
            print(f'  Price in INR: ₹{data["data"]["price_inr"]:.2f}')
    else:
        print(f'❌ AAPL Error: {response.status_code}')

    print('\n🎉 Currency conversion testing completed!')

if __name__ == "__main__":
    test_currency_display()
