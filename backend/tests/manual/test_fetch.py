#!/usr/bin/env python3
"""
Test script to debug the on-demand fetching functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from main import fetch_recent_data_if_needed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data - simulate what we currently have
test_data = [
    {
        'date': '2025-10-13',
        'open': 187.965,
        'high': 190.1099,
        'low': 185.96,
        'close': 188.78,
        'volume': 0
    }
]

print("Testing fetch_recent_data_if_needed function...")
print(f"Input data: {len(test_data)} points")
print(f"First point: {test_data[0]}")

# Test with week period
result = fetch_recent_data_if_needed('NVDA', 'us_stocks', 'week', test_data)
print(f"Result: {len(result)} additional points")
if result:
    print(f"First additional point: {result[0]}")
    print(f"Last additional point: {result[-1]}")
