#!/usr/bin/env python3
"""
Test ISIN storage in dynamic index
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-fetching', 'ind_stocks', 'current-fetching'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-fetching'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))

# Load environment variables
load_dotenv()

def test_isin_storage():
    """Test ISIN storage in dynamic index"""
    print("=" * 60)
    print("ISIN STORAGE TEST")
    print("=" * 60)
    
    try:
        # Import with absolute path to avoid relative import issues
        import ind_current_fetcher
        IndianCurrentFetcher = ind_current_fetcher.IndianCurrentFetcher
        
        # Initialize fetcher
        fetcher = IndianCurrentFetcher()
        
        # Test with a stock that should get ISIN from instruments file
        print("\n1. Testing TITAN (should get ISIN from instruments file):")
        result = fetcher.fetch_current_price('TITAN')
        print(f"   Result: {result['source']} - ₹{result['price']}")
        
        # Check if ISIN was saved to dynamic index
        print("\n2. Checking if ISIN was saved to dynamic index:")
        from shared.index_manager import DynamicIndexManager
        index_manager = DynamicIndexManager(fetcher.data_dir)
        isin = index_manager.get_isin('TITAN', 'ind_stocks')
        if isin:
            print(f"   ✓ ISIN found in dynamic index: {isin}")
        else:
            print("   ⚠ No ISIN found in dynamic index")
        
        # Test second lookup (should use saved ISIN)
        print("\n3. Testing second lookup (should use saved ISIN):")
        result2 = fetcher.fetch_current_price('TITAN')
        print(f"   Result: {result2['source']} - ₹{result2['price']}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_isin_storage()
