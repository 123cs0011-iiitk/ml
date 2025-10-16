#!/usr/bin/env python3
"""
Populate ISINs for Indian Stocks

This script populates ISIN codes for all Indian stocks in the dynamic index
using the Upstox instruments file.
"""

import pandas as pd
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-fetching', 'ind_stocks', 'current-fetching'))

def populate_indian_isins():
    """Populate ISIN codes for all Indian stocks"""
    print("=" * 60)
    print("POPULATING ISINs FOR INDIAN STOCKS")
    print("=" * 60)
    
    # Paths
    dynamic_path = os.path.join('..', 'data', 'index_ind_stocks_dynamic.csv')
    
    # Check if dynamic index exists
    if not os.path.exists(dynamic_path):
        print(f"❌ Indian dynamic index not found: {dynamic_path}")
        return False
    
    # Read Indian dynamic index
    print(f"📖 Reading Indian dynamic index: {dynamic_path}")
    df = pd.read_csv(dynamic_path)
    print(f"   Found {len(df)} stocks in dynamic index")
    
    # Check if ISIN column exists
    if 'isin' not in df.columns:
        print("   Adding ISIN column...")
        df['isin'] = ''
    
    # Initialize instruments fetcher
    print("\n🔧 Initializing Upstox instruments fetcher...")
    try:
        from upstox_instruments import get_instruments_fetcher
        fetcher = get_instruments_fetcher()
        print("   ✅ Instruments fetcher initialized")
        
        # Test if instruments file can be downloaded
        print("   🧪 Testing instruments file download...")
        try:
            test_instrument = fetcher.get_instrument_key('RELIANCE')
            if test_instrument:
                print(f"   ✅ Test successful: RELIANCE -> {test_instrument}")
            else:
                print("   ⚠️ Test returned None for RELIANCE")
        except Exception as test_e:
            print(f"   ❌ Test failed: {test_e}")
            print("   🛑 Stopping ISIN population due to instruments file issues")
            return False
            
    except Exception as e:
        print(f"   ❌ Failed to initialize instruments fetcher: {e}")
        return False
    
    # Count stocks that need ISIN population
    stocks_needing_isin = 0
    stocks_with_isin = 0
    
    for idx, row in df.iterrows():
        current_isin = str(row.get('isin', ''))
        if current_isin and current_isin != 'nan' and len(current_isin.strip()) > 0:
            stocks_with_isin += 1
        else:
            stocks_needing_isin += 1
    
    print(f"\n📊 ISIN Status:")
    print(f"   Stocks with ISIN: {stocks_with_isin}")
    print(f"   Stocks needing ISIN: {stocks_needing_isin}")
    
    if stocks_needing_isin == 0:
        print("✅ All stocks already have ISIN codes")
        return True
    
    # Populate ISINs
    print(f"\n🔄 Populating ISINs for {stocks_needing_isin} stocks...")
    updated_count = 0
    failed_count = 0
    
    for idx, row in df.iterrows():
        symbol = row['symbol']
        current_isin = str(row.get('isin', ''))
        
        # Skip if ISIN already populated
        if current_isin and current_isin != 'nan' and len(current_isin.strip()) > 0:
            continue
        
        # Lookup ISIN
        try:
            print(f"   Looking up {symbol}...", end=' ')
            instrument_key = fetcher.get_instrument_key(symbol)
            if instrument_key:
                isin = instrument_key.split('|')[1]
                df.at[idx, 'isin'] = isin
                updated_count += 1
                print(f"✅ {isin}")
            else:
                print("❌ No ISIN found")
                failed_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_count += 1
        
        # Rate limiting to avoid overwhelming the API
        time.sleep(0.1)
        
        # Progress update every 50 stocks
        if (updated_count + failed_count) % 50 == 0:
            print(f"   Progress: {updated_count + failed_count}/{stocks_needing_isin}")
    
    # Save updated index
    print(f"\n💾 Saving updated index...")
    df.to_csv(dynamic_path, index=False)
    
    print(f"\n📊 Results:")
    print(f"   ✅ Successfully populated: {updated_count} stocks")
    print(f"   ❌ Failed to populate: {failed_count} stocks")
    print(f"   📁 Saved to: {dynamic_path}")
    
    # Show some examples of populated ISINs
    populated_stocks = df[df['isin'].str.len() > 0].head(10)
    if not populated_stocks.empty:
        print(f"\n📋 Sample populated ISINs:")
        for _, row in populated_stocks.iterrows():
            print(f"   {row['symbol']}: {row['isin']}")
    
    return True

if __name__ == "__main__":
    try:
        success = populate_indian_isins()
        if success:
            print("\n🎉 ISIN population completed successfully!")
        else:
            print("\n❌ ISIN population failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during ISIN population: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
