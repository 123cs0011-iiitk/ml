#!/usr/bin/env python3
"""
Restore US Stocks from Permanent Index

This script restores the 500 US stocks from the permanent index to the dynamic index.
It preserves any existing stocks in the dynamic index and only adds missing ones.
"""

import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def restore_us_index():
    """Restore US stocks from permanent index to dynamic index"""
    print("=" * 60)
    print("RESTORING US STOCKS FROM PERMANENT INDEX")
    print("=" * 60)
    
    # Paths
    permanent_path = os.path.join('..', 'permanent', 'us_stocks', 'index_us_stocks.csv')
    dynamic_path = os.path.join('..', 'data', 'index_us_stocks_dynamic.csv')
    
    # Check if permanent index exists
    if not os.path.exists(permanent_path):
        print(f"❌ Permanent US index not found: {permanent_path}")
        return False
    
    # Read permanent US index (500 stocks)
    print(f"📖 Reading permanent US index: {permanent_path}")
    df_permanent = pd.read_csv(permanent_path)
    print(f"   Found {len(df_permanent)} stocks in permanent index")
    
    # Read current dynamic index
    if os.path.exists(dynamic_path):
        print(f"📖 Reading current dynamic index: {dynamic_path}")
        df_dynamic = pd.read_csv(dynamic_path)
        print(f"   Found {len(df_dynamic)} stocks in dynamic index")
    else:
        print("📝 No dynamic index found, creating new one")
        df_dynamic = pd.DataFrame()
    
    # Add currency column to permanent data if not present
    if 'currency' not in df_permanent.columns:
        df_permanent['currency'] = 'USD'
        print("   Added currency column to permanent data")
    
    # Ensure dynamic index has all required columns
    required_columns = ['symbol', 'company_name', 'sector', 'market_cap', 'headquarters', 'exchange', 'currency']
    for col in required_columns:
        if col not in df_dynamic.columns:
            df_dynamic[col] = ''
    
    # Find missing stocks (in permanent but not in dynamic)
    existing_symbols = set(df_dynamic['symbol'].str.upper()) if not df_dynamic.empty else set()
    missing_stocks = df_permanent[~df_permanent['symbol'].str.upper().isin(existing_symbols)]
    
    print(f"\n📊 Analysis:")
    print(f"   Existing in dynamic: {len(existing_symbols)}")
    print(f"   Missing from dynamic: {len(missing_stocks)}")
    print(f"   Total after restore: {len(existing_symbols) + len(missing_stocks)}")
    
    if len(missing_stocks) == 0:
        print("✅ All stocks already present in dynamic index")
        return True
    
    # Combine existing and missing stocks
    print(f"\n🔄 Restoring {len(missing_stocks)} missing stocks...")
    df_combined = pd.concat([df_dynamic, missing_stocks], ignore_index=True)
    
    # Sort alphabetically by symbol
    df_combined = df_combined.sort_values('symbol').reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(dynamic_path), exist_ok=True)
    
    # Save combined index
    df_combined.to_csv(dynamic_path, index=False)
    
    print(f"✅ Successfully restored {len(missing_stocks)} US stocks")
    print(f"   Total stocks in dynamic index: {len(df_combined)}")
    print(f"   Saved to: {dynamic_path}")
    
    # Show some examples
    print(f"\n📋 Sample restored stocks:")
    for i, (_, row) in enumerate(missing_stocks.head(5).iterrows()):
        print(f"   {i+1}. {row['symbol']} - {row['company_name']}")
    if len(missing_stocks) > 5:
        print(f"   ... and {len(missing_stocks) - 5} more")
    
    return True

if __name__ == "__main__":
    try:
        success = restore_us_index()
        if success:
            print("\n🎉 US index restoration completed successfully!")
        else:
            print("\n❌ US index restoration failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during restoration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
