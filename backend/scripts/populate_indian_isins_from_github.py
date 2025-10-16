#!/usr/bin/env python3
"""
Populate ISINs for Indian Stocks from GitHub Data

This script populates ISIN codes for all Indian stocks in the dynamic index
using the india-isin-data repository from GitHub.
"""

import pandas as pd
import os
import sys
import requests
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def download_isin_data():
    """Download ISIN data from GitHub repository"""
    print("📥 Downloading ISIN data from GitHub...")
    
    # URL to the raw ISIN.csv file
    url = "https://raw.githubusercontent.com/captn3m0/india-isin-data/main/ISIN.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = os.path.join(os.path.dirname(__file__), 'temp_isin_data.csv')
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"   ✅ Downloaded {len(response.text)} characters")
        return temp_file
        
    except Exception as e:
        print(f"   ❌ Failed to download ISIN data: {e}")
        return None

def load_isin_mapping(isin_file):
    """Load ISIN data and create symbol to ISIN mapping"""
    print("📖 Loading ISIN data...")
    
    try:
        # Read the ISIN CSV
        df_isin = pd.read_csv(isin_file)
        print(f"   Found {len(df_isin)} ISIN records")
        
        # Create symbol to ISIN mapping
        # The CSV has columns: ISIN, Security Name, etc.
        symbol_to_isin = {}
        
        for _, row in df_isin.iterrows():
            isin = str(row.get('ISIN', '')).strip()
            description = str(row.get('Description', '')).strip()
            issuer = str(row.get('Issuer', '')).strip()
            security_type = str(row.get('Type', '')).strip()
            status = str(row.get('Status', '')).strip()
            
            # Only process active equity shares (INE prefix)
            if (isin and isin.startswith('INE') and len(isin) == 12 and 
                status == 'ACTIVE' and security_type == 'EQUITY SHARES'):
                
                # Extract symbol from description or issuer
                symbol = None
                
                # Try to extract from issuer first (usually cleaner)
                if issuer and issuer != 'nan':
                    # Clean issuer name
                    symbol = issuer.replace(' LTD', '').replace(' LIMITED', '').replace(' CORP', '').replace(' CORPORATION', '')
                    symbol = symbol.replace(' INC', '').replace(' INCORPORATED', '').replace(' CO', '').replace(' COMPANY', '')
                    symbol = symbol.strip().upper()
                elif description and description != 'nan':
                    # Try to extract from description
                    # Look for patterns like "SYMBOL - COMPANY NAME" or "SYMBOL COMPANY NAME"
                    if ' - ' in description:
                        symbol = description.split(' - ')[0].strip().upper()
                    elif ' ' in description:
                        parts = description.split()
                        if parts and parts[0].isupper() and len(parts[0]) <= 20:
                            symbol = parts[0].strip().upper()
                
                # Clean and validate symbol
                if symbol:
                    symbol = symbol.replace(' LTD', '').replace(' LIMITED', '').replace(' CORP', '').replace(' CORPORATION', '')
                    symbol = symbol.replace(' INC', '').replace(' INCORPORATED', '').replace(' CO', '').replace(' COMPANY', '')
                    symbol = symbol.strip()
                    
                    # Only add if symbol is reasonable length and contains letters
                    if (symbol and len(symbol) <= 20 and len(symbol) >= 2 and 
                        any(c.isalpha() for c in symbol)):
                        symbol_to_isin[symbol] = isin
        
        print(f"   Created mapping for {len(symbol_to_isin)} symbols")
        
        # Show some examples
        print("   📋 Sample mappings:")
        for i, (symbol, isin) in enumerate(list(symbol_to_isin.items())[:5]):
            print(f"      {symbol} -> {isin}")
        
        return symbol_to_isin
        
    except Exception as e:
        print(f"   ❌ Failed to load ISIN data: {e}")
        return {}

def populate_indian_isins():
    """Populate ISIN codes for all Indian stocks"""
    print("=" * 60)
    print("POPULATING ISINs FOR INDIAN STOCKS (GitHub Source)")
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
    
    # Download ISIN data
    isin_file = download_isin_data()
    if not isin_file:
        return False
    
    # Load ISIN mapping
    symbol_to_isin = load_isin_mapping(isin_file)
    if not symbol_to_isin:
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
        # Clean up temp file
        if os.path.exists(isin_file):
            os.remove(isin_file)
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
        
        # Lookup ISIN in mapping
        isin = symbol_to_isin.get(symbol.upper())
        if isin:
            df.at[idx, 'isin'] = isin
            updated_count += 1
            if updated_count <= 10:  # Show first 10 updates
                print(f"   ✅ {symbol} -> {isin}")
        else:
            failed_count += 1
            if failed_count <= 10:  # Show first 10 failures
                print(f"   ❌ {symbol} -> No ISIN found")
        
        # Progress update every 50 stocks
        if (updated_count + failed_count) % 50 == 0:
            print(f"   Progress: {updated_count + failed_count}/{stocks_needing_isin}")
    
    # Save updated index
    print(f"\n💾 Saving updated index...")
    df.to_csv(dynamic_path, index=False)
    
    # Clean up temp file
    if os.path.exists(isin_file):
        os.remove(isin_file)
    
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
