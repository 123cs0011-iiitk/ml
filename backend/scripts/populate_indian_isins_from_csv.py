#!/usr/bin/env python3
"""
Populate ISINs for Indian Stocks from Downloaded CSV

This script populates ISIN codes for all Indian stocks in the dynamic index
using the manually downloaded india-isin-data CSV file with improved symbol matching.
"""

import pandas as pd
import os
import sys
import re

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def extract_symbols_from_description(description):
    """Extract possible NSE symbols from description with improved matching"""
    symbols = []
    
    if not description or pd.isna(description):
        return symbols
    
    # Clean description
    desc = str(description).upper().strip()
    
    # Remove common suffixes and patterns
    suffixes_to_remove = [
        ' LTD EQ', ' LIMITED EQ', ' EQ', ' LTD', ' LIMITED',
        ' CORPORATION', ' CORP', ' INC', ' INCORPORATED',
        ' COMPANY', ' CO', ' PVT', ' PRIVATE',
        ' HOLDINGS', ' HLDG', ' GROUP', ' GRP'
    ]
    
    for suffix in suffixes_to_remove:
        desc = desc.replace(suffix, '')
    
    # Extract different possible symbols
    words = desc.split()
    
    if words:
        # Strategy 1: First word (often the symbol)
        first_word = words[0]
        if len(first_word) >= 2 and len(first_word) <= 20:
            symbols.append(first_word)
        
        # Strategy 2: First two words combined (for multi-word symbols)
        if len(words) >= 2:
            two_words = words[0] + words[1]
            if len(two_words) <= 20:
                symbols.append(two_words)
        
        # Strategy 3: All words combined (for complex symbols)
        all_words = ''.join(words)
        if len(all_words) <= 20:
            symbols.append(all_words)
    
    # Strategy 4: Handle special characters
    # Replace common separators with different combinations
    special_chars = ['-', '&', ' ', '.', '/']
    for char in special_chars:
        if char in desc:
            # Try with and without the character
            without_char = desc.replace(char, '')
            if len(without_char) >= 2 and len(without_char) <= 20:
                symbols.append(without_char)
            
            # Try with underscore instead
            with_underscore = desc.replace(char, '_')
            if len(with_underscore) >= 2 and len(with_underscore) <= 20:
                symbols.append(with_underscore)
    
    # Strategy 5: Extract from common patterns
    # Look for patterns like "SYMBOL - COMPANY NAME"
    if ' - ' in desc:
        before_dash = desc.split(' - ')[0].strip()
        if len(before_dash) >= 2 and len(before_dash) <= 20:
            symbols.append(before_dash)
    
    # Remove duplicates and filter
    symbols = list(set(symbols))
    symbols = [s for s in symbols if s and len(s) >= 2 and len(s) <= 20]
    
    return symbols

def load_isin_mapping(csv_path):
    """Load ISIN data and create comprehensive symbol to ISIN mapping"""
    print("📖 Loading ISIN data from downloaded CSV...")
    
    try:
        # Read the ISIN CSV
        df_isin = pd.read_csv(csv_path)
        print(f"   Found {len(df_isin)} total ISIN records")
        
        # Filter for active equity shares
        equity_df = df_isin[
            (df_isin['ISIN'].str.startswith('INE', na=False)) & 
            (df_isin['Status'] == 'ACTIVE') &
            (df_isin['Type'] == 'EQUITY SHARES')
        ]
        print(f"   Found {len(equity_df)} active equity shares")
        
        # Create symbol to ISIN mapping
        symbol_to_isin = {}
        symbol_sources = {}  # Track where each symbol came from
        
        for _, row in equity_df.iterrows():
            isin = str(row['ISIN']).strip()
            description = str(row['Description']).strip()
            issuer = str(row['Issuer']).strip() if 'Issuer' in row else ''
            
            # Extract symbols from description
            symbols = extract_symbols_from_description(description)
            
            # Also try issuer if available
            if issuer and issuer != 'nan':
                issuer_symbols = extract_symbols_from_description(issuer)
                symbols.extend(issuer_symbols)
            
            # Add all symbols to mapping
            for symbol in symbols:
                if symbol not in symbol_to_isin:
                    symbol_to_isin[symbol] = isin
                    symbol_sources[symbol] = description[:50] + "..." if len(description) > 50 else description
        
        print(f"   Created mapping for {len(symbol_to_isin)} unique symbols")
        
        # Show some examples
        print("   📋 Sample mappings:")
        for i, (symbol, isin) in enumerate(list(symbol_to_isin.items())[:10]):
            source = symbol_sources.get(symbol, "Unknown")
            print(f"      {symbol} -> {isin} (from: {source})")
        
        return symbol_to_isin
        
    except Exception as e:
        print(f"   ❌ Failed to load ISIN data: {e}")
        return {}

def populate_indian_isins():
    """Populate ISIN codes for all Indian stocks"""
    print("=" * 60)
    print("POPULATING ISINs FOR INDIAN STOCKS (Downloaded CSV)")
    print("=" * 60)
    
    # Paths
    csv_path = os.path.join('..', 'inspirations', 'ISIN.csv')
    dynamic_path = os.path.join('..', 'data', 'index_ind_stocks_dynamic.csv')
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"❌ ISIN CSV not found: {csv_path}")
        return False
    
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
    
    # Load ISIN mapping
    symbol_to_isin = load_isin_mapping(csv_path)
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
        return True
    
    # Populate ISINs with improved matching
    print(f"\n🔄 Populating ISINs for {stocks_needing_isin} stocks...")
    updated_count = 0
    failed_count = 0
    failed_symbols = []
    
    for idx, row in df.iterrows():
        symbol = row['symbol']
        current_isin = str(row.get('isin', ''))
        
        # Skip if ISIN already populated
        if current_isin and current_isin != 'nan' and len(current_isin.strip()) > 0:
            continue
        
        # Try multiple matching strategies
        isin = None
        
        # Strategy 1: Exact match
        if symbol.upper() in symbol_to_isin:
            isin = symbol_to_isin[symbol.upper()]
        
        # Strategy 2: Try without special characters
        if not isin:
            clean_symbol = symbol.replace('-', '').replace('&', '').replace(' ', '').upper()
            if clean_symbol in symbol_to_isin:
                isin = symbol_to_isin[clean_symbol]
        
        # Strategy 3: Try with different separators
        if not isin:
            for sep in ['-', '&', ' ']:
                if sep in symbol:
                    alt_symbol = symbol.replace(sep, '_').upper()
                    if alt_symbol in symbol_to_isin:
                        isin = symbol_to_isin[alt_symbol]
                        break
        
        # Strategy 4: Try partial matches (first few characters)
        if not isin:
            for mapped_symbol in symbol_to_isin.keys():
                if (len(symbol) >= 4 and 
                    symbol.upper().startswith(mapped_symbol[:4]) and
                    abs(len(symbol) - len(mapped_symbol)) <= 2):
                    isin = symbol_to_isin[mapped_symbol]
                    break
        
        if isin:
            df.at[idx, 'isin'] = isin
            updated_count += 1
            if updated_count <= 20:  # Show first 20 updates
                print(f"   ✅ {symbol} -> {isin}")
        else:
            failed_count += 1
            failed_symbols.append(symbol)
            if failed_count <= 20:  # Show first 20 failures
                print(f"   ❌ {symbol} -> No ISIN found")
        
        # Progress update every 50 stocks
        if (updated_count + failed_count) % 50 == 0:
            print(f"   Progress: {updated_count + failed_count}/{stocks_needing_isin}")
    
    # Save updated index
    print(f"\n💾 Saving updated index...")
    df.to_csv(dynamic_path, index=False)
    
    print(f"\n📊 Results:")
    print(f"   ✅ Successfully populated: {updated_count} stocks")
    print(f"   ❌ Failed to populate: {failed_count} stocks")
    print(f"   📈 Coverage: {(stocks_with_isin + updated_count)/len(df)*100:.1f}%")
    print(f"   📁 Saved to: {dynamic_path}")
    
    # Show some examples of populated ISINs
    populated_stocks = df[df['isin'].str.len() > 0].head(10)
    if not populated_stocks.empty:
        print(f"\n📋 Sample populated ISINs:")
        for _, row in populated_stocks.iterrows():
            print(f"   {row['symbol']}: {row['isin']}")
    
    # Show some failed symbols for manual review
    if failed_symbols:
        print(f"\n⚠️  Failed symbols (first 20):")
        for symbol in failed_symbols[:20]:
            print(f"   {symbol}")
        if len(failed_symbols) > 20:
            print(f"   ... and {len(failed_symbols) - 20} more")
    
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
