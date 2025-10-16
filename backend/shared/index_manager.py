"""
Dynamic Index Manager
Manages the master dynamic index files for stock metadata
"""

import os
import pandas as pd
from typing import Dict, Optional

class DynamicIndexManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.us_index_path = os.path.join(data_dir, 'index_us_stocks_dynamic.csv')
        self.ind_index_path = os.path.join(data_dir, 'index_ind_stocks_dynamic.csv')
    
    def get_index_path(self, category: str) -> str:
        """Get path to dynamic index for category"""
        if category == 'us_stocks':
            return self.us_index_path
        elif category == 'ind_stocks':
            return self.ind_index_path
        else:
            raise ValueError(f"Unknown category: {category}")
    
    def stock_exists(self, symbol: str, category: str) -> bool:
        """Check if stock exists in dynamic index"""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            return False
        df = pd.read_csv(index_path)
        return symbol.upper() in df['symbol'].str.upper().values
    
    def get_stock_info(self, symbol: str, category: str) -> Optional[Dict]:
        """Get stock metadata from dynamic index"""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            return None
        df = pd.read_csv(index_path)
        row = df[df['symbol'].str.upper() == symbol.upper()]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    
    def add_stock(self, symbol: str, stock_info: Dict, category: str):
        """Add new stock to dynamic index (alphabetically sorted)"""
        index_path = self.get_index_path(category)
        
        # Read existing index
        if os.path.exists(index_path):
            df = pd.read_csv(index_path)
        else:
            df = pd.DataFrame(columns=['symbol', 'company_name', 'sector', 
                                      'market_cap', 'headquarters', 'exchange', 'currency'])
        
        # Check if already exists
        if symbol.upper() in df['symbol'].str.upper().values:
            print(f"Stock {symbol} already exists in {category} index")
            return
        
        # Add new stock
        new_row = {
            'symbol': symbol.upper(),
            'company_name': stock_info.get('company_name', symbol),
            'sector': stock_info.get('sector', 'N/A'),
            'market_cap': stock_info.get('market_cap', ''),
            'headquarters': stock_info.get('headquarters', 'N/A'),
            'exchange': stock_info.get('exchange', 'N/A'),
            'currency': 'INR' if category == 'ind_stocks' else 'USD'
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Sort alphabetically by symbol
        df = df.sort_values('symbol').reset_index(drop=True)
        
        # Save
        df.to_csv(index_path, index=False)
        print(f"Added {symbol} to {category} dynamic index")
    
    def get_all_symbols(self, category: str) -> list:
        """Get all symbols from dynamic index"""
        index_path = self.get_index_path(category)
        if not os.path.exists(index_path):
            return []
        df = pd.read_csv(index_path)
        return df['symbol'].tolist()
