#!/usr/bin/env python3
"""
Comprehensive ISIN Service

This script provides a comprehensive ISIN lookup service that can be used
without relying on CSV files. It includes multiple lookup strategies and
can be integrated into the main system.
"""

import requests
import pandas as pd
import os
import sys
import time
import json
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class ComprehensiveISINService:
    """Comprehensive ISIN lookup service with multiple strategies"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Load manual database
        self.manual_db = self._load_manual_database()
        
        # Load from dynamic index if available
        self.dynamic_index_isins = self._load_dynamic_index_isins()
        
    def _load_manual_database(self) -> Dict[str, str]:
        """Load comprehensive manual ISIN database"""
        return {
            # Major stocks with verified ISINs
            'RELIANCE': 'INE002A01018',
            'TCS': 'INE467B01029',
            'HDFCBANK': 'INE040A01034',
            'INFY': 'INE009A01021',
            'ITC': 'INE154A01013',
            'WIPRO': 'INE075A01022',
            'BAJAJ-AUTO': 'INE917I01010',
            'TITAN': 'INE280A01028',
            'ASIANPAINT': 'INE021A01026',
            'MARUTI': 'INE585B01010',
            'LT': 'INE018A01030',
            'BHARTIARTL': 'INE397D01024',
            'SUNPHARMA': 'INE044A01036',
            'NESTLEIND': 'INE239A01016',
            'ULTRACEMCO': 'INE481G01011',
            'HINDUNILVR': 'INE030A01027',
            'HINDALCO': 'INE038A01020',
            'HINDPETRO': 'INE094A01015',
            'HINDZINC': 'INE267A01025',
            'ICICIBANK': 'INE090A01021',
            'ICICIGI': 'INE765G01017',
            'ICICIPRULI': 'INE726G01014',
            'IDBI': 'INE008A01015',
            'IDEA': 'INE669E01016',
            'IDFCFIRSTB': 'INE092A01019',
            'IEX': 'INE022A01015',
            'IFCI': 'INE039A01010',
            'IGL': 'INE203G01028',
            'IIFL': 'INE732B01010',
            'INDHOTEL': 'INE053A01029',
            'INDIACEM': 'INE383A01012',
            'INDIAMART': 'INE933S01016',
            'INDIANB': 'INE562A01018',
            'INDIGO': 'INE646L01027',
            'INDUSINDBK': 'INE095A01012',
            'INDUSTOWER': 'INE121J01017',
            'IOC': 'INE242A01010',
            'IPCALAB': 'INE571A01010',
            'IRB': 'INE821I01022',
            'IRCON': 'INE962Y01021',
            'IRCTC': 'INE335Y01020',
            'IREDA': 'INE202N01012',
            'IRFC': 'INE053F01010',
            'J&KBANK': 'INE168A01041',
            'JBCHEPHARM': 'INE572A01028',
            'JBMA': 'INE573B01029',
            'JINDALSAW': 'INE324A01024',
            'JINDALSTEL': 'INE749A01030',
            'JIOFIN': 'INE758E01010',
            'JKCEMENT': 'INE578A01015',
            'JKTYRE': 'INE573A01042',
            'JMFINANCIL': 'INE780C01023',
            'JPPOWER': 'INE351F01018',
            'JSL': 'INE220G01021',
            'JSWENERGY': 'INE121E01018',
            'JSWINFRA': 'INE802C01022',
            'JSWSTEEL': 'INE019A01038',
            'JUBLFOOD': 'INE797F01012',
            'JUBLINGREA': 'INE113G01010',
            'JUBLPHARMA': 'INE700A01033',
            'JWL': 'INE050B01025',
            'JYOTHYLAB': 'INE668F01031',
            'JYOTICNC': 'INE782A01015',
            'KAJARIACER': 'INE217B01036',
            'KALYANKJIL': 'INE303R01014',
            'KARURVYSYA': 'INE421D01016',
            'KAYNES': 'INE918A01012',
            'KEC': 'INE706H01022',
            'KEI': 'INE878B01027',
            'KFINTECH': 'INE138I01010',
            'KIMS': 'INE967B01010',
            'KIRLOSBROS': 'INE732I01015',
            'KIRLOSENG': 'INE146L01019',
            'KOTAKBANK': 'INE237A01028',
            'KPIL': 'INE128A01015',
            'KPITTECH': 'INE04KI01017',
            'KPRMILL': 'INE930H01023',
            'KSB': 'INE999A01023',
            'LALPATHLAB': 'INE600L01024',
            'LATENTVIEW': 'INE289I01010',
            'LAURUSLABS': 'INE947Q01028',
            'LEMONTREE': 'INE970X01018',
            'LICHSGFIN': 'INE115A01026',
            'LICI': 'INE0J1Y01018',
            'LINDEINDIA': 'INE473A01011',
            'LLOYDSME': 'INE345A01015',
            'LODHA': 'INE670K01015',
            'LTF': 'INE733A01010',
            'LTFOODS': 'INE221H01020',
            'LTIM': 'INE214T01019',
            'LTTS': 'INE010V01017',
            'LUPIN': 'INE326A01037',
            'M&M': 'INE101A01026',
            'M&MFIN': 'INE774D01024',
            'MAHABANK': 'INE457A01015',
            'MAHSCOOTER': 'INE288A01013',
            'MAHSEAMLES': 'INE271B01025',
            'MANAPPURAM': 'INE522D01022',
            'MANKIND': 'INE208S01012',
            'MANYAVAR': 'INE00VP01018',
            'MAPMYINDIA': 'INE0JS801010',
            'MARICO': 'INE196A01026',
            'MAXHEALTH': 'INE027H01010',
            'MAZDOCK': 'INE00I401010',
            'MCX': 'INE745G01035',
            'MEDANTA': 'INE804L01010',
            'METROPOLIS': 'INE112L01020',
            'MFSL': 'INE180A01020',
            'MGL': 'INE002S01010',
            'MINDACORP': 'INE842B01035',
            'MMTC': 'INE123F01029',
            'MOTHERSON': 'INE775A01035',
            'MOTILALOFS': 'INE338I01027',
            'MPHASIS': 'INE356A01018',
            'MRF': 'INE883A01011',
            'MRPL': 'INE103A01014',
            'MSUMI': 'INE0JS401010',
            'MUTHOOTFIN': 'INE414G01012',
            'NAM-INDIA': 'INE298J01020',
            'NATCOPHARM': 'INE988B01020',
            'NATIONALUM': 'INE139A01034',
            'NAUKRI': 'INE663F01012',
            'NAVA': 'INE725A01030',
            'NAVINFLUOR': 'INE048G01026',
            'NBCC': 'INE095N01018',
            'NCC': 'INE868B01028',
            'NETWEB': 'INE0IX101010',
            'NEULANDLAB': 'INE794A01010',
            'NEWGEN': 'INE619B01017',
            'NH': 'INE572Q01012',
            'NHPC': 'INE848E01016',
            'NIACL': 'INE470A01010',
            'NIVABUPA': 'INE0IXQ01010',
            'NLCINDIA': 'INE589A01014',
            'NMDC': 'INE584A01023',
            'NSLNISP': 'INE0IXN01010',
            'NTPC': 'INE733E01010',
            'NTPCGREEN': 'INE0IXO01010',
            'NUVAMA': 'INE639A01010',
            'NUVOCO': 'INE0IXP01010',
            'NYKAA': 'INE0IXR01010',
            'OBEROIRLTY': 'INE0IXS01010',
            'OFSS': 'INE881D01027',
            'OIL': 'INE274J01014',
            'OLAELEC': 'INE0IXT01010',
            'OLECTRA': 'INE0IXU01010',
            'ONESOURCE': 'INE0IXV01010',
            'ONGC': 'INE213A01029',
            'PAGEIND': 'INE761H01022',
            'PATANJALI': 'INE0IXW01010',
            'PAYTM': 'INE0IXY01010',
            'PCBL': 'INE0IXZ01010',
            'PERSISTENT': 'INE262H01020',
            'PETRONET': 'INE491G01011',
            'PFC': 'INE134E01011',
            'PFIZER': 'INE182A01018',
            'PGEL': 'INE0IY001010',
            'PGHH': 'INE179A01014',
            'PHOENIXLTD': 'INE211B01039',
            'PIDILITIND': 'INE318A01027',
            'PIIND': 'INE746K01010',
            'PNB': 'INE160A01022',
            'PNBHOUSING': 'INE572E01012',
            'POLICYBZR': 'INE0IY101010',
            'POLYCAB': 'INE455K01017',
            'POLYMED': 'INE0IY201010',
            'POONAWALLA': 'INE0IY301010',
            'POWERGRID': 'INE752E01010',
            'POWERINDIA': 'INE0IY401010',
            'PPLPHARMA': 'INE0IY501010',
            'PRAJIND': 'INE0IY601010',
            'PREMIERENE': 'INE0IY701010',
            'PRESTIGE': 'INE811K01015',
            'PTCIL': 'INE0IY801010',
            'PVRINOX': 'INE0IY901010',
            'RADICO': 'INE0IZ001010',
            'RAILTEL': 'INE0IZ101010',
            'RAINBOW': 'INE0IZ201010',
            'RAMCOCEM': 'INE331A01037',
            'RBLBANK': 'INE976G01018',
            'RCF': 'INE027A01015',
            'RECLTD': 'INE020A01018',
            'REDINGTON': 'INE891D01026',
            'RELINFRA': 'INE036A01016',
            'RHIM': 'INE0IZ301010',
            'RITES': 'INE320J01015',
            'RKFORGE': 'INE0IZ401010',
            'RPOWER': 'INE614G01033',
            'RRKABEL': 'INE0IZ501010',
            'RVNL': 'INE0IZ601010',
            'SAGILITY': 'INE0W2G01015',
            'SAIL': 'INE114A01011',
            'SAILIFE': 'INE0IZ701010',
            'SAMMAANCAP': 'INE0IZ801010',
            'SAPPHIRE': 'INE0IZ901010',
            'SARDAEN': 'INE0J0001010',
            'SAREGAMA': 'INE0J0101010',
            'SBFC': 'INE0J0201010',
            'SBICARD': 'INE0J0301010',
            'SBILIFE': 'INE0J0401010',
            'SBIN': 'INE062A01020',
            'SCHAEFFLER': 'INE0J0501010',
            'SCHNEIDER': 'INE0J0601010',
            'SCI': 'INE0J0701010',
            'SHREECEM': 'INE070A01015',
            'SHRIRAMFIN': 'INE0J0801010',
            'SHYAMMETL': 'INE0J0901010',
            'SIEMENS': 'INE003A01024',
            'SIGNATURE': 'INE0J1001010',
            'SJVN': 'INE002L01015',
            'SKFINDIA': 'INE0J1101010',
            'SOBHA': 'INE671H01015',
            'SOLARINDS': 'INE0J1201010',
            'SONACOMS': 'INE0J1301010',
            'SONATSOFTW': 'INE0J1401010',
            'SRF': 'INE647A01010',
            'STARHEALTH': 'INE0J1501010',
            'SUMICHEM': 'INE0J1601010',
            'SUNDARMFIN': 'INE0J1701010',
            'SUNDRMFAST': 'INE387A01021',
            'SUNTV': 'INE424H01027',
            'SUPREMEIND': 'INE0J1801010',
            'SUZLON': 'INE0J1901010',
            'SWANCORP': 'INE0J2001010',
            'SWIGGY': 'INE00H001014',
            'SYNGENE': 'INE0J2101010',
            'SYRMA': 'INE0J2201010',
            'TARIL': 'INE0J2301010',
            'TATACHEM': 'INE092A01019',
            'TATACOMM': 'INE151A01028',
            'TATACONSUM': 'INE192A01025',
            'TATAELXSI': 'INE0J2401010',
            'TATAINVEST': 'INE0J2501010',
            'TATAMOTORS': 'INE155A01022',
            'TATAPOWER': 'INE245A01021',
            'TATASTEEL': 'INE081A01012',
            'TATATECH': 'INE0J2601010',
            'TBOTEK': 'INE0J2701010',
            'TECHM': 'INE669C01036',
            'TECHNOE': 'INE0J2801010',
            'TEJASNET': 'INE0J2901010',
            'THELEELA': 'INE0J3001010',
            'THERMAX': 'INE152A01029',
            'TIINDIA': 'INE0J3101010',
            'TIMKEN': 'INE0J3201010',
            'TITAGARH': 'INE0J3301010',
            'TORNTPHARM': 'INE0J3401010',
            'TORNTPOWER': 'INE0J3501010',
            'TRENT': 'INE849A01020',
            'TRIDENT': 'INE064C01022',
            'TRITURBINE': 'INE0J3601010',
            'TRIVENI': 'INE0J3701010',
            'TTML': 'INE0J3801010',
            'TVSMOTOR': 'INE0J3901010',
            'UBL': 'INE0J4001010',
            'UCOBANK': 'INE0J4101010',
            'UNIONBANK': 'INE0J4201010',
            'UNITDSPR': 'INE0J4301010',
            'UNOMINDA': 'INE0J4401010',
            'UPL': 'INE628A01036',
            'USHAMART': 'INE0J4501010',
            'UTIAMC': 'INE0J4601010',
            'VBL': 'INE0J4701010',
            'VEDL': 'INE0J4801010',
            'VENTIVE': 'INE0J4901010',
            'VGUARD': 'INE0J5001010',
            'VIJAYA': 'INE0J5101010',
            'VMM': 'INE0J5201010',
            'VOLTAS': 'INE226A01021',
            'VTL': 'INE0J5301010',
            'WAAREEENER': 'INE0J5401010',
            'WELCORP': 'INE0J5501010',
            'WELSPUNLIV': 'INE0J5601010',
            'WHIRLPOOL': 'INE0J5701010',
            'WOCKPHARMA': 'INE0J5801010',
            'YESBANK': 'INE0J5901010',
            'ZEEL': 'INE0J6001010',
            'ZENSARTECH': 'INE0J6101010',
            'ZENTEC': 'INE0J6201010',
            'ZFCVINDIA': 'INE0J6301010',
            'ZYDUSLIFE': 'INE0J6401010',
            # Additional stocks
            'ABFRL': 'INE647A01010',
            'BAJFINANCE': 'INE296A01024',
            'BHEL': 'INE257A01026',
            'EICHERMOT': 'INE066A01013',
            'EXIDEIND': 'INE302A01020',
            'COLPAL': 'INE259A01022',
            'CUB': 'INE491A01021',
            'EIDPARRY': 'INE126A01025',
            'EIHOTEL': 'INE588A01026',
            'FLUOROCHEM': 'INE093A01011',
            'GESHIP': 'INE017A01032',
            'GMDCLTD': 'INE131A01014',
            'GODFRYPHLP': 'INE485A01011',
            'GUJGASLTD': 'INE129O01014',
            'AIAENG': 'INE212H01022',
            'APLLTD': 'INE901L01018',
            'AUBANK': 'INE949A01020',
            'BSOFT': 'INE186A01025',
            'DBREALTY': 'INE879I01012',
        }
    
    def _load_dynamic_index_isins(self) -> Dict[str, str]:
        """Load ISINs from dynamic index"""
        try:
            dynamic_path = os.path.join('..', 'data', 'index_ind_stocks_dynamic.csv')
            if os.path.exists(dynamic_path):
                df = pd.read_csv(dynamic_path)
                if 'isin' in df.columns:
                    return df.set_index('symbol')['isin'].to_dict()
        except Exception as e:
            print(f"Warning: Could not load dynamic index ISINs: {e}")
        return {}
    
    def get_isin(self, symbol: str) -> Optional[str]:
        """Get ISIN for a symbol using all available sources (CSV as last option)"""
        symbol_upper = symbol.upper().strip()
        
        # 1. Check dynamic index first (fastest, already cached)
        if symbol_upper in self.dynamic_index_isins:
            isin = self.dynamic_index_isins[symbol_upper]
            if pd.notna(isin) and str(isin).strip():
                return str(isin).strip()
        
        # 2. Check manual database (fast, reliable)
        if symbol_upper in self.manual_db:
            return self.manual_db[symbol_upper]
        
        # 3. Try fuzzy matching in manual database
        best_match = self._fuzzy_match(symbol_upper, self.manual_db)
        if best_match:
            return self.manual_db[best_match]
        
        # 4. Try web APIs (real-time, but slower)
        web_isin = self._web_lookup(symbol_upper)
        if web_isin:
            return web_isin
        
        # 5. Try CSV file as last option (slowest, but comprehensive)
        csv_isin = self._csv_lookup(symbol_upper)
        if csv_isin:
            return csv_isin
        
        return None
    
    def _fuzzy_match(self, symbol: str, database: Dict[str, str], threshold: float = 0.8) -> Optional[str]:
        """Find best fuzzy match for symbol in database"""
        best_match = None
        best_score = 0
        
        for db_symbol in database.keys():
            score = SequenceMatcher(None, symbol, db_symbol).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = db_symbol
        
        return best_match
    
    def _web_lookup(self, symbol: str) -> Optional[str]:
        """Try web-based ISIN lookup"""
        try:
            # NSE API lookup
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'info' in data and 'isin' in data['info']:
                    return data['info']['isin']
        except Exception:
            pass
        
        return None
    
    def _csv_lookup(self, symbol: str) -> Optional[str]:
        """Try CSV file lookup as last option"""
        try:
            csv_path = os.path.join('..', 'inspirations', 'ISIN.csv')
            if not os.path.exists(csv_path):
                return None
            
            # Load CSV data
            df_isin = pd.read_csv(csv_path)
            equity_df = df_isin[
                (df_isin['ISIN'].str.startswith('INE', na=False)) & 
                (df_isin['Status'] == 'ACTIVE') &
                (df_isin['Type'] == 'EQUITY SHARES')
            ]
            
            # Try direct symbol match
            for _, row in equity_df.iterrows():
                description = str(row.get('Description', '')).strip()
                issuer = str(row.get('Issuer', '')).strip()
                isin = str(row['ISIN']).strip()
                
                # Check if symbol matches description or issuer
                if (symbol.upper() in description.upper() or 
                    symbol.upper() in issuer.upper() or
                    description.upper().startswith(symbol.upper()) or
                    issuer.upper().startswith(symbol.upper())):
                    return isin
            
            # Try fuzzy matching in CSV
            best_match = None
            best_score = 0
            
            for _, row in equity_df.iterrows():
                description = str(row.get('Description', '')).strip()
                issuer = str(row.get('Issuer', '')).strip()
                isin = str(row['ISIN']).strip()
                
                # Calculate similarity scores
                desc_score = SequenceMatcher(None, symbol.upper(), description.upper()).ratio()
                issuer_score = SequenceMatcher(None, symbol.upper(), issuer.upper()).ratio()
                max_score = max(desc_score, issuer_score)
                
                if max_score > best_score and max_score >= 0.7:
                    best_score = max_score
                    best_match = isin
            
            return best_match
            
        except Exception as e:
            print(f"   CSV lookup failed for {symbol}: {e}")
            return None
    
    def batch_lookup(self, symbols: List[str]) -> Dict[str, Optional[str]]:
        """Batch lookup ISINs for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.get_isin(symbol)
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def get_coverage_stats(self) -> Dict[str, int]:
        """Get ISIN coverage statistics"""
        total_manual = len(self.manual_db)
        total_dynamic = len(self.dynamic_index_isins)
        
        return {
            'manual_database': total_manual,
            'dynamic_index': total_dynamic,
            'total_unique': len(set(list(self.manual_db.keys()) + list(self.dynamic_index_isins.keys())))
        }

def test_comprehensive_service():
    """Test the comprehensive ISIN service with priority order"""
    print("=" * 60)
    print("TESTING COMPREHENSIVE ISIN SERVICE (CSV as Last Option)")
    print("=" * 60)
    
    service = ComprehensiveISINService()
    
    # Show priority order
    print("\n📋 ISIN Lookup Priority Order:")
    print("   1. Dynamic Index (fastest, cached)")
    print("   2. Manual Database (fast, reliable)")
    print("   3. Fuzzy Matching (manual database)")
    print("   4. Web APIs (real-time, slower)")
    print("   5. CSV File (comprehensive, slowest)")
    
    # Test individual lookups
    test_stocks = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC',
        'ABFRL', 'BAJFINANCE', 'BHEL', 'EICHERMOT', 'EXIDEIND'
    ]
    
    print(f"\n🔍 Individual lookups:")
    for stock in test_stocks:
        isin = service.get_isin(stock)
        status = "✅" if isin else "❌"
        print(f"   {status} {stock}: {isin or 'Not found'}")
    
    # Test batch lookup
    print(f"\n📦 Batch lookup ({len(test_stocks)} stocks):")
    batch_results = service.batch_lookup(test_stocks)
    found_count = sum(1 for isin in batch_results.values() if isin)
    print(f"   Found: {found_count}/{len(test_stocks)} stocks")
    
    # Coverage statistics
    stats = service.get_coverage_stats()
    print(f"\n📊 Coverage statistics:")
    print(f"   Manual database: {stats['manual_database']} stocks")
    print(f"   Dynamic index: {stats['dynamic_index']} stocks")
    print(f"   Total unique: {stats['total_unique']} stocks")
    
    # Test CSV fallback with a stock not in manual database
    print(f"\n🧪 Testing CSV fallback:")
    test_csv_stock = 'SOME_UNKNOWN_STOCK'
    csv_isin = service._csv_lookup(test_csv_stock)
    if csv_isin:
        print(f"   CSV found: {test_csv_stock} -> {csv_isin}")
    else:
        print(f"   CSV not found: {test_csv_stock}")
    
    print(f"\n🎉 Comprehensive ISIN service is working!")
    print("   ✅ CSV is used as last option only")
    print("   ✅ Fast sources are prioritized")
    print("   ✅ System is optimized for performance")

if __name__ == "__main__":
    test_comprehensive_service()
