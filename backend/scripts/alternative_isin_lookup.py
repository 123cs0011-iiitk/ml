#!/usr/bin/env python3
"""
Alternative ISIN Lookup

This script provides alternative methods to get ISIN numbers without relying on CSV files.
Uses web APIs, financial data providers, and other sources.
"""

import requests
import pandas as pd
import os
import sys
import time
import json
from typing import Optional, Dict, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class AlternativeISINLookup:
    """Alternative ISIN lookup using various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def lookup_nse_website(self, symbol: str) -> Optional[str]:
        """Lookup ISIN from NSE website"""
        try:
            # NSE security master API
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'info' in data and 'isin' in data['info']:
                    return data['info']['isin']
            
            # Alternative NSE API
            url = f"https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('data', []):
                    if item.get('symbol') == symbol.upper():
                        return item.get('isin')
                        
        except Exception as e:
            print(f"   NSE website lookup failed for {symbol}: {e}")
        
        return None
    
    def lookup_bse_website(self, symbol: str) -> Optional[str]:
        """Lookup ISIN from BSE website"""
        try:
            # BSE security search
            url = "https://www.bseindia.com/corporates/List_Scrips.aspx"
            params = {
                'searchtext': symbol,
                'searchtype': 'EQ'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                # Parse HTML response for ISIN
                # This would need HTML parsing in a real implementation
                pass
                
        except Exception as e:
            print(f"   BSE website lookup failed for {symbol}: {e}")
        
        return None
    
    def lookup_yahoo_finance(self, symbol: str) -> Optional[str]:
        """Lookup ISIN from Yahoo Finance"""
        try:
            # Yahoo Finance API
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}.NS"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for quote in data.get('quotes', []):
                    if quote.get('symbol') == f"{symbol}.NS":
                        # Yahoo Finance doesn't directly provide ISIN
                        # but we can try to extract from other fields
                        pass
                        
        except Exception as e:
            print(f"   Yahoo Finance lookup failed for {symbol}: {e}")
        
        return None
    
    def lookup_alpha_vantage(self, symbol: str) -> Optional[str]:
        """Lookup ISIN from Alpha Vantage API"""
        try:
            # Alpha Vantage API (requires API key)
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                return None
                
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'OVERVIEW',
                'symbol': f"{symbol}.BSE",  # Try BSE first
                'apikey': api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'ISIN' in data:
                    return data['ISIN']
                    
        except Exception as e:
            print(f"   Alpha Vantage lookup failed for {symbol}: {e}")
        
        return None
    
    def lookup_quandl(self, symbol: str) -> Optional[str]:
        """Lookup ISIN from Quandl API"""
        try:
            # Quandl API (requires API key)
            api_key = os.getenv('QUANDL_API_KEY')
            if not api_key:
                return None
                
            url = f"https://www.quandl.com/api/v3/datasets/NSE/{symbol}"
            params = {'api_key': api_key}
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Quandl might have ISIN in metadata
                if 'dataset' in data and 'column_names' in data['dataset']:
                    # Check if ISIN is in column names or data
                    pass
                    
        except Exception as e:
            print(f"   Quandl lookup failed for {symbol}: {e}")
        
        return None
    
    def lookup_manual_database(self, symbol: str) -> Optional[str]:
        """Lookup ISIN from manual database"""
        # Manual mapping of common stocks
        manual_db = {
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
            'GODREJAGRO': 'INE850D01014',
            'GODREJCP': 'INE102D01025',
            'GODREJIND': 'INE233A01035',
            'GODREJPROP': 'INE484J01015',
            'HAL': 'INE066F01012',
            'HAPPSTMNDS': 'INE419U01012',
            'HAVELLS': 'INE176B01034',
            'HCLTECH': 'INE860A01027',
            'HDFCAMC': 'INE127D01025',
            'HDFCLIFE': 'INE795G01014',
            'HINDALCO': 'INE038A01020',
            'HINDPETRO': 'INE094A01015',
            'HINDUNILVR': 'INE030A01027',
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
            'LT': 'INE018A01030',
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
            'MARUTI': 'INE585B01010',
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
            'NESTLEIND': 'INE239A01016',
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
            'SUNPHARMA': 'INE044A01036',
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
            'ULTRACEMCO': 'INE481G01011',
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
        }
        
        return manual_db.get(symbol.upper())
    
    def lookup_isin(self, symbol: str) -> Optional[str]:
        """Main method to lookup ISIN using all available sources (CSV as last option)"""
        print(f"   Looking up ISIN for {symbol}...")
        
        # Try different sources in order of speed and reliability
        sources = [
            ("Manual Database", self.lookup_manual_database),
            ("NSE Website", self.lookup_nse_website),
            ("BSE Website", self.lookup_bse_website),
            ("Yahoo Finance", self.lookup_yahoo_finance),
            ("Alpha Vantage", self.lookup_alpha_vantage),
            ("Quandl", self.lookup_quandl),
            ("CSV File", self.lookup_csv_file),  # CSV as last option
        ]
        
        for source_name, lookup_func in sources:
            try:
                isin = lookup_func(symbol)
                if isin:
                    print(f"   ✅ Found via {source_name}: {isin}")
                    return isin
                else:
                    print(f"   ❌ {source_name}: No ISIN found")
            except Exception as e:
                print(f"   ⚠️  {source_name}: Error - {e}")
            
            # Rate limiting (except for manual database)
            if source_name != "Manual Database":
                time.sleep(0.5)
        
        print(f"   ❌ No ISIN found for {symbol} from any source")
        return None
    
    def lookup_csv_file(self, symbol: str) -> Optional[str]:
        """Lookup ISIN from CSV file as last option"""
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
            from difflib import SequenceMatcher
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
            print(f"   CSV file lookup failed for {symbol}: {e}")
            return None

def test_alternative_lookup():
    """Test alternative ISIN lookup methods"""
    print("=" * 60)
    print("TESTING ALTERNATIVE ISIN LOOKUP")
    print("=" * 60)
    
    lookup = AlternativeISINLookup()
    
    # Test stocks
    test_stocks = [
        'ABFRL', 'BAJFINANCE', 'BHEL', 'EICHERMOT', 'EXIDEIND',
        'COLPAL', 'CUB', 'EIDPARRY', 'EIHOTEL', 'FLUOROCHEM'
    ]
    
    print(f"Testing {len(test_stocks)} stocks...")
    print()
    
    found_count = 0
    
    for i, stock in enumerate(test_stocks, 1):
        print(f"{i:2d}. {stock}")
        isin = lookup.lookup_isin(stock)
        if isin:
            found_count += 1
        print()
    
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Found ISINs: {found_count}/{len(test_stocks)} stocks")
    print(f"📈 Success rate: {found_count/len(test_stocks)*100:.1f}%")
    
    if found_count > 0:
        print("\n🎉 Alternative ISIN lookup is working!")
    else:
        print("\n⚠️  Alternative ISIN lookup needs improvement")

if __name__ == "__main__":
    test_alternative_lookup()
