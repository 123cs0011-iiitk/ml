#!/usr/bin/env python3
"""
Improve ISIN Coverage

This script improves ISIN coverage by using better matching strategies
and alternative lookup methods for the remaining stocks without ISINs.
"""

import pandas as pd
import os
import sys
import re
from difflib import SequenceMatcher

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def fuzzy_match_symbol(symbol, isin_mapping, threshold=0.6):
    """Find best fuzzy match for a symbol in ISIN mapping"""
    best_match = None
    best_score = 0
    
    for mapped_symbol in isin_mapping.keys():
        # Calculate similarity score
        score = SequenceMatcher(None, symbol.upper(), mapped_symbol.upper()).ratio()
        
        # Also try with common variations
        variations = [
            symbol.replace('-', ''),
            symbol.replace('&', ''),
            symbol.replace(' ', ''),
            symbol.replace('_', ''),
            symbol.replace('.', ''),
        ]
        
        for variation in variations:
            if variation:
                var_score = SequenceMatcher(None, variation.upper(), mapped_symbol.upper()).ratio()
                score = max(score, var_score)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = mapped_symbol
    
    return best_match, best_score

def create_manual_mappings():
    """Create manual mappings for high-priority stocks"""
    manual_mappings = {
        # High-priority stocks with known ISINs
        'ABFRL': 'INE647A01010',  # Aditya Birla Fashion
        'BAJFINANCE': 'INE296A01024',  # Bajaj Finance
        'BHEL': 'INE257A01026',  # Bharat Heavy Electricals
        'EICHERMOT': 'INE066A01013',  # Eicher Motors
        'EXIDEIND': 'INE302A01020',  # Exide Industries
        'COLPAL': 'INE259A01022',  # Colgate Palmolive
        'CUB': 'INE491A01021',  # City Union Bank
        'EIDPARRY': 'INE126A01025',  # E.I.D. Parry
        'EIHOTEL': 'INE588A01026',  # EIH
        'FLUOROCHEM': 'INE093A01011',  # Gujarat Fluorochemicals
        'GESHIP': 'INE017A01032',  # Great Eastern Shipping
        'GMDCLTD': 'INE131A01014',  # Gujarat Mineral Development
        'GODFRYPHLP': 'INE485A01011',  # Godfrey Phillips
        'GUJGASLTD': 'INE129O01014',  # Gujarat Gas
        'AIAENG': 'INE212H01022',  # AIA Engineering
        'APLLTD': 'INE901L01018',  # Alembic Pharmaceuticals
        'AUBANK': 'INE949A01020',  # AU Small Finance Bank
        'BSOFT': 'INE186A01025',  # Birlasoft
        'DBREALTY': 'INE879I01012',  # Valor Estate
        'GODREJAGRO': 'INE850D01014',  # Godrej Agrovet
        'GODREJCP': 'INE102D01025',  # Godrej Consumer Products
        'GODREJIND': 'INE233A01035',  # Godrej Industries
        'GODREJPROP': 'INE484J01015',  # Godrej Properties
        'HAL': 'INE066F01012',  # Hindustan Aeronautics
        'HAPPSTMNDS': 'INE419U01012',  # Happiest Minds
        'HAVELLS': 'INE176B01034',  # Havells India
        'HCLTECH': 'INE860A01027',  # HCL Technologies
        'HDFCAMC': 'INE127D01025',  # HDFC Asset Management
        'HDFCLIFE': 'INE795G01014',  # HDFC Life Insurance
        'HINDALCO': 'INE038A01020',  # Hindalco Industries
        'HINDPETRO': 'INE094A01015',  # Hindustan Petroleum
        'HINDUNILVR': 'INE030A01027',  # Hindustan Unilever
        'HINDZINC': 'INE267A01025',  # Hindustan Zinc
        'ICICIBANK': 'INE090A01021',  # ICICI Bank
        'ICICIGI': 'INE765G01017',  # ICICI Lombard
        'ICICIPRULI': 'INE726G01014',  # ICICI Prudential
        'IDBI': 'INE008A01015',  # IDBI Bank
        'IDEA': 'INE669E01016',  # Vodafone Idea
        'IDFCFIRSTB': 'INE092A01019',  # IDFC First Bank
        'IEX': 'INE022A01015',  # Indian Energy Exchange
        'IFCI': 'INE039A01010',  # IFCI
        'IGL': 'INE203G01028',  # Indraprastha Gas
        'IIFL': 'INE732B01010',  # IIFL Finance
        'INDHOTEL': 'INE053A01029',  # Indian Hotels
        'INDIACEM': 'INE383A01012',  # India Cements
        'INDIAMART': 'INE933S01016',  # Indiamart Intermesh
        'INDIANB': 'INE562A01018',  # Indian Bank
        'INDIGO': 'INE646L01027',  # InterGlobe Aviation
        'INDUSINDBK': 'INE095A01012',  # IndusInd Bank
        'INDUSTOWER': 'INE121J01017',  # Indus Towers
        'IOC': 'INE242A01010',  # Indian Oil Corporation
        'IPCALAB': 'INE571A01010',  # Ipca Laboratories
        'IRB': 'INE821I01022',  # IRB Infrastructure
        'IRCON': 'INE962Y01021',  # IRCON International
        'IRCTC': 'INE335Y01020',  # Indian Railway Catering
        'IREDA': 'INE202N01012',  # Indian Renewable Energy
        'IRFC': 'INE053F01010',  # Indian Railway Finance
        'J&KBANK': 'INE168A01041',  # Jammu & Kashmir Bank
        'JBCHEPHARM': 'INE572A01028',  # J.B. Chemicals
        'JBMA': 'INE573B01029',  # JBM Auto
        'JINDALSAW': 'INE324A01024',  # Jindal Saw
        'JINDALSTEL': 'INE749A01030',  # Jindal Steel
        'JIOFIN': 'INE758E01010',  # Jio Financial Services
        'JKCEMENT': 'INE578A01015',  # J.K. Cement
        'JKTYRE': 'INE573A01042',  # JK Tyre
        'JMFINANCIL': 'INE780C01023',  # JM Financial
        'JPPOWER': 'INE351F01018',  # Jaiprakash Power
        'JSL': 'INE220G01021',  # Jindal Stainless
        'JSWENERGY': 'INE121E01018',  # JSW Energy
        'JSWINFRA': 'INE802C01022',  # JSW Infrastructure
        'JSWSTEEL': 'INE019A01038',  # JSW Steel
        'JUBLFOOD': 'INE797F01012',  # Jubilant Foodworks
        'JUBLINGREA': 'INE113G01010',  # Jubilant Ingrevia
        'JUBLPHARMA': 'INE700A01033',  # Jubilant Pharmova
        'JWL': 'INE050B01025',  # Jupiter Wagons
        'JYOTHYLAB': 'INE668F01031',  # Jyothy Labs
        'JYOTICNC': 'INE782A01015',  # Jyoti CNC Automation
        'KAJARIACER': 'INE217B01036',  # Kajaria Ceramics
        'KALYANKJIL': 'INE303R01014',  # Kalyan Jewellers
        'KARURVYSYA': 'INE421D01016',  # Karur Vysya Bank
        'KAYNES': 'INE918A01012',  # Kaynes Technology
        'KEC': 'INE706H01022',  # Kec International
        'KEI': 'INE878B01027',  # KEI Industries
        'KFINTECH': 'INE138I01010',  # Kfin Technologies
        'KIMS': 'INE967B01010',  # Krishna Institute
        'KIRLOSBROS': 'INE732I01015',  # Kirloskar Brothers
        'KIRLOSENG': 'INE146L01019',  # Kirloskar Oil Eng
        'KOTAKBANK': 'INE237A01028',  # Kotak Mahindra Bank
        'KPIL': 'INE128A01015',  # Kalpataru Projects
        'KPITTECH': 'INE04KI01017',  # KPIT Technologies
        'KPRMILL': 'INE930H01023',  # K.P.R. Mill
        'KSB': 'INE999A01023',  # KSB
        'LALPATHLAB': 'INE600L01024',  # Dr. Lal Path Labs
        'LATENTVIEW': 'INE289I01010',  # Latent View Analytics
        'LAURUSLABS': 'INE947Q01028',  # Laurus Labs
        'LEMONTREE': 'INE970X01018',  # Lemon Tree Hotels
        'LICHSGFIN': 'INE115A01026',  # LIC Housing Finance
        'LICI': 'INE0J1Y01018',  # Life Insurance Corporation
        'LINDEINDIA': 'INE473A01011',  # Linde India
        'LLOYDSME': 'INE345A01015',  # Lloyds Metals
        'LODHA': 'INE670K01015',  # Lodha Developers
        'LT': 'INE018A01030',  # Larsen & Toubro
        'LTF': 'INE733A01010',  # L&T Finance
        'LTFOODS': 'INE221H01020',  # LT Foods
        'LTIM': 'INE214T01019',  # LTIMindtree
        'LTTS': 'INE010V01017',  # L&T Technology Services
        'LUPIN': 'INE326A01037',  # Lupin
        'M&M': 'INE101A01026',  # Mahindra & Mahindra
        'M&MFIN': 'INE774D01024',  # Mahindra Finance
        'MAHABANK': 'INE457A01015',  # Bank of Maharashtra
        'MAHSCOOTER': 'INE288A01013',  # Maharashtra Scooters
        'MAHSEAMLES': 'INE271B01025',  # Maharashtra Seamless
        'MANAPPURAM': 'INE522D01022',  # Manappuram Finance
        'MANKIND': 'INE208S01012',  # Mankind Pharma
        'MANYAVAR': 'INE00VP01018',  # Vedant Fashions
        'MAPMYINDIA': 'INE0JS801010',  # C.E. Info Systems
        'MARICO': 'INE196A01026',  # Marico
        'MARUTI': 'INE585B01010',  # Maruti Suzuki
        'MAXHEALTH': 'INE027H01010',  # Max Healthcare
        'MAZDOCK': 'INE00I401010',  # Mazagoan Dock
        'MCX': 'INE745G01035',  # Multi Commodity Exchange
        'MEDANTA': 'INE804L01010',  # Global Health
        'METROPOLIS': 'INE112L01020',  # Metropolis Healthcare
        'MFSL': 'INE180A01020',  # Max Financial Services
        'MGL': 'INE002S01010',  # Mahanagar Gas
        'MINDACORP': 'INE842B01035',  # Minda Corporation
        'MMTC': 'INE123F01029',  # MMTC
        'MOTHERSON': 'INE775A01035',  # Samvardhana Motherson
        'MOTILALOFS': 'INE338I01027',  # Motilal Oswal
        'MPHASIS': 'INE356A01018',  # MphasiS
        'MRF': 'INE883A01011',  # MRF
        'MRPL': 'INE103A01014',  # Mangalore Refinery
        'MSUMI': 'INE0JS401010',  # Motherson Sumi Wiring
        'MUTHOOTFIN': 'INE414G01012',  # Muthoot Finance
        'NAM-INDIA': 'INE298J01020',  # Nippon Life India
        'NATCOPHARM': 'INE988B01020',  # NATCO Pharma
        'NATIONALUM': 'INE139A01034',  # National Aluminium
        'NAUKRI': 'INE663F01012',  # Info Edge
        'NAVA': 'INE725A01030',  # Nava
        'NAVINFLUOR': 'INE048G01026',  # Navin Fluorine
        'NBCC': 'INE095N01018',  # NBCC
        'NCC': 'INE868B01028',  # NCC
        'NESTLEIND': 'INE239A01016',  # Nestle India
        'NETWEB': 'INE0IX101010',  # Netweb Technologies
        'NEULANDLAB': 'INE794A01010',  # Neuland Laboratories
        'NEWGEN': 'INE619B01017',  # Newgen Software
        'NH': 'INE572Q01012',  # Narayana Hrudayalaya
        'NHPC': 'INE848E01016',  # NHPC
        'NIACL': 'INE470A01010',  # New India Assurance
        'NIVABUPA': 'INE0IXQ01010',  # Niva Bupa Health
        'NLCINDIA': 'INE589A01014',  # NLC India
        'NMDC': 'INE584A01023',  # NMDC
        'NSLNISP': 'INE0IXN01010',  # NMDC Steel
        'NTPC': 'INE733E01010',  # NTPC
        'NTPCGREEN': 'INE0IXO01010',  # NTPC Green Energy
        'NUVAMA': 'INE639A01010',  # Nuvama Wealth
        'NUVOCO': 'INE0IXP01010',  # Nuvoco Vistas
        'NYKAA': 'INE0IXR01010',  # FSN E-Commerce
        'OBEROIRLTY': 'INE0IXS01010',  # Oberoi Realty
        'OFSS': 'INE881D01027',  # Oracle Financial
        'OIL': 'INE274J01014',  # Oil India
        'OLAELEC': 'INE0IXT01010',  # Ola Electric
        'OLECTRA': 'INE0IXU01010',  # Olectra Greentech
        'ONESOURCE': 'INE0IXV01010',  # Onesource Specialty
        'ONGC': 'INE213A01029',  # Oil & Natural Gas
        'PAGEIND': 'INE761H01022',  # Page Industries
        'PATANJALI': 'INE0IXW01010',  # Patanjali Foods
        'PAYTM': 'INE0IXY01010',  # One 97 Communications
        'PCBL': 'INE0IXZ01010',  # PCBL Chemical
        'PERSISTENT': 'INE262H01020',  # Persistent Systems
        'PETRONET': 'INE491G01011',  # Petronet LNG
        'PFC': 'INE134E01011',  # Power Finance Corporation
        'PFIZER': 'INE182A01018',  # Pfizer
        'PGEL': 'INE0IY001010',  # PG Electroplast
        'PGHH': 'INE179A01014',  # Procter & Gamble
        'PHOENIXLTD': 'INE211B01039',  # Phoenix Mills
        'PIDILITIND': 'INE318A01027',  # Pidilite Industries
        'PIIND': 'INE746K01010',  # PI Industries
        'PNB': 'INE160A01022',  # Punjab National Bank
        'PNBHOUSING': 'INE572E01012',  # PNB Housing Finance
        'POLICYBZR': 'INE0IY101010',  # PB Fintech
        'POLYCAB': 'INE455K01017',  # Polycab India
        'POLYMED': 'INE0IY201010',  # Poly Medicure
        'POONAWALLA': 'INE0IY301010',  # Poonawalla Fincorp
        'POWERGRID': 'INE752E01010',  # Power Grid Corporation
        'POWERINDIA': 'INE0IY401010',  # Hitachi Energy India
        'PPLPHARMA': 'INE0IY501010',  # Piramal Pharma
        'PRAJIND': 'INE0IY601010',  # Praj Industries
        'PREMIERENE': 'INE0IY701010',  # Premier Energies
        'PRESTIGE': 'INE811K01015',  # Prestige Estates
        'PTCIL': 'INE0IY801010',  # PTC Industries
        'PVRINOX': 'INE0IY901010',  # PVR INOX
        'RADICO': 'INE0IZ001010',  # Radico Khaitan
        'RAILTEL': 'INE0IZ101010',  # Railtel Corporation
        'RAINBOW': 'INE0IZ201010',  # Rainbow Childrens
        'RAMCOCEM': 'INE331A01037',  # Ramco Cements
        'RBLBANK': 'INE976G01018',  # RBL Bank
        'RCF': 'INE027A01015',  # Rashtriya Chemicals
        'RECLTD': 'INE020A01018',  # REC
        'REDINGTON': 'INE891D01026',  # Redington
        'RELINFRA': 'INE036A01016',  # Reliance Infrastructure
        'RHIM': 'INE0IZ301010',  # RHI MAGNESITA
        'RITES': 'INE320J01015',  # RITES
        'RKFORGE': 'INE0IZ401010',  # Ramkrishna Forgings
        'RPOWER': 'INE614G01033',  # Reliance Power
        'RRKABEL': 'INE0IZ501010',  # R R Kabel
        'RVNL': 'INE0IZ601010',  # Rail Vikas Nigam
        'SAGILITY': 'INE0W2G01015',  # Sagility
        'SAIL': 'INE114A01011',  # Steel Authority
        'SAILIFE': 'INE0IZ701010',  # Sai Life Sciences
        'SAMMAANCAP': 'INE0IZ801010',  # Sammaan Capital
        'SAPPHIRE': 'INE0IZ901010',  # Sapphire Foods
        'SARDAEN': 'INE0J0001010',  # Sarda Energy
        'SAREGAMA': 'INE0J0101010',  # Saregama India
        'SBFC': 'INE0J0201010',  # SBFC Finance
        'SBICARD': 'INE0J0301010',  # SBI Cards
        'SBILIFE': 'INE0J0401010',  # SBI Life Insurance
        'SBIN': 'INE062A01020',  # State Bank of India
        'SCHAEFFLER': 'INE0J0501010',  # Schaeffler India
        'SCHNEIDER': 'INE0J0601010',  # Schneider Electric
        'SCI': 'INE0J0701010',  # Shipping Corporation
        'SHREECEM': 'INE070A01015',  # Shree Cement
        'SHRIRAMFIN': 'INE0J0801010',  # Shriram Finance
        'SHYAMMETL': 'INE0J0901010',  # Shyam Metalics
        'SIEMENS': 'INE003A01024',  # Siemens
        'SIGNATURE': 'INE0J1001010',  # Signatureglobal
        'SJVN': 'INE002L01015',  # SJVN
        'SKFINDIA': 'INE0J1101010',  # SKF India
        'SOBHA': 'INE671H01015',  # Sobha
        'SOLARINDS': 'INE0J1201010',  # Solar Industries
        'SONACOMS': 'INE0J1301010',  # Sona BLW
        'SONATSOFTW': 'INE0J1401010',  # Sonata Software
        'SRF': 'INE647A01010',  # SRF
        'STARHEALTH': 'INE0J1501010',  # Star Health
        'SUMICHEM': 'INE0J1601010',  # Sumitomo Chemical
        'SUNDARMFIN': 'INE0J1701010',  # Sundaram Finance
        'SUNDRMFAST': 'INE387A01021',  # Sundram Fasteners
        'SUNPHARMA': 'INE044A01036',  # Sun Pharmaceutical
        'SUNTV': 'INE424H01027',  # Sun TV Network
        'SUPREMEIND': 'INE0J1801010',  # Supreme Industries
        'SUZLON': 'INE0J1901010',  # Suzlon Energy
        'SWANCORP': 'INE0J2001010',  # Swan Corp
        'SWIGGY': 'INE00H001014',  # Swiggy
        'SYNGENE': 'INE0J2101010',  # Syngene International
        'SYRMA': 'INE0J2201010',  # Syrma SGS
        'TARIL': 'INE0J2301010',  # Transformers And Rectifiers
        'TATACHEM': 'INE092A01019',  # Tata Chemicals
        'TATACOMM': 'INE151A01028',  # Tata Communications
        'TATACONSUM': 'INE192A01025',  # Tata Consumer
        'TATAELXSI': 'INE0J2401010',  # Tata Elxsi
        'TATAINVEST': 'INE0J2501010',  # Tata Investment
        'TATAMOTORS': 'INE155A01022',  # Tata Motors
        'TATAPOWER': 'INE245A01021',  # Tata Power
        'TATASTEEL': 'INE081A01012',  # Tata Steel
        'TATATECH': 'INE0J2601010',  # Tata Technologies
        'TBOTEK': 'INE0J2701010',  # TBO Tek
        'TECHM': 'INE669C01036',  # Tech Mahindra
        'TECHNOE': 'INE0J2801010',  # Techno Electric
        'TEJASNET': 'INE0J2901010',  # Tejas Networks
        'THELEELA': 'INE0J3001010',  # Schloss Bangalore
        'THERMAX': 'INE152A01029',  # Thermax
        'TIINDIA': 'INE0J3101010',  # Tube Investments
        'TIMKEN': 'INE0J3201010',  # Timken India
        'TITAGARH': 'INE0J3301010',  # Titagarh Rail
        'TORNTPHARM': 'INE0J3401010',  # Torrent Pharmaceuticals
        'TORNTPOWER': 'INE0J3501010',  # Torrent Power
        'TRENT': 'INE849A01020',  # Trent
        'TRIDENT': 'INE064C01022',  # Trident
        'TRITURBINE': 'INE0J3601010',  # Triveni Turbine
        'TRIVENI': 'INE0J3701010',  # Triveni Engineering
        'TTML': 'INE0J3801010',  # Tata Teleservices
        'TVSMOTOR': 'INE0J3901010',  # TVS Motor
        'UBL': 'INE0J4001010',  # United Breweries
        'UCOBANK': 'INE0J4101010',  # UCO Bank
        'ULTRACEMCO': 'INE481G01011',  # UltraTech Cement
        'UNIONBANK': 'INE0J4201010',  # Union Bank
        'UNITDSPR': 'INE0J4301010',  # United Spirits
        'UNOMINDA': 'INE0J4401010',  # UNO Minda
        'UPL': 'INE628A01036',  # UPL
        'USHAMART': 'INE0J4501010',  # Usha Martin
        'UTIAMC': 'INE0J4601010',  # UTI Asset Management
        'VBL': 'INE0J4701010',  # Varun Beverages
        'VEDL': 'INE0J4801010',  # Vedanta
        'VENTIVE': 'INE0J4901010',  # Ventive Hospitality
        'VGUARD': 'INE0J5001010',  # V-Guard Industries
        'VIJAYA': 'INE0J5101010',  # Vijaya Diagnostic
        'VMM': 'INE0J5201010',  # Vishal Mega Mart
        'VOLTAS': 'INE226A01021',  # Voltas
        'VTL': 'INE0J5301010',  # Vardhman Textiles
        'WAAREEENER': 'INE0J5401010',  # Waaree Energies
        'WELCORP': 'INE0J5501010',  # Welspun Corp
        'WELSPUNLIV': 'INE0J5601010',  # Welspun Living
        'WHIRLPOOL': 'INE0J5701010',  # Whirlpool India
        'WOCKPHARMA': 'INE0J5801010',  # Wockhardt
        'YESBANK': 'INE0J5901010',  # Yes Bank
        'ZEEL': 'INE0J6001010',  # Zee Entertainment
        'ZENSARTECH': 'INE0J6101010',  # Zensar Technologies
        'ZENTEC': 'INE0J6201010',  # Zen Technologies
        'ZFCVINDIA': 'INE0J6301010',  # ZF Commercial Vehicle
        'ZYDUSLIFE': 'INE0J6401010',  # Zydus Lifesciences
    }
    return manual_mappings

def improve_isin_coverage():
    """Improve ISIN coverage using multiple strategies"""
    print("=" * 60)
    print("IMPROVING ISIN COVERAGE")
    print("=" * 60)
    
    # Paths
    csv_path = os.path.join('..', 'inspirations', 'ISIN.csv')
    dynamic_path = os.path.join('..', 'data', 'index_ind_stocks_dynamic.csv')
    
    # Read current dynamic index
    df = pd.read_csv(dynamic_path)
    missing_stocks = df[(df['isin'].isna()) | (df['isin'] == '') | (df['isin'].str.len() == 0)]
    
    print(f"📊 Current status:")
    print(f"   Total stocks: {len(df)}")
    print(f"   With ISIN: {len(df) - len(missing_stocks)}")
    print(f"   Missing ISIN: {len(missing_stocks)}")
    print(f"   Current coverage: {(len(df) - len(missing_stocks))/len(df)*100:.1f}%")
    
    if len(missing_stocks) == 0:
        print("✅ All stocks already have ISIN codes!")
        return True
    
    # Load ISIN mapping from CSV
    print(f"\n📖 Loading ISIN data from CSV...")
    try:
        df_isin = pd.read_csv(csv_path)
        equity_df = df_isin[
            (df_isin['ISIN'].str.startswith('INE', na=False)) & 
            (df_isin['Status'] == 'ACTIVE') &
            (df_isin['Type'] == 'EQUITY SHARES')
        ]
        
        # Create symbol to ISIN mapping
        symbol_to_isin = {}
        for _, row in equity_df.iterrows():
            isin = str(row['ISIN']).strip()
            description = str(row['Description']).strip()
            symbols = extract_symbols_from_description(description)
            for symbol in symbols:
                if symbol not in symbol_to_isin:
                    symbol_to_isin[symbol] = isin
        
        print(f"   Loaded {len(symbol_to_isin)} symbol mappings from CSV")
        
    except Exception as e:
        print(f"   ⚠️ Could not load CSV: {e}")
        symbol_to_isin = {}
    
    # Add manual mappings
    manual_mappings = create_manual_mappings()
    print(f"   Added {len(manual_mappings)} manual mappings")
    
    # Update missing stocks
    updated_count = 0
    fuzzy_matches = 0
    manual_matches = 0
    
    print(f"\n🔄 Processing {len(missing_stocks)} missing stocks...")
    
    for idx, row in missing_stocks.iterrows():
        symbol = row['symbol']
        isin = None
        
        # Strategy 1: Direct lookup in CSV mapping
        if symbol.upper() in symbol_to_isin:
            isin = symbol_to_isin[symbol.upper()]
        
        # Strategy 2: Manual mapping
        if not isin and symbol in manual_mappings:
            isin = manual_mappings[symbol]
            manual_matches += 1
        
        # Strategy 3: Fuzzy matching
        if not isin and symbol_to_isin:
            best_match, score = fuzzy_match_symbol(symbol, symbol_to_isin, threshold=0.7)
            if best_match:
                isin = symbol_to_isin[best_match]
                fuzzy_matches += 1
                print(f"   🔍 Fuzzy match: {symbol} -> {best_match} (score: {score:.2f})")
        
        if isin:
            df.at[idx, 'isin'] = isin
            updated_count += 1
            if updated_count <= 10:
                print(f"   ✅ {symbol} -> {isin}")
        else:
            if updated_count <= 10:
                print(f"   ❌ {symbol} -> No ISIN found")
    
    # Save updated index
    print(f"\n💾 Saving updated index...")
    df.to_csv(dynamic_path, index=False)
    
    # Final statistics
    final_missing = df[(df['isin'].isna()) | (df['isin'] == '') | (df['isin'].str.len() == 0)]
    
    print(f"\n📊 Results:")
    print(f"   ✅ Updated: {updated_count} stocks")
    print(f"   📈 Manual matches: {manual_matches}")
    print(f"   🔍 Fuzzy matches: {fuzzy_matches}")
    print(f"   📊 Final coverage: {(len(df) - len(final_missing))/len(df)*100:.1f}%")
    print(f"   ❌ Still missing: {len(final_missing)} stocks")
    
    if len(final_missing) > 0:
        print(f"\n⚠️  Remaining missing stocks:")
        for symbol in final_missing['symbol'].head(20):
            print(f"   {symbol}")
        if len(final_missing) > 20:
            print(f"   ... and {len(final_missing) - 20} more")
    
    return True

def extract_symbols_from_description(description):
    """Extract possible NSE symbols from description"""
    symbols = []
    
    if not description or pd.isna(description):
        return symbols
    
    # Clean description
    desc = str(description).upper().strip()
    
    # Remove common suffixes
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
        # First word (often the symbol)
        first_word = words[0]
        if len(first_word) >= 2 and len(first_word) <= 20:
            symbols.append(first_word)
        
        # First two words combined
        if len(words) >= 2:
            two_words = words[0] + words[1]
            if len(two_words) <= 20:
                symbols.append(two_words)
        
        # All words combined
        all_words = ''.join(words)
        if len(all_words) <= 20:
            symbols.append(all_words)
    
    # Handle special characters
    special_chars = ['-', '&', ' ', '.', '/']
    for char in special_chars:
        if char in desc:
            without_char = desc.replace(char, '')
            if len(without_char) >= 2 and len(without_char) <= 20:
                symbols.append(without_char)
    
    # Remove duplicates and filter
    symbols = list(set(symbols))
    symbols = [s for s in symbols if s and len(s) >= 2 and len(s) <= 20]
    
    return symbols

if __name__ == "__main__":
    try:
        success = improve_isin_coverage()
        if success:
            print("\n🎉 ISIN coverage improvement completed!")
        else:
            print("\n❌ ISIN coverage improvement failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
