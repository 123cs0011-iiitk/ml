import os
import socket
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from dotenv import load_dotenv
from live_data.live_fetcher import LiveFetcher

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize live fetcher
live_fetcher = LiveFetcher()

def find_free_port(start_port=5000):
    """Find a free port starting from start_port"""
    port = start_port
    while port < start_port + 100:  # Try up to 100 ports
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            port += 1
    raise RuntimeError("Could not find a free port")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Live Stock Price API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/live_price', methods=['GET'])
def get_live_price():
    """Get live stock price for a symbol"""
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({
            'error': 'Symbol parameter is required',
            'message': 'Please provide a stock symbol using ?symbol=SYMBOL'
        }), 400
    
    try:
        logger.info(f"Fetching live price for symbol: {symbol}")
        result = live_fetcher.fetch_live_price(symbol)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except ValueError as e:
        logger.warning(f"Invalid symbol {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Invalid symbol',
            'message': str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch price',
            'message': f'Unable to fetch live price for {symbol}. Please try again later.',
            'details': str(e)
        }), 500

@app.route('/latest_prices', methods=['GET'])
def get_latest_prices():
    """Get latest prices for all or specific category"""
    category = request.args.get('category')  # us_stocks, ind_stocks, others_stocks
    
    try:
            data = live_fetcher.get_latest_prices(category)
            
            if not data or (hasattr(data, 'empty') and data.empty):
                return jsonify({
                    'success': True,
                    'data': [],
                    'message': 'No data available'
                })
            
            # Handle both pandas DataFrame and list
            if hasattr(data, 'to_dict'):
                return jsonify({
                    'success': True,
                    'data': data.to_dict('records')
                })
            else:
                return jsonify({
                    'success': True,
                    'data': data
                })
        
    except Exception as e:
        logger.error(f"Error getting latest prices: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get latest prices',
            'message': str(e)
        }), 500

@app.route('/search', methods=['GET'])
def search_stocks():
    """Search for stocks by symbol or company name"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({
            'success': True,
            'data': []
        })
    
    try:
        results = []
        query_lower = query.lower()
        
        # Define paths to search
        search_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'permanent', 'us_stocks', 'index_us_stocks.csv'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'permanent', 'ind_stocks', 'index_ind_stocks.csv')
        ]
        
        # Search each CSV file
        for csv_path in search_paths:
            if os.path.exists(csv_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    
                    # Filter results based on query
                    matches = df[
                        (df['symbol'].str.lower().str.contains(query_lower, na=False)) |
                        (df['company_name'].str.lower().str.contains(query_lower, na=False))
                    ]
                    
                    # Convert to list of dictionaries
                    for _, row in matches.head(10).iterrows():  # Limit to 10 per file
                        results.append({
                            'symbol': row['symbol'],
                            'name': row['company_name']
                        })
                        
                except ImportError:
                    # Fallback to csv module
                    import csv
                    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if (query_lower in row['symbol'].lower() or 
                                query_lower in row['company_name'].lower()):
                                results.append({
                                    'symbol': row['symbol'],
                                    'name': row['company_name']
                                })
                                if len(results) >= 20:  # Limit total results
                                    break
        
        # Remove duplicates and limit results
        seen = set()
        unique_results = []
        for result in results:
            if result['symbol'] not in seen:
                seen.add(result['symbol'])
                unique_results.append(result)
                if len(unique_results) >= 20:
                    break
        
        return jsonify({
            'success': True,
            'data': unique_results
        })
        
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to search stocks',
            'message': str(e)
        }), 500

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """Get all available symbols by category"""
    category = request.args.get('category')
    
    try:
        if category:
            # Get symbols for specific category
            csv_path = os.path.join(live_fetcher.data_dir, f'index_{category}_dynamic.csv')
            if os.path.exists(csv_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    symbols = df['symbol'].tolist()
                except ImportError:
                    # Fallback to csv module
                    import csv
                    symbols = []
                    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            symbols.append(row['symbol'])
            else:
                symbols = []
        else:
            # Get symbols for all categories
            symbols = {}
            for cat in ['us_stocks', 'ind_stocks', 'others_stocks']:
                csv_path = os.path.join(live_fetcher.data_dir, f'index_{cat}_dynamic.csv')
                if os.path.exists(csv_path):
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        symbols[cat] = df['symbol'].tolist()
                    except ImportError:
                        # Fallback to csv module
                        import csv
                        cat_symbols = []
                        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                cat_symbols.append(row['symbol'])
                        symbols[cat] = cat_symbols
                else:
                    symbols[cat] = []
        
        return jsonify({
            'success': True,
            'data': symbols
        })
        
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get symbols',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Not found',
        'message': 'The requested endpoint was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

def start_server(port=None):
    """Start the Flask server"""
    if port is None:
        port = int(os.getenv('PORT', 5000))
    
    try:
        # Try to use the specified port
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError:
        # Port is busy, find a free port
        free_port = find_free_port(port)
        logger.info(f"Port {port} is busy, using port {free_port}")
        app.run(host='0.0.0.0', port=free_port, debug=False)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting Live Stock Price API server on port {port}")
    start_server(port)
