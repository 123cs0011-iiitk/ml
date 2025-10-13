"""
Main Coordinator Module

This is the central coordinator for the stock prediction system.
It orchestrates all components:
- Live data fetching (live-data module)
- Company information (company-info module) 
- Algorithm selection and execution (algorithms module)
- Prediction generation (prediction module)
- Shared utilities (shared module)

The main.py serves as the entry point and API coordinator.
"""

import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# Import modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'live-data'))
from live_data_manager import LiveFetcher
from shared.utilities import Config, setup_logger, Constants, StockDataError, get_current_timestamp

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize configuration
config = Config()

# Initialize components
live_fetcher = LiveFetcher()

# [FUTURE] Initialize other components when implemented
# company_info_fetcher = CompanyInfoFetcher()
# prediction_engine = PredictionEngine()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Stock Prediction API',
        'version': '1.0.0',
        'timestamp': get_current_timestamp()
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
    category = request.args.get('category')
    
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
            os.path.join(config.permanent_dir, 'us_stocks', 'index_us_stocks.csv'),
            os.path.join(config.permanent_dir, 'ind_stocks', 'index_ind_stocks.csv')
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
                    for _, row in matches.head(10).iterrows():
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
                                if len(results) >= Constants.MAX_SEARCH_RESULTS:
                                    break
        
        # Remove duplicates and limit results
        seen = set()
        unique_results = []
        for result in results:
            if result['symbol'] not in seen:
                seen.add(result['symbol'])
                unique_results.append(result)
                if len(unique_results) >= Constants.MAX_SEARCH_RESULTS:
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
            csv_path = os.path.join(config.data_dir, f'index_{category}_dynamic.csv')
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
            for cat in [Constants.US_STOCKS, Constants.INDIAN_STOCKS, Constants.OTHER_STOCKS]:
                csv_path = os.path.join(config.data_dir, f'index_{cat}_dynamic.csv')
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

# [FUTURE] Prediction endpoints
@app.route('/predict', methods=['POST'])
def predict_stock_price():
    """
    [FUTURE] Generate stock price prediction using selected algorithm
    
    Expected JSON payload:
    {
        "symbol": "AAPL",
        "algorithm": "LSTM",
        "days_ahead": 7
    }
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        algorithm = data.get('algorithm', 'LSTM')
        days_ahead = data.get('days_ahead', 7)
        
        if not symbol:
            return jsonify({
                'success': False,
                'error': 'Symbol is required'
            }), 400
        
        # [FUTURE] Implement prediction logic
        return jsonify({
            'success': False,
            'error': 'Prediction feature not yet implemented',
            'message': 'This endpoint will be available in a future update'
        }), 501
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/company_info', methods=['GET'])
def get_company_info():
    """
    [FUTURE] Get comprehensive company information
    
    Query parameters:
    - symbol: Stock symbol
    - info_type: fundamentals, metadata, news, etc.
    """
    symbol = request.args.get('symbol')
    info_type = request.args.get('info_type', 'all')
    
    if not symbol:
        return jsonify({
            'success': False,
            'error': 'Symbol parameter is required'
        }), 400
    
    # [FUTURE] Implement company info fetching
    return jsonify({
        'success': False,
        'error': 'Company info feature not yet implemented',
        'message': 'This endpoint will be available in a future update'
    }), 501

@app.route('/algorithms', methods=['GET'])
def get_available_algorithms():
    """
    [FUTURE] Get list of available prediction algorithms
    """
    # [FUTURE] Return actual algorithm list
    return jsonify({
        'success': True,
        'data': {
            'available_algorithms': [
                'Random Forest',
                'LSTM',
                'ARIMA',
                'Linear Regression',
                'Support Vector Machine',
                'Gradient Boosting',
                'Neural Network',
                'Technical Analysis',
                'Sentiment Analysis',
                'Ensemble Method'
            ],
            'status': 'Placeholder - algorithms not yet implemented'
        }
    })

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

def find_free_port(start_port=5000):
    """Find a free port starting from start_port"""
    import socket
    port = start_port
    while port < start_port + 100:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            port += 1
    raise RuntimeError("Could not find a free port")

def start_server(port=None):
    """Start the Flask server"""
    if port is None:
        port = config.port
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError:
        free_port = find_free_port(port)
        logger.info(f"Port {port} is busy, using port {free_port}")
        app.run(host='0.0.0.0', port=free_port, debug=False)

if __name__ == '__main__':
    logger.info(f"Starting Stock Prediction API server on port {config.port}")
    start_server(config.port)
