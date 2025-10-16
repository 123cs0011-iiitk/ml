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
import pandas as pd
import json
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import yfinance as yf

# Import modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-fetching'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-fetching', 'ind_stocks', 'current-fetching'))
from data_manager import LiveFetcher
from ind_current_fetcher import IndianCurrentFetcher
from us_stocks.latest_fetching.yfinance_latest import USLatestFetcher
from ind_stocks.latest_fetching.yfinance_latest import IndianLatestFetcher
from shared.utilities import Config, setup_logger, Constants, StockDataError, get_current_timestamp, categorize_stock, validate_and_categorize_stock
from shared.currency_converter import convert_usd_to_inr, convert_inr_to_usd, get_exchange_rate_info

# Load environment variables
load_dotenv()

# Custom JSON encoder to handle NaN values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return super().default(obj)

# Setup logging
logger = setup_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.json_encoder = CustomJSONEncoder
CORS(app)

# Initialize configuration
config = Config()

# Initialize components
live_fetcher = LiveFetcher()
indian_fetcher = IndianCurrentFetcher()
us_latest_fetcher = USLatestFetcher()
ind_latest_fetcher = IndianLatestFetcher()

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
        
        # Use appropriate fetcher based on stock category (simplified to only Indian or US)
        # Use validation-based categorization for more accuracy
        category = validate_and_categorize_stock(symbol)
        if category == 'ind_stocks':
            result = indian_fetcher.fetch_current_price(symbol)
        else:  # Default to US stocks
            result = live_fetcher.fetch_live_price(symbol)
        
        # Add currency conversion information
        try:
            exchange_info = get_exchange_rate_info()
            result['exchange_rate'] = exchange_info['rate']
            result['exchange_source'] = exchange_info['source']
            
            # Add converted prices
            if result.get('currency') == 'USD':
                result['price_inr'] = convert_usd_to_inr(result['price'])
            elif result.get('currency') == 'INR':
                result['price_usd'] = convert_inr_to_usd(result['price'])
                
        except Exception as e:
            logger.warning(f"Currency conversion failed: {e}")
            # Continue without currency conversion if it fails
        
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
        return jsonify({'success': True, 'data': []})
    
    try:
        from shared.index_manager import DynamicIndexManager
        index_manager = DynamicIndexManager(config.data_dir)
        
        results = []
        query_lower = query.lower()
        
        # Search both US and Indian stocks from dynamic indexes
        for category in ['us_stocks', 'ind_stocks']:
            index_path = index_manager.get_index_path(category)
            if os.path.exists(index_path):
                df = pd.read_csv(index_path)
                matches = df[
                    (df['symbol'].str.lower().str.contains(query_lower, na=False)) |
                    (df['company_name'].str.lower().str.contains(query_lower, na=False))
                ]
                for _, row in matches.head(10).iterrows():
                    results.append({
                        'symbol': row['symbol'],
                        'name': row.get('company_name', row['symbol'])
                    })
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for result in results:
            if result['symbol'] not in seen:
                seen.add(result['symbol'])
                unique_results.append(result)
                if len(unique_results) >= Constants.MAX_SEARCH_RESULTS:
                    break
        
        # If no results found, try to fetch the query as a new stock symbol
        if not unique_results and query:
            try:
                # Determine if it's likely an Indian stock (no dots, uppercase)
                if query.isupper() and '.' not in query:
                    logger.info(f"Attempting to fetch new Indian stock: {query}")
                    result = indian_fetcher.fetch_current_price(query)
                    if result:
                        unique_results.append({
                            'symbol': result['symbol'],
                            'name': result.get('company_name', result['symbol'])
                        })
                        logger.info(f"Successfully added new stock to search results: {query}")
            except Exception as e:
                logger.info(f"Could not fetch new stock {query}: {e}")
        
        return jsonify({'success': True, 'data': unique_results})
        
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to search stocks', 'message': str(e)}), 500

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """Get all available symbols by category"""
    category = request.args.get('category')
    
    try:
        from shared.index_manager import DynamicIndexManager
        index_manager = DynamicIndexManager(config.data_dir)
        
        if category:
            symbols = index_manager.get_all_symbols(category)
        else:
            symbols = {
                'us_stocks': index_manager.get_all_symbols('us_stocks'),
                'ind_stocks': index_manager.get_all_symbols('ind_stocks')
            }
        
        return jsonify({'success': True, 'data': symbols})
        
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to get symbols', 'message': str(e)}), 500

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

@app.route('/stock_info', methods=['GET'])
def get_stock_info():
    """Get stock metadata from dynamic index"""
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({'success': False, 'error': 'Symbol parameter is required'}), 400
    
    try:
        from shared.index_manager import DynamicIndexManager
        
        logger.info(f"Fetching stock info for symbol: {symbol}")
        category = validate_and_categorize_stock(symbol)
        
        index_manager = DynamicIndexManager(config.data_dir)
        stock_info = index_manager.get_stock_info(symbol, category)
        
        if not stock_info:
            return jsonify({
                'success': False,
                'error': 'Symbol not found',
                'message': f'Symbol {symbol} not found in {category} index'
            }), 404
        
        return jsonify({'success': True, 'data': stock_info})
        
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch stock info',
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

@app.route('/historical', methods=['GET'])
def get_historical_data():
    """
    Get historical stock data for chart visualization.
    
    Query parameters:
    - symbol: Stock symbol
    - period: Time period (week, month, year, 5year)
    """
    symbol = request.args.get('symbol')
    period = request.args.get('period')
    
    if not symbol:
        return jsonify({
            'success': False,
            'error': 'Symbol parameter is required'
        }), 400
    
    if not period or period not in ['week', 'month', 'year', '5year']:
        return jsonify({
            'success': False,
            'error': 'Period parameter is required and must be one of: week, month, year, 5year'
        }), 400
    
    try:
        logger.info(f"Fetching historical data for {symbol} ({period})")
        
        # Determine stock category
        category = validate_and_categorize_stock(symbol)
        
        # Calculate date range based on period
        from datetime import datetime, timedelta
        today = datetime.now().date()
        
        if period == 'week':
            start_date = today - timedelta(days=7)
        elif period == 'month':
            start_date = today - timedelta(days=30)
        elif period == 'year':
            # For year period, use data from 01-01-2024 to current date
            start_date = datetime(2024, 1, 1).date()
        elif period == '5year':
            # For 5year period, use all available historical data (2020-2025)
            # Don't filter by date range - we'll use all available data
            start_date = None
        
        
        # Define file paths
        past_file = os.path.join(config.data_dir, 'past', category, 'individual_files', f'{symbol}.csv')
        latest_file = os.path.join(config.data_dir, 'latest', category, 'individual_files', f'{symbol}.csv')
        
        historical_data = []
        
        # Read past data (2020-2024)
        if os.path.exists(past_file):
            try:
                import pandas as pd
                df_past = pd.read_csv(past_file)
                # Handle both timezone-aware and timezone-naive dates in past data
                try:
                    # Try parsing as timezone-aware first
                    df_past['date'] = pd.to_datetime(df_past['date'], utc=True).dt.tz_localize(None).dt.date
                except:
                    # Fallback to timezone-naive parsing
                    df_past['date'] = pd.to_datetime(df_past['date']).dt.date
                historical_data.append(df_past)
                logger.info(f"Loaded {len(df_past)} records from past data")
            except Exception as e:
                logger.warning(f"Could not read past data for {symbol}: {e}")
        
        # Read latest data (2025+)
        if os.path.exists(latest_file):
            try:
                import pandas as pd
                df_latest = pd.read_csv(latest_file)
                # Ensure consistent date parsing for latest data
                df_latest['date'] = pd.to_datetime(df_latest['date']).dt.date
                historical_data.append(df_latest)
                logger.info(f"Loaded {len(df_latest)} records from latest data")
            except Exception as e:
                logger.warning(f"Could not read latest data for {symbol}: {e}")
        
        if not historical_data:
            return jsonify({
                'success': False,
                'error': 'No data files found',
                'message': f'No historical data found for symbol {symbol}'
            }), 404
        
        # Combine all data
        try:
            import pandas as pd
            combined_df = pd.concat(historical_data, ignore_index=True)
            
            # Remove duplicates and sort by date
            combined_df = combined_df.drop_duplicates(subset=['date']).sort_values('date')
            
            # Debug: Log date range of combined data
            if len(combined_df) > 0:
                logger.info(f"Combined data date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
                logger.info(f"Total combined data points: {len(combined_df)}")
            
            # Filter by date range or get last year of data
            if period == 'year':
                # For year period, filter data from 01-01-2024 to current date
                logger.info(f"Getting 1 year historical data for {symbol} from 2024-01-01")
                filtered_df = combined_df[combined_df['date'] >= start_date]
                logger.info(f"Selected {len(filtered_df)} data points for 1-year chart")
                if len(filtered_df) > 0:
                    logger.info(f"Filtered data date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
            elif period == '5year':
                # For 5year period, use all available historical data (2020-2025)
                logger.info(f"Getting all available historical data for {symbol} (5-year chart)")
                filtered_df = combined_df
                logger.info(f"Selected {len(filtered_df)} data points for 5-year chart")
            else:
                # For other periods, filter by date range
                filtered_df = combined_df[combined_df['date'] >= start_date]
            
            # Convert to list of dictionaries for on-demand fetching check
            price_points = []
            for _, row in filtered_df.iterrows():
                price_points.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'open': float(row['open']) if pd.notna(row['open']) else None,
                    'high': float(row['high']) if pd.notna(row['high']) else None,
                    'low': float(row['low']) if pd.notna(row['low']) else None,
                    'close': float(row['close']) if pd.notna(row['close']) else None,
                    'volume': int(row['volume']) if pd.notna(row['volume']) else 0
                })
            
            # For year and 5year periods, we already have the data we need
            if period not in ['year', '5year']:
                # Check if we need to fetch additional recent data for other periods
                logger.info(f"Checking if additional data needed for {symbol} {period}: {len(price_points)} existing points")
                
                # Use appropriate fetcher based on category
                if category == 'us_stocks':
                    additional_data = us_latest_fetcher.fetch_recent_data_on_demand(symbol, period, price_points)
                elif category == 'ind_stocks':
                    additional_data = ind_latest_fetcher.fetch_recent_data_on_demand(symbol, period, price_points)
                else:
                    additional_data = []
                
                logger.info(f"Additional data fetched: {len(additional_data)} points")
                
                # Combine existing and additional data
                if additional_data:
                    # Remove duplicates and sort by date
                    all_data = price_points + additional_data
                    seen_dates = set()
                    unique_data = []
                    for point in sorted(all_data, key=lambda x: x['date']):
                        if point['date'] not in seen_dates:
                            seen_dates.add(point['date'])
                            unique_data.append(point)
                    price_points = unique_data
                    logger.info(f"Combined data: {len(price_points)} total points for {symbol} ({period})")
            
            if not price_points:
                return jsonify({
                    'success': False,
                    'error': 'No data in date range',
                    'message': f'No data available for {symbol} in the specified {period} period'
                }), 404
            
            logger.info(f"Returning {len(price_points)} price points for {symbol} ({period})")
            
            return jsonify({
                'success': True,
                'data': price_points
            })
            
        except ImportError:
            # Fallback without pandas
            import csv
            from datetime import datetime
            
            all_records = []
            for file_path in [past_file, latest_file]:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', newline='', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                try:
                                    date_obj = datetime.strptime(row['date'][:10], '%Y-%m-%d').date()
                                    if date_obj >= start_date:
                                        all_records.append({
                                            'date': row['date'][:10],
                                            'open': float(row['open']) if row['open'] else None,
                                            'high': float(row['high']) if row['high'] else None,
                                            'low': float(row['low']) if row['low'] else None,
                                            'close': float(row['close']) if row['close'] else None,
                                            'volume': int(float(row['volume'])) if row['volume'] else 0
                                        })
                                except (ValueError, KeyError) as e:
                                    continue
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
            
            if not all_records:
                return jsonify({
                    'success': False,
                    'error': 'No data in date range',
                    'message': f'No data available for {symbol} in the specified {period} period'
                }), 404
            
            # Sort by date
            all_records.sort(key=lambda x: x['date'])
            
            return jsonify({
                'success': True,
                'data': all_records
            })
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch historical data',
            'message': f'Unable to fetch historical data for {symbol}. Please try again later.',
            'details': str(e)
        }), 500

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
