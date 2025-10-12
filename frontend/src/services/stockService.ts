import { projectId, publicAnonKey } from '../utils/supabase/info';

// Stock data interfaces
export interface StockData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: string;
  lastUpdated: string;
}

export interface PricePoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionResult {
  predictedPrice: number;
  confidence: number;
  algorithm: string;
  timeframe: string;
}

// Base URL for API calls
const BASE_URL = `https://${projectId}.supabase.co/functions/v1/make-server-5283ab00`;

// API call helper with error handling
async function apiCall<T>(endpoint: string): Promise<T> {
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      headers: {
        'Authorization': `Bearer ${publicAnonKey}`,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API call failed for ${endpoint}:`, error);
    throw error instanceof Error ? error : new Error('Unknown API error');
  }
}

export const stockService = {
  // Get current stock data
  getStockData: async (symbol: string): Promise<StockData> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }
    
    try {
      return await apiCall<StockData>(`/stock/${symbol.toUpperCase()}`);
    } catch (error) {
      console.error(`Failed to fetch stock data for ${symbol}:`, error);
      throw new Error(`Unable to fetch data for ${symbol}. Please try again later.`);
    }
  },

  // Get historical data
  getHistoricalData: async (symbol: string, period: 'week' | 'month' | 'year'): Promise<PricePoint[]> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }
    
    if (!['week', 'month', 'year'].includes(period)) {
      throw new Error('Invalid period. Must be week, month, or year.');
    }
    
    try {
      return await apiCall<PricePoint[]>(`/historical/${symbol.toUpperCase()}/${period}`);
    } catch (error) {
      console.error(`Failed to fetch historical data for ${symbol} (${period}):`, error);
      throw new Error(`Unable to fetch historical data for ${symbol}. Please try again later.`);
    }
  },

  // Get stock prediction
  getPrediction: async (symbol: string): Promise<PredictionResult> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }
    
    try {
      return await apiCall<PredictionResult>(`/prediction/${symbol.toUpperCase()}`);
    } catch (error) {
      console.error(`Failed to get prediction for ${symbol}:`, error);
      throw new Error(`Unable to generate prediction for ${symbol}. Please try again later.`);
    }
  },

  // Search stocks
  searchStocks: async (query: string): Promise<{ symbol: string; name: string }[]> => {
    try {
      const encodedQuery = encodeURIComponent(query.trim());
      return await apiCall<{ symbol: string; name: string }[]>(`/search?q=${encodedQuery}`);
    } catch (error) {
      console.error(`Failed to search stocks with query "${query}":`, error);
      // Return empty array instead of throwing to avoid breaking the UI
      return [];
    }
  },

  // Get popular stocks
  getPopularStocks: async (): Promise<{ symbol: string; name: string }[]> => {
    try {
      return await apiCall<{ symbol: string; name: string }[]>('/popular');
    } catch (error) {
      console.error('Failed to fetch popular stocks:', error);
      // Return fallback list
      return [
        { symbol: 'AAPL', name: 'Apple Inc.' },
        { symbol: 'GOOGL', name: 'Alphabet Inc.' },
        { symbol: 'MSFT', name: 'Microsoft Corporation' },
        { symbol: 'AMZN', name: 'Amazon.com Inc.' },
        { symbol: 'TSLA', name: 'Tesla Inc.' },
        { symbol: 'META', name: 'Meta Platforms Inc.' },
        { symbol: 'NVDA', name: 'NVIDIA Corporation' },
        { symbol: 'NFLX', name: 'Netflix Inc.' }
      ];
    }
  },

  // Health check for server status
  checkHealth: async (): Promise<{ status: string; timestamp: string }> => {
    try {
      return await apiCall<{ status: string; timestamp: string }>('/health');
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error('Server is unavailable');
    }
  }
};