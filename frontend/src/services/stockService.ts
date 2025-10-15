// Live stock data interfaces
export interface StockData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: string;
  lastUpdated: string;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
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

// Live price response from backend
export interface LivePriceResponse {
  symbol: string;
  price: number;
  timestamp: string;
  source: string;
  company_name: string;
  currency: string;
  exchange_rate?: number;
  exchange_source?: string;
  price_inr?: number;
  price_usd?: number;
  sector?: string;
  market_cap?: string;
  headquarters?: string;
  exchange?: string;
  open?: number;
  high?: number;
  low?: number;
  volume?: number;
  close?: number;
}

// Stock info response from backend (fast metadata)
export interface StockInfoResponse {
  symbol: string;
  company_name: string;
  sector: string;
  market_cap: string;
  headquarters: string;
  exchange: string;
  category: string;
}

// API response wrapper
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Backend API configuration
const BACKEND_BASE_URL = 'http://localhost:5000';
const REQUEST_TIMEOUT = 30000; // 30 seconds

// Cache for storing live price data
const cache = new Map<string, { data: any; timestamp: number }>();

// Cache helper functions
function getCachedData<T>(key: string, maxAge: number): T | null {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < maxAge) {
    return cached.data;
  }
  return null;
}

function setCachedData(key: string, data: any): void {
  cache.set(key, { data, timestamp: Date.now() });
}

// Utility function to make API requests with timeout
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = REQUEST_TIMEOUT): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('Request timed out. Please try again.');
    }
    throw error;
  }
}

export const stockService = {
  // Get stock metadata quickly (without live price)
  getStockInfo: async (symbol: string): Promise<StockInfoResponse> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }

    const cacheKey = `stock_info_${symbol}`;
    
    // Check cache first (5 minutes cache)
    const cachedData = getCachedData<StockInfoResponse>(cacheKey, 5 * 60 * 1000);
    if (cachedData) {
      return cachedData;
    }

    try {
      const response = await fetchWithTimeout(`${BACKEND_BASE_URL}/stock_info?symbol=${encodeURIComponent(symbol)}`);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result: ApiResponse<StockInfoResponse> = await response.json();

      if (!result.success || !result.data) {
        throw new Error(result.message || 'Failed to fetch stock info');
      }

      setCachedData(cacheKey, result.data);
      return result.data;

    } catch (error) {
      console.error(`Failed to fetch stock info for ${symbol}:`, error);
      if (error instanceof Error) {
        if (error.message.includes('timed out')) {
          throw new Error('Request timed out. Please try again.');
        } else if (error.message.includes('Failed to fetch')) {
          throw new Error('Unable to connect to server. Please ensure the backend is running.');
        } else {
          throw error;
        }
      }
      throw new Error(`Unable to fetch stock info for ${symbol}. Please try again later.`);
    }
  },

  // Get live stock price from backend
  getLivePrice: async (symbol: string, forceRefresh: boolean = false): Promise<LivePriceResponse> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }

    const cacheKey = `live_price_${symbol}`;
    
    // Only use cache if not forcing refresh
    if (!forceRefresh) {
      const cachedData = getCachedData<LivePriceResponse>(cacheKey, 2 * 60 * 1000); // 2 minutes cache
      if (cachedData) {
        return cachedData;
      }
    }

    try {
      const response = await fetchWithTimeout(`${BACKEND_BASE_URL}/live_price?symbol=${encodeURIComponent(symbol)}`);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result: ApiResponse<LivePriceResponse> = await response.json();

      if (!result.success || !result.data) {
        throw new Error(result.message || 'Failed to fetch live price');
      }

      setCachedData(cacheKey, result.data);
      return result.data;

    } catch (error) {
      console.error(`Failed to fetch live price for ${symbol}:`, error);
      if (error instanceof Error) {
        if (error.message.includes('timed out')) {
          throw new Error('Request timed out. Please try again.');
        } else if (error.message.includes('Failed to fetch')) {
          throw new Error('Unable to connect to server. Please ensure the backend is running.');
        } else {
          throw error;
        }
      }
      throw new Error(`Unable to fetch live price for ${symbol}. Please try again later.`);
    }
  },

  // Get current stock data (converted from live price for compatibility)
  getStockData: async (symbol: string): Promise<StockData> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }

    try {
      const livePrice = await stockService.getLivePrice(symbol);

      // Calculate change and change percent from open and current price
      let change = 0;
      let changePercent = 0;
      
      if (livePrice.open && livePrice.open > 0) {
        change = livePrice.price - livePrice.open;
        changePercent = (change / livePrice.open) * 100;
      }

      // Convert live price to StockData format using actual data
      const stockData: StockData = {
        symbol: livePrice.symbol,
        name: livePrice.company_name,
        price: livePrice.price,
        change: change,
        changePercent: changePercent,
        volume: livePrice.volume || 0,
        marketCap: livePrice.market_cap || 'N/A',
        lastUpdated: livePrice.timestamp,
        open: livePrice.open,
        high: livePrice.high,
        low: livePrice.low
      };

      return stockData;
    } catch (error) {
      console.error(`Failed to get stock data for ${symbol}:`, error);
      throw error;
    }
  },

  // Get historical data (placeholder - would need separate backend endpoint)
  getHistoricalData: async (symbol: string, period: 'week' | 'month' | 'year'): Promise<PricePoint[]> => {
    // For now, return empty array as this would require a separate historical data endpoint
    console.warn(`Historical data for ${symbol} (${period}) not implemented yet`);
    return [];
  },

  // Get stock prediction (placeholder - would need ML backend)
  getPrediction: async (symbol: string): Promise<PredictionResult> => {
    // For now, return a placeholder prediction
    console.warn(`Prediction for ${symbol} not implemented yet`);
    return {
      predictedPrice: 0,
      confidence: 0,
      algorithm: 'Not Available',
      timeframe: 'N/A'
    };
  },

  // Search stocks with backend integration
  searchStocks: async (query: string): Promise<{ symbol: string; name: string }[]> => {
    console.log(`🔍 Searching for: "${query}"`);
    
    const cacheKey = `search_${query}`;
    const cachedData = getCachedData<{ symbol: string; name: string }[]>(cacheKey, 5 * 60 * 1000); // 5 minutes cache

    if (cachedData) {
      console.log(`📦 Using cached data for: "${query}"`);
      return cachedData;
    }

    // Popular stocks for fallback
    const popularStocks = [
      { symbol: 'AAPL', name: 'Apple Inc.' },
      { symbol: 'GOOGL', name: 'Alphabet Inc.' },
      { symbol: 'MSFT', name: 'Microsoft Corporation' },
      { symbol: 'AMZN', name: 'Amazon.com Inc.' },
      { symbol: 'TSLA', name: 'Tesla Inc.' },
      { symbol: 'META', name: 'Meta Platforms Inc.' },
      { symbol: 'NVDA', name: 'NVIDIA Corporation' },
      { symbol: 'NFLX', name: 'Netflix Inc.' }
    ];

    if (!query.trim()) {
      console.log(`📋 Returning popular stocks for empty query`);
      return popularStocks;
    }

    try {
      const url = `${BACKEND_BASE_URL}/search?q=${encodeURIComponent(query)}`;
      console.log(`🌐 Making request to: ${url}`);
      
      const response = await fetchWithTimeout(url);
      console.log(`📡 Response status: ${response.status}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result: ApiResponse<{ symbol: string; name: string }[]> = await response.json();
      console.log(`📊 Backend response:`, result);

      if (result.success && result.data && result.data.length > 0) {
        console.log(`✅ Found ${result.data.length} results from backend`);
        setCachedData(cacheKey, result.data);
        return result.data;
      } else {
        console.log(`⚠️ Backend returned no results, using fallback`);
        // Fallback to hardcoded popular stocks if backend returns no results
        const filteredPopular = popularStocks.filter(stock =>
          stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
          stock.name.toLowerCase().includes(query.toLowerCase())
        );
        return filteredPopular;
      }

    } catch (error) {
      console.error(`❌ Failed to search stocks for "${query}":`, error);

      // Fallback to hardcoded popular stocks on error
      const filteredPopular = popularStocks.filter(stock =>
        stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
        stock.name.toLowerCase().includes(query.toLowerCase())
      );
      console.log(`🔄 Using fallback with ${filteredPopular.length} results`);
      return filteredPopular;
    }
  },

  // Get popular stocks
  getPopularStocks: async (): Promise<{ symbol: string; name: string }[]> => {
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
  },

  // Health check for backend status
  checkHealth: async (): Promise<{ status: string; timestamp: string }> => {
    try {
      const response = await fetchWithTimeout(`${BACKEND_BASE_URL}/health`);

      if (!response.ok) {
        throw new Error(`Backend health check failed: ${response.status}`);
      }

      const result: ApiResponse<{ status: string; timestamp: string }> = await response.json();

      if (!result.success || !result.data) {
        throw new Error('Backend health check failed');
      }

      return result.data;

    } catch (error) {
      console.error('Backend health check failed:', error);
      throw new Error('Backend server is unavailable');
    }
  }
};