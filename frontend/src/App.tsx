import React, { useState, useEffect } from 'react';
import { BarChart3, AlertCircle, Info } from 'lucide-react';
import { StockSearch } from './components/StockSearch';
import { StockInfo } from './components/StockInfo';
import { StockChart } from './components/StockChart';
import { StockPrediction } from './components/StockPrediction';
import { CurrencyToggle } from './components/CurrencyToggle';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';
import { Alert, AlertDescription } from './components/ui/alert';
import { stockService, StockData, PricePoint, PredictionResult, LivePriceResponse, StockInfoResponse } from './services/stockService';
import { Currency } from './utils/currency';

export default function App() {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [stockData, setStockData] = useState<StockData | null>(null);
  const [livePriceData, setLivePriceData] = useState<LivePriceResponse | null>(null);
  const [stockInfoData, setStockInfoData] = useState<StockInfoResponse | null>(null);
  const [chartData, setChartData] = useState<PricePoint[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictionHorizon, setPredictionHorizon] = useState<string>('1d');
  const [chartPeriod, setChartPeriod] = useState<'year' | '5year'>('year');
  const [currency, setCurrency] = useState<Currency>('USD');
  const [loading, setLoading] = useState({
    stock: false,
    chart: false,
    prediction: false
  });
  const [errors, setErrors] = useState({
    stock: '',
    chart: '',
    prediction: ''
  });

  // Load stock data when symbol changes
  useEffect(() => {
    if (selectedSymbol) {
      loadStockData(selectedSymbol);
      loadChartData(selectedSymbol, chartPeriod);
      loadPrediction(selectedSymbol, predictionHorizon);
    }
  }, [selectedSymbol]);

  // Reload prediction when horizon changes
  useEffect(() => {
    if (selectedSymbol) {
      loadPrediction(selectedSymbol, predictionHorizon);
    }
  }, [predictionHorizon, selectedSymbol]);

  // Reload chart data when period changes
  useEffect(() => {
    if (selectedSymbol) {
      loadChartData(selectedSymbol, chartPeriod);
    }
  }, [chartPeriod, selectedSymbol]);

  const loadStockData = async (symbol: string, forceRefresh: boolean = false) => {
    setLoading(prev => ({ ...prev, stock: true }));
    setErrors(prev => ({ ...prev, stock: '' }));
    try {
      // First, fetch stock info quickly (metadata)
      try {
        const stockInfo = await stockService.getStockInfo(symbol);
        setStockInfoData(stockInfo);
      } catch (error) {
        console.warn('Failed to load stock info:', error);
        // Continue without stock info - live price will provide fallback
      }

      // Then, fetch live price data (slower)
      const livePrice = await stockService.getLivePrice(symbol, forceRefresh);
      setLivePriceData(livePrice);

      // Convert to StockData format for compatibility
      const data = await stockService.getStockData(symbol);
      setStockData(data);
    } catch (error) {
      console.error('Failed to load stock data:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to load stock data';
      setErrors(prev => ({ ...prev, stock: errorMessage }));
      setStockData(null);
      setLivePriceData(null);
      setStockInfoData(null);
    } finally {
      setLoading(prev => ({ ...prev, stock: false }));
    }
  };

  const loadChartData = async (symbol: string, period: 'year' | '5year') => {
    setLoading(prev => ({ ...prev, chart: true }));
    setErrors(prev => ({ ...prev, chart: '' }));
    try {
      const data = await stockService.getHistoricalData(symbol, period);
      setChartData(data);
    } catch (error) {
      console.error('Failed to load chart data:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to load chart data';
      setErrors(prev => ({ ...prev, chart: errorMessage }));
      setChartData([]);
    } finally {
      setLoading(prev => ({ ...prev, chart: false }));
    }
  };

  const loadPrediction = async (symbol: string, horizon: string = predictionHorizon) => {
    setLoading(prev => ({ ...prev, prediction: true }));
    setErrors(prev => ({ ...prev, prediction: '' }));
    try {
      const data = await stockService.getPrediction(symbol, horizon);
      setPrediction(data);
    } catch (error) {
      console.error('Failed to load prediction:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate prediction';
      setErrors(prev => ({ ...prev, prediction: errorMessage }));
      setPrediction(null);
    } finally {
      setLoading(prev => ({ ...prev, prediction: false }));
    }
  };

  const handleStockSelect = (symbol: string) => {
    setSelectedSymbol(symbol);
  };

  const handlePeriodChange = (period: 'year' | '5year') => {
    setChartPeriod(period);
  };

  const handleCurrencyChange = (newCurrency: Currency) => {
    setCurrency(newCurrency);
  };

  const handleHorizonChange = (horizon: string) => {
    setPredictionHorizon(horizon);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="main-app-container space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="header-container">
            <BarChart3 className="stock-price-prediction-icon text-primary" />
            <h1 className="stock-price-prediction-header">Stock Price Prediction</h1>
          </div>
          <p className="text-2xl text-muted-foreground max-w-2xl mx-auto">
            Analyze real-time stock data and get AI-powered price predictions using machine learning algorithms.
            Always conduct your own research before making investment decisions.
          </p>
        </div>

        {/* Warning Banner */}
        <Alert className="alert-scaled">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Investment Warning:</strong> Stock market predictions are inherently uncertain.
            This tool provides statistical analysis for educational purposes only. Past performance
            does not guarantee future results. Always consult with financial advisors and do thorough
            research before making investment decisions.
          </AlertDescription>
        </Alert>

        <div className="card-grid-layout">
          {/* Left Sidebar */}
          <div className="sidebar-container">
            <StockSearch
              onStockSelect={handleStockSelect}
              selectedSymbol={selectedSymbol}
            />
            <StockInfo
              data={stockData}
              loading={loading.stock}
              error={errors.stock}
              currency={currency}
              onCurrencyChange={handleCurrencyChange}
              livePriceData={livePriceData}
              stockInfoData={stockInfoData}
              onRefresh={() => selectedSymbol && loadStockData(selectedSymbol, true)}
            />
          </div>

          {/* Main Content */}
          <div className="main-content-container">
            <StockChart
              data={chartData}
              symbol={selectedSymbol || 'Select a Stock'}
              onPeriodChange={handlePeriodChange}
              currentPeriod={chartPeriod}
              loading={loading.chart}
              error={errors.chart}
              currency={currency}
            />

            <StockPrediction
              prediction={prediction}
              currentPrice={stockData?.price}
              loading={loading.prediction}
              symbol={selectedSymbol}
              error={errors.prediction}
              currency={currency}
              onHorizonChange={handleHorizonChange}
            />
          </div>
        </div>

        {/* Footer */}
        <Card>
          <CardHeader>
            <CardTitle className="card-title-scaled card-title-with-icon">
              <Info className="card-icon-scaled" />
              About This Tool
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-2xl text-muted-foreground">
              This stock prediction dashboard uses k-nearest neighbor (KNN) machine learning algorithm
              to analyze recent price patterns and predict short-term price movements. The algorithm
              examines the most recent trading data to identify similar patterns and estimate future prices.
            </p>
            <div className="text-2xl text-muted-foreground space-y-2">
              <p><strong>Backend:</strong> Powered by Supabase edge functions with real-time data processing</p>
              <p><strong>Prediction Model:</strong> K-Nearest Neighbor algorithm with weighted recent data analysis</p>
              <p><strong>Timeframes:</strong> Weekly, monthly, and yearly historical data analysis</p>
              <p><strong>Caching:</strong> Smart caching for performance - stock data (5 min), historical data (1 hour), predictions (15 min)</p>
              <p><strong>Currency Support:</strong> Real-time conversion between USD and INR with live formatting</p>
              <p><strong>Future:</strong> Ready for Yahoo Finance API integration with environment variable configuration</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}