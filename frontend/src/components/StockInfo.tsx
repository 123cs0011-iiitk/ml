import React from 'react';
import { TrendingUp, TrendingDown, Calendar, BarChart3, Clock, Database, RefreshCw, Building2, Globe, MapPin, ArrowRightCircle, ArrowUpCircle, ArrowDownCircle, PieChart } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Skeleton } from './ui/skeleton';
import { Alert, AlertDescription } from './ui/alert';
import { Button } from './ui/button';
import { StockData, LivePriceResponse, StockInfoResponse } from '../services/stockService';
import { formatPrice, formatPriceDirect, Currency, setExchangeRate, convertPrice, getExchangeRate } from '../utils/currency';
import { CurrencyToggle } from './CurrencyToggle';

interface StockInfoProps {
  data: StockData | null;
  loading: boolean;
  error: string;
  currency: Currency;
  onCurrencyChange: (currency: Currency) => void;
  livePriceData?: LivePriceResponse | null;
  stockInfoData?: StockInfoResponse | null;
  onRefresh: () => void;
}

export function StockInfo({ data, loading, error, currency, onCurrencyChange, livePriceData, stockInfoData, onRefresh }: StockInfoProps) {
  // Update exchange rate when live price data is available
  React.useEffect(() => {
    if (livePriceData?.exchange_rate) {
      setExchangeRate(livePriceData.exchange_rate);
    }
  }, [livePriceData?.exchange_rate]);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className='flex items-center gap-2'>
              <PieChart className='w-5 h-5' />
              Stock Information
              <Button
                variant="ghost"
                size="sm"
                onClick={onRefresh}
                className="h-8 w-8 p-0"
                disabled={loading}
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
            <CurrencyToggle currency={currency} onCurrencyChange={onCurrencyChange} />
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-8 w-32" />
          <Skeleton className="h-6 w-24" />
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className='flex items-center gap-2'>
              <PieChart className='w-5 h-5' />
              Stock Information
              <Button
                variant="ghost"
                size="sm"
                onClick={onRefresh}
                className="h-8 w-8 p-0"
                disabled={loading}
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
            <CurrencyToggle currency={currency} onCurrencyChange={onCurrencyChange} />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className='flex items-center gap-2'>
              <PieChart className='w-5 h-5' />
              Stock Information
              <Button
                variant="ghost"
                size="sm"
                onClick={onRefresh}
                className="h-8 w-8 p-0"
                disabled={loading}
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
            <CurrencyToggle currency={currency} onCurrencyChange={onCurrencyChange} />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            Select a stock to view detailed information
          </p>
        </CardContent>
      </Card>
    );
  }

  const isPositiveChange = (data.change || 0) >= 0;

  // Determine the actual price to display and whether it's already converted
  const getDisplayPriceInfo = () => {
    if (livePriceData) {
      // Use converted prices from backend if available
      if (currency === 'INR' && livePriceData.price_inr) {
        return { price: livePriceData.price_inr, isConverted: true };
      } else if (currency === 'USD' && livePriceData.price_usd) {
        return { price: livePriceData.price_usd, isConverted: true };
      }
      
      // Check if original currency matches target currency
      if (livePriceData.currency === currency) {
        return { price: data.price, isConverted: true };
      }
      
      // Fallback to original price with conversion
      return { price: data.price, isConverted: false };
    }
    return { price: data.price, isConverted: false };
  };

  const { price: displayPrice, isConverted } = getDisplayPriceInfo();

  // Helper function to format price values consistently
  const formatFieldPrice = (value: number | undefined | null) => {
    if (value === undefined || value === null) return null;
    
    // Determine the source currency (currency of the stock data)
    const sourceCurrency: Currency = livePriceData?.currency as Currency || 'USD';
    
    // Check if conversion is needed
    if (sourceCurrency === currency) {
      // No conversion needed - source and target currencies match
      return formatPriceDirect(value, currency);
    }
    
    // Conversion needed - use convertPrice to handle the conversion properly
    const rate = livePriceData?.exchange_rate || getExchangeRate();
    const converted = convertPrice(value, sourceCurrency, currency, rate);
    
    // Format the converted value
    return formatPriceDirect(converted, currency);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <PieChart className="w-5 h-5" />
            Stock Information
            <Button
              variant="ghost"
              size="sm"
              onClick={onRefresh}
              className="h-8 w-8 p-0"
              disabled={loading}
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
          <CurrencyToggle currency={currency} onCurrencyChange={onCurrencyChange} />
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="text-2xl font-bold">{
            isConverted 
              ? formatPriceDirect(displayPrice, currency)
              : formatPrice(displayPrice, currency, livePriceData?.exchange_rate)
          }</div>
          <div className="flex items-center gap-2 mt-1">
            {isPositiveChange ? (
              <TrendingUp className="w-4 h-4 text-green-600" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-600" />
            )}
            <span className={`text-sm font-medium ${isPositiveChange ? 'text-green-600' : 'text-red-600'
              }`}>
              {formatFieldPrice(Math.abs(data.change || 0))} ({(data.changePercent || 0).toFixed(2)}%)
            </span>
          </div>

          {/* Live price indicator */}
          {livePriceData && (
            <div className="space-y-1 mt-2 text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <Database className="w-3 h-3" />
                <span>Live data from {livePriceData.source}</span>
                <Clock className="w-3 h-3 ml-2" />
                <span>{new Date(livePriceData.timestamp).toLocaleTimeString()}</span>
              </div>
              {livePriceData.exchange_rate && (
                <div className="flex items-center gap-2">
                  <span>Exchange Rate: 1 USD = ₹{livePriceData.exchange_rate.toFixed(2)}</span>
                  <span className="text-xs opacity-75">({livePriceData.exchange_source})</span>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Symbol</span>
            <Badge variant="secondary">{data.symbol}</Badge>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Company</span>
            <span className="text-sm font-medium">{data.name}</span>
          </div>

          {data.open !== undefined && data.open !== null && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground flex items-center gap-1">
                <ArrowRightCircle className="w-3 h-3" />
                Open
              </span>
              <span className="text-sm font-medium">{
                formatFieldPrice(data.open)
              }</span>
            </div>
          )}

          {data.high !== undefined && data.high !== null && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground flex items-center gap-1">
                <ArrowUpCircle className="w-3 h-3 text-green-600" />
                High
              </span>
              <span className="text-sm font-medium text-green-600">{
                formatFieldPrice(data.high)
              }</span>
            </div>
          )}

          {data.low !== undefined && data.low !== null && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground flex items-center gap-1">
                <ArrowDownCircle className="w-3 h-3 text-red-600" />
                Low
              </span>
              <span className="text-sm font-medium text-red-600">{
                formatFieldPrice(data.low)
              }</span>
            </div>
          )}

          {data.close !== undefined && data.close !== null && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                Close
              </span>
              <span className="text-sm font-medium">{
                formatFieldPrice(data.close)
              }</span>
            </div>
          )}

          {data.marketCap && data.marketCap !== 'N/A' && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Market Cap</span>
              <span className="text-sm font-medium">{data.marketCap}</span>
            </div>
          )}

          {/* Additional metadata from stock info data (fast) or live price data (fallback) */}
          {(stockInfoData?.sector || livePriceData?.sector) && (stockInfoData?.sector || livePriceData?.sector) !== 'N/A' && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground flex items-center gap-1">
                <Building2 className="w-3 h-3" />
                Sector
              </span>
              <span className="text-sm font-medium">{stockInfoData?.sector || livePriceData?.sector}</span>
            </div>
          )}

          {(stockInfoData?.market_cap || livePriceData?.market_cap) && (stockInfoData?.market_cap || livePriceData?.market_cap) !== 'N/A' && (stockInfoData?.market_cap || livePriceData?.market_cap) !== data.marketCap && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground flex items-center gap-1">
                <BarChart3 className="w-3 h-3" />
                Market Cap
              </span>
              <span className="text-sm font-medium">{stockInfoData?.market_cap || livePriceData?.market_cap}</span>
            </div>
          )}

          {(stockInfoData?.headquarters || livePriceData?.headquarters) && (stockInfoData?.headquarters || livePriceData?.headquarters) !== 'N/A' && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground flex items-center gap-1">
                <MapPin className="w-3 h-3" />
                Headquarters
              </span>
              <span className="text-sm font-medium">{stockInfoData?.headquarters || livePriceData?.headquarters}</span>
            </div>
          )}

          {(stockInfoData?.exchange || livePriceData?.exchange) && (stockInfoData?.exchange || livePriceData?.exchange) !== 'N/A' && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground flex items-center gap-1">
                <Globe className="w-3 h-3" />
                Exchange
              </span>
              <span className="text-sm font-medium">{stockInfoData?.exchange || livePriceData?.exchange}</span>
            </div>
          )}

          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Last Updated</span>
            <span className="text-sm font-medium">{new Date(data.lastUpdated).toLocaleString()}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}