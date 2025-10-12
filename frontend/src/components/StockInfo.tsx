import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Calendar, BarChart3, Clock, Database } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Skeleton } from './ui/skeleton';
import { Alert, AlertDescription } from './ui/alert';
import { StockData, LivePriceResponse } from '../services/stockService';
import { formatPrice, Currency } from '../utils/currency';

interface StockInfoProps {
  data: StockData | null;
  loading: boolean;
  error: string;
  currency: Currency;
  livePriceData?: LivePriceResponse | null;
}

export function StockInfo({ data, loading, error, currency, livePriceData }: StockInfoProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            Stock Information
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
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            Stock Information
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
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            Stock Information
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

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <DollarSign className="w-5 h-5" />
          Stock Information
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="text-2xl font-bold">{formatPrice(data.price, currency)}</div>
          <div className="flex items-center gap-2 mt-1">
            {isPositiveChange ? (
              <TrendingUp className="w-4 h-4 text-green-600" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-600" />
            )}
            <span className={`text-sm font-medium ${isPositiveChange ? 'text-green-600' : 'text-red-600'
              }`}>
              {formatPrice(Math.abs(data.change || 0), currency)} ({(data.changePercent || 0).toFixed(2)}%)
            </span>
          </div>

          {/* Live price indicator */}
          {livePriceData && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <Database className="w-3 h-3" />
              <span>Live data from {livePriceData.source}</span>
              <Clock className="w-3 h-3 ml-2" />
              <span>{new Date(livePriceData.timestamp).toLocaleTimeString()}</span>
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

          {data.open && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Open</span>
              <span className="text-sm font-medium">{formatPrice(data.open, currency)}</span>
            </div>
          )}

          {data.high && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">High</span>
              <span className="text-sm font-medium text-green-600">{formatPrice(data.high, currency)}</span>
            </div>
          )}

          {data.low && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Low</span>
              <span className="text-sm font-medium text-red-600">{formatPrice(data.low, currency)}</span>
            </div>
          )}

          {data.volume && data.volume > 0 && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Volume</span>
              <span className="text-sm font-medium">{data.volume.toLocaleString()}</span>
            </div>
          )}

          {data.marketCap && data.marketCap !== 'N/A' && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Market Cap</span>
              <span className="text-sm font-medium">{data.marketCap}</span>
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