import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Brain, Target, Calendar, AlertTriangle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Skeleton } from './ui/skeleton';
import { Alert, AlertDescription } from './ui/alert';
import { Button } from './ui/button';
import { PredictionResult } from '../services/stockService';
import { formatPrice, Currency } from '../utils/currency';

interface StockPredictionProps {
  prediction: PredictionResult | null;
  currentPrice?: number;
  loading: boolean;
  symbol: string;
  error: string;
  currency: Currency;
  onHorizonChange?: (horizon: string) => void;
}

const HORIZON_OPTIONS = [
  { key: '1d', label: '1D' },
  { key: '1w', label: '1W' },
  { key: '1m', label: '1M' },
  { key: '1y', label: '1Y' },
  { key: '5y', label: '5Y' },
];

export function StockPrediction({ 
  prediction, 
  currentPrice, 
  loading, 
  symbol, 
  error,
  currency,
  onHorizonChange
}: StockPredictionProps) {
  const [selectedHorizon, setSelectedHorizon] = useState('1d');

  const handleHorizonChange = (horizon: string) => {
    setSelectedHorizon(horizon);
    if (onHorizonChange) {
      onHorizonChange(horizon);
    }
  };
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="card-title-scaled card-title-with-icon flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Brain className="card-icon-scaled" />
              <span>AI Price Prediction</span>
            </div>
            <div className="flex gap-1">
              {HORIZON_OPTIONS.map(option => (
                <Button
                  key={option.key}
                  variant={selectedHorizon === option.key ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handleHorizonChange(option.key)}
                >
                  {option.label}
                </Button>
              ))}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Skeleton className="h-6 w-32" />
            <Skeleton className="h-8 w-24" />
          </div>
          <Skeleton className="h-4 w-full" />
          <div className="grid grid-cols-2 gap-4">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="card-title-scaled card-title-with-icon flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Brain className="card-icon-scaled" />
              <span>AI Price Prediction</span>
            </div>
            <div className="flex gap-1">
              {HORIZON_OPTIONS.map(option => (
                <Button
                  key={option.key}
                  variant={selectedHorizon === option.key ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handleHorizonChange(option.key)}
                >
                  {option.label}
                </Button>
              ))}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <Alert>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!prediction) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="card-title-scaled card-title-with-icon flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Brain className="card-icon-scaled" />
              <span>AI Price Prediction</span>
            </div>
            <div className="flex gap-1">
              {HORIZON_OPTIONS.map(option => (
                <Button
                  key={option.key}
                  variant={selectedHorizon === option.key ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handleHorizonChange(option.key)}
                >
                  {option.label}
                </Button>
              ))}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-muted-foreground prediction-empty-state">
            Select a stock to see AI-powered price predictions
          </p>
        </CardContent>
      </Card>
    );
  }

  const isPositiveChange = prediction.predictedPrice > (currentPrice || 0);
  const change = (currentPrice && prediction.predictedPrice) ? prediction.predictedPrice - currentPrice : 0;
  const changePercent = (currentPrice && change !== 0) ? ((change / currentPrice) * 100) : 0;

  const getConfidenceColor = (confidence: number | undefined | null) => {
    if (!confidence || isNaN(confidence)) return 'text-muted-foreground';
    if (confidence >= 70) return 'text-green-600';
    if (confidence >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBadgeVariant = (confidence: number | undefined | null): "default" | "secondary" | "destructive" | "outline" => {
    if (!confidence || isNaN(confidence)) return 'outline';
    if (confidence >= 70) return 'default';
    if (confidence >= 50) return 'secondary';
    return 'destructive';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="card-title-scaled card-title-with-icon flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Brain className="card-icon-scaled" />
            <span>AI Price Prediction for {symbol}</span>
          </div>
          <div className="flex gap-1">
            {HORIZON_OPTIONS.map(option => (
              <Button
                key={option.key}
                variant={selectedHorizon === option.key ? 'default' : 'outline'}
                size="sm"
                onClick={() => handleHorizonChange(option.key)}
              >
                {option.label}
              </Button>
            ))}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Prediction Alert */}
        <Alert className="prediction-alert">
          <AlertTriangle className="prediction-alert-icon" />
          <AlertDescription className="prediction-alert-text">
            This prediction is based on historical data analysis using machine learning. 
            Market conditions can change rapidly and predictions may not reflect future performance.
          </AlertDescription>
        </Alert>

        {/* Main Prediction */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="prediction-metadata-label text-muted-foreground">Predicted Price ({prediction.timeframe})</span>
            <div className="flex items-center gap-2">
              {isPositiveChange ? (
                <TrendingUp className="prediction-icon text-green-600" />
              ) : (
                <TrendingDown className="prediction-icon text-red-600" />
              )}
              <Badge variant={getConfidenceBadgeVariant(prediction.confidence)} className="prediction-confidence-badge">
                {prediction.confidence ? prediction.confidence.toFixed(1) : '0.0'}% confidence
              </Badge>
            </div>
          </div>
          
          <div className="prediction-price-main">
            {formatPrice(prediction.predictedPrice, currency)}
          </div>
          
          {currentPrice && (
            <div className={`prediction-change-info ${
              isPositiveChange ? 'text-green-600' : 'text-red-600'
            }`}>
              {isPositiveChange ? '+' : ''}{formatPrice(change, currency)} ({changePercent.toFixed(2)}%)
            </div>
          )}
        </div>

        {/* Confidence Indicator */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="prediction-metadata-label text-muted-foreground">Prediction Confidence</span>
            <span className={`prediction-metadata-value ${getConfidenceColor(prediction.confidence)}`}>
              {prediction.confidence ? prediction.confidence.toFixed(1) : '0.0'}%
            </span>
          </div>
          <Progress value={prediction.confidence || 0} className="h-2" />
        </div>

        {/* Additional Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Target className="prediction-small-icon text-muted-foreground" />
              <span className="prediction-metadata-label text-muted-foreground">Price Range</span>
            </div>
            <div className="prediction-metadata-value">
              {formatPrice(prediction.predictedPrice * 0.95, currency)} - {formatPrice(prediction.predictedPrice * 1.05, currency)}
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Calendar className="prediction-small-icon text-muted-foreground" />
              <span className="prediction-metadata-label text-muted-foreground">Time Frame</span>
            </div>
            <div className="prediction-metadata-value capitalize">{prediction.timeframe}</div>
          </div>
        </div>

        {/* Model Info */}
        <div className="pt-6 border-t space-y-3">
          <h4 className="prediction-model-heading">Model Information</h4>
          <div className="prediction-model-text text-muted-foreground space-y-1">
            <p><strong>Algorithm:</strong> {prediction.algorithm} (K-Nearest Neighbor)</p>
            <p><strong>Data Points:</strong> N/A recent price movements analyzed</p>
            <p><strong>Last Updated:</strong> {new Date().toLocaleString()}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}