import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Brain, AlertTriangle, ChevronDown, ChevronRight } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Skeleton } from './ui/skeleton';
import { Alert, AlertDescription } from './ui/alert';
import { Button } from './ui/button';
import { PredictionResult } from '../services/stockService';
import { formatPrice, Currency } from '../utils/currency';

// Static dummy data for testing purposes
const DUMMY_PREDICTION: PredictionResult = {
  predictedPrice: 52800.00,
  confidence: 73.5,
  model: 'KNN',
  timeframe: '5 years',
  priceRange: [50160.00, 55440.00],
  timeFrameDays: 1825,
  modelInfo: {
    model: 'K-Nearest Neighbors'
  },
  dataPointsUsed: 1250,
  lastUpdated: '2025-10-18T20:00:00Z',
  currency: 'USD'
};

interface StockPredictionProps {
  prediction: PredictionResult | null;
  currentPrice?: number;
  loading: boolean;
  symbol: string;
  error: string;
  currency: Currency;
  onHorizonChange?: (horizon: string) => void;
  onModelChange?: (model: string) => void;
}

const HORIZON_OPTIONS = [
  { key: '1d', label: '1D' },
  { key: '1w', label: '1W' },
  { key: '1m', label: '1M' },
  { key: '1y', label: '1Y' },
  { key: '5y', label: '5Y' },
];

const MODEL_OPTIONS = {
  basic: [
    { key: 'decision_tree', label: 'Decision Tree' },
    { key: 'linear_regression', label: 'Linear Regression' },
    { key: 'random_forest', label: 'Random Forest' },
    { key: 'svm', label: 'Support Vector Machine' },
  ],
  advanced: [
    { key: 'ann', label: 'Artificial Neural Network' },
    { key: 'arima', label: 'AutoRegressive Integrated Moving Average' },
    { key: 'autoencoders', label: 'Autoencoders' },
    { key: 'cnn', label: 'Convolutional Neural Network' },
    { key: 'knn', label: 'K-Nearest Neighbors' },
  ]
};

export function StockPrediction({ 
  prediction, 
  currentPrice, 
  loading, 
  symbol, 
  error,
  currency,
  onHorizonChange,
  onModelChange
}: StockPredictionProps) {
  const [selectedHorizon, setSelectedHorizon] = useState('5y');
  const [selectedModel, setSelectedModel] = useState<string>('knn');
  const [showBasic, setShowBasic] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleHorizonChange = (horizon: string) => {
    setSelectedHorizon(horizon);
    if (onHorizonChange) {
      onHorizonChange(horizon);
    }
  };

  const handleModelToggle = (model: string) => {
    setSelectedModel(model);
    if (onModelChange) {
      onModelChange(model);
    }
  };
  
  // Use dummy data for testing - log error to console for debugging
  if (error) {
    console.warn('Prediction error (using dummy data):', error);
  }

  // Always use dummy data for testing purposes
  const displayPrediction = DUMMY_PREDICTION;
  
  // Debug logging
  console.log('StockPrediction component rendering with dummy data:', displayPrediction);
  const isPositiveChange = displayPrediction.predictedPrice > (currentPrice || 0);
  const change = (currentPrice && displayPrediction.predictedPrice) ? displayPrediction.predictedPrice - currentPrice : 0;
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

  const getProgressBarColor = (confidence: number | undefined | null) => {
    if (!confidence || isNaN(confidence)) return 'bg-gray-400';
    if (confidence >= 70) return 'bg-green-600';
    if (confidence >= 50) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getProgressBarStyle = (confidence: number | undefined | null) => {
    if (!confidence || isNaN(confidence)) return { backgroundColor: '#9ca3af' };
    if (confidence >= 70) return { backgroundColor: '#16a34a' }; // green-600
    if (confidence >= 50) return { backgroundColor: '#f97316' }; // orange-500
    return { backgroundColor: '#ef4444' }; // red-500
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
            <span className="prediction-metadata-label text-muted-foreground">Predicted Price</span>
            <div className="flex items-center gap-2">
              {isPositiveChange ? (
                <TrendingUp className="prediction-icon text-green-600" />
              ) : (
                <TrendingDown className="prediction-icon text-red-600" />
              )}
              <Badge variant={getConfidenceBadgeVariant(displayPrediction.confidence)} className="prediction-confidence-badge">
                {displayPrediction.confidence ? displayPrediction.confidence.toFixed(1) : '0.0'}% confidence
              </Badge>
            </div>
          </div>
          
          <div className="prediction-price-main">
            {formatPrice(displayPrediction.predictedPrice, currency)}
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
            <span className={`prediction-metadata-value ${getConfidenceColor(displayPrediction.confidence)}`}>
              {displayPrediction.confidence ? displayPrediction.confidence.toFixed(1) : '0.0'}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2" style={{ backgroundColor: '#e5e7eb' }}>
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${getProgressBarColor(displayPrediction.confidence)}`}
              style={{ 
                width: `${displayPrediction.confidence || 0}%`,
                ...getProgressBarStyle(displayPrediction.confidence)
              }}
            />
          </div>
        </div>

        {/* Additional Metrics - Removed Price Range section */}

      </CardContent>
      
      {/* Model Selection */}
      <div className="px-6 pb-16">
        <div className="space-y-4">
          <h3 className="text-2xl font-bold text-foreground">Models</h3>
          
          {/* Basic Models */}
          <div className="space-y-2">
            <button
              onClick={() => setShowBasic(!showBasic)}
              className="flex items-center gap-2 text-base font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground transition-colors"
            >
              {showBasic ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              Basic Models
            </button>
            {showBasic && (
              <div className="space-y-2">
                {/* Single row - all 4 buttons */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.basic.map(model => (
                    <Button
                      key={model.key}
                      variant={selectedModel === model.key ? 'default' : 'outline'}
                      size="default"
                      onClick={() => handleModelToggle(model.key)}
                      className="text-lg model-toggle-button"
                    >
                      {model.label}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Advanced Models */}
          <div className="space-y-2 mb-12">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-base font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground transition-colors"
            >
              {showAdvanced ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              Advanced Models
            </button>
            {showAdvanced && (
              <div className="space-y-2">
                {/* First row - 3 buttons */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.advanced.slice(0, 3).map(model => (
                    <Button
                      key={model.key}
                      variant={selectedModel === model.key ? 'default' : 'outline'}
                      size="default"
                      onClick={() => handleModelToggle(model.key)}
                      className="text-lg model-toggle-button"
                    >
                      {model.label}
                    </Button>
                  ))}
                </div>
                {/* Second row - 2 buttons */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.advanced.slice(3).map(model => (
                    <Button
                      key={model.key}
                      variant={selectedModel === model.key ? 'default' : 'outline'}
                      size="default"
                      onClick={() => handleModelToggle(model.key)}
                      className="text-lg model-toggle-button"
                    >
                      {model.label}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Bottom separator and spacing */}
          <div className="border-t border-gray-200 mt-6 pt-6"></div>
        </div>
      </div>
    </Card>
  );
}