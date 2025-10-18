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
  supervised: [
    { key: 'ann', label: 'Artificial Neural Network' },
    { key: 'arima', label: 'AutoRegressive Integrated Moving Average' },
    { key: 'cnn', label: 'Convolutional Neural Network' },
    { key: 'decision_tree', label: 'Decision Tree' },
    { key: 'knn', label: 'K-Nearest Neighbors' },
    { key: 'linear_regression', label: 'Linear Regression' },
    { key: 'logistic_regression', label: 'Logistic Regression' },
    { key: 'naive_bayes', label: 'Naive Bayes' },
    { key: 'random_forest', label: 'Random Forest' },
    { key: 'svm', label: 'Support Vector Machine' },
  ],
  unsupervised: [
    { key: 'autoencoders', label: 'Autoencoders' },
    { key: 'dbscan', label: 'Density-Based Spatial Clustering' },
    { key: 'general_clustering', label: 'General Clustering' },
    { key: 'hierarchical_clustering', label: 'Hierarchical Clustering' },
    { key: 'kmeans', label: 'K-Means Clustering' },
    { key: 'lazy_learning', label: 'Lazy Learning' },
    { key: 'pca', label: 'Principal Component Analysis' },
    { key: 'svd', label: 'Singular Value Decomposition' },
    { key: 'tsne', label: 't-Distributed Stochastic Neighbor Embedding' },
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
  const [showSupervised, setShowSupervised] = useState(true);
  const [showUnsupervised, setShowUnsupervised] = useState(false);

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
  // Skip loading state for testing - always show dummy data
  // if (loading) {
  //   return (
  //     <Card>
  //       <CardHeader>
  //         <CardTitle className="card-title-scaled card-title-with-icon flex items-center justify-between">
  //           <div className="flex items-center gap-4">
  //             <Brain className="card-icon-scaled" />
  //             <span>AI Price Prediction</span>
  //           </div>
  //           <div className="flex gap-1">
  //             {HORIZON_OPTIONS.map(option => (
  //               <Button
  //                 key={option.key}
  //                 variant={selectedHorizon === option.key ? 'default' : 'outline'}
  //                 size="sm"
  //                 onClick={() => handleHorizonChange(option.key)}
  //               >
  //                 {option.label}
  //               </Button>
  //             ))}
  //           </div>
  //         </CardTitle>
  //       </CardHeader>
  //       <CardContent className="space-y-6">
  //         <div className="space-y-2">
  //           <Skeleton className="h-6 w-32" />
  //           <Skeleton className="h-8 w-24" />
  //         </div>
  //         <Skeleton className="h-4 w-full" />
  //         <div className="grid grid-cols-2 gap-4">
  //           <Skeleton className="h-16 w-full" />
  //           <Skeleton className="h-16 w-full" />
  //         </div>
  //       </CardContent>
  //     </Card>
  //   );
  // }

  // Use dummy data for testing - log error to console for debugging
  if (error) {
    console.warn('Prediction error (using dummy data):', error);
  }

  // Always use dummy data for testing purposes
  const displayPrediction = DUMMY_PREDICTION;
  
  // Debug logging
  console.log('StockPrediction component rendering with dummy data:', displayPrediction);

  // Always show dummy data for testing
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
    if (confidence >= 70) return 'bg-green-500';
    if (confidence >= 50) return 'bg-orange-500';  // Changed from yellow to orange
    return 'bg-red-500';
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
      
      {/* Model Selection */}
      <div className="px-6 pb-4">
        <div className="space-y-4">
          <h3 className="text-2xl font-bold text-foreground">Models</h3>
          
          {/* Supervised Models */}
          <div className="space-y-2">
            <button
              onClick={() => setShowSupervised(!showSupervised)}
              className="flex items-center gap-2 text-base font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground transition-colors"
            >
              {showSupervised ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              Supervised Models
            </button>
            {showSupervised && (
              <div className="space-y-2">
                {/* First row - 3 buttons */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.supervised.slice(0, 3).map(model => (
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
                {/* Second row - 4 buttons */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.supervised.slice(3, 7).map(model => (
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
                {/* Third row - 3 buttons */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.supervised.slice(7).map(model => (
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
          
          {/* Unsupervised Models */}
          <div className="space-y-2">
            <button
              onClick={() => setShowUnsupervised(!showUnsupervised)}
              className="flex items-center gap-2 text-base font-medium text-muted-foreground uppercase tracking-wide hover:text-foreground transition-colors"
            >
              {showUnsupervised ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              Unsupervised Models
            </button>
            {showUnsupervised && (
              <div className="space-y-2">
                {/* First row - 4 buttons */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.unsupervised.slice(0, 4).map(model => (
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
                {/* Second row - 4 buttons */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.unsupervised.slice(4, 8).map(model => (
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
                {/* Third row - 1 button */}
                <div className="flex flex-wrap gap-6">
                  {MODEL_OPTIONS.unsupervised.slice(8).map(model => (
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
        </div>
      </div>
      
      <CardContent className="space-y-6">
        {/* Static Dummy Data Warning */}
        <Alert className="border-orange-200 bg-orange-50">
          <AlertTriangle className="h-4 w-4 text-orange-600" />
          <AlertDescription className="text-orange-800 font-semibold">
            ⚠️ STATIC DUMMY DATA - FOR TESTING ONLY
          </AlertDescription>
        </Alert>

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
          <Progress 
            value={displayPrediction.confidence || 0} 
            indicatorClassName={getProgressBarColor(displayPrediction.confidence)}
            className="h-2" 
          />
        </div>

        {/* Additional Metrics - Removed Price Range section */}

      </CardContent>
    </Card>
  );
}