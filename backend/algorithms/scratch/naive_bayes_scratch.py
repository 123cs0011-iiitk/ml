# Source: course assignment — Author: Ankit Kumar
# Written by Ankit — course implementation — do not modify

import math
import numpy as np
from collections import Counter


class NaiveBayesScratch:
    """Naive Bayes classifier implementation from scratch"""
    
    def __init__(self):
        self.priors = {}
        self.class_info = {}  # will contain mean and std for each class
        
    def gaussian_pdf(self, x, mean, std):
        """Calculate Gaussian probability density function"""
        if std == 0:  # avoid zero division
            return 1.0 if x == mean else 1e-9
        exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        
        Args:
            X: Feature matrix (2D array)
            y: Target labels (1D array)
        """
        # Calculate class priors
        class_counts = Counter(y)
        self.priors = {c: (class_counts[c] / len(y)) for c in class_counts}
        
        # Calculate mean and std for each class
        for c in class_counts:
            rows = [X[i] for i in range(len(X)) if y[i] == c]
            rows = np.array(rows)
            means = np.mean(rows, axis=0)
            stds = np.std(rows, axis=0, ddof=0)  # here used ddof for default population std
            self.class_info[c] = (means, stds)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for new data
        
        Args:
            X: Feature matrix to predict on
            
        Returns:
            probabilities: List of dictionaries with class probabilities
        """
        predictions = []
        for row in X:
            likelihoods = {}
            for c in self.class_info:
                means, stds = self.class_info[c]
                probs = [self.gaussian_pdf(row[i], means[i], stds[i]) for i in range(len(row))]
                likelihoods[c] = self.priors[c] * math.prod(probs)
            predictions.append(likelihoods)
        return predictions
    
    def predict(self, X):
        """
        Make class predictions for new data
        
        Args:
            X: Feature matrix to predict on
            
        Returns:
            predictions: List of predicted class labels
        """
        proba_predictions = self.predict_proba(X)
        return [max(probs, key=probs.get) for probs in proba_predictions]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and return metrics
        
        Args:
            X_test: Test feature matrix
            y_test: Test target labels
            
        Returns:
            metrics: Dictionary with accuracy, precision, recall, f1
        """
        predictions = self.predict(X_test)
        
        # Calculate confusion matrix components
        tp = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and y_test[i] == 1)
        tn = sum(1 for i in range(len(predictions)) if predictions[i] == 0 and y_test[i] == 0)
        fp = sum(1 for i in range(len(predictions)) if predictions[i] == 1 and y_test[i] == 0)
        fn = sum(1 for i in range(len(predictions)) if predictions[i] == 0 and y_test[i] == 1)
        
        accuracy = (tp + tn) / len(y_test)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions
        }


def naive_bayes_demo():
    """Demo function showing usage of the scratch implementation"""
    # Example usage
    n = int(input())
    data = []
    for _ in range(n):
        arr = input().split()
        data.append(arr)
    
    X, y = [], []
    for row in data:
        row = list(map(float, row))
        X.append(row[:-2])  # all features except last two
        y.append(int(row[-2]))  # second to last is the label
    
    # Split data
    split_size = int(0.7 * n)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]
    
    # Train model
    model = NaiveBayesScratch()
    model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class 0 Prior: {model.priors.get(0, 0):.2f}")
    print(f"Class 1 Prior: {model.priors.get(1, 0):.2f}")
    print("Predictions:", metrics['predictions'])
    print("Actual:", y_test)
    print(f"Accuracy={metrics['accuracy']:.2f}")
    print(f"Precision={metrics['precision']:.2f}")
    print(f"Recall={metrics['recall']:.2f}")
    print(f"F1={metrics['f1']:.2f}")


if __name__ == "__main__":
    naive_bayes_demo()
