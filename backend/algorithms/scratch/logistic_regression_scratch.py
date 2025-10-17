# Source: course assignment — Author: Ankit Kumar
# Written by Ankit — course implementation — do not modify

import numpy as np
import pandas as pd


class LogisticRegressionScratch:
    """Logistic Regression implementation from scratch using gradient descent"""
    
    def __init__(self, learning_rate=0.01, epochs=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.theta0 = 0.0
        self.theta1 = 0.0
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        
        Args:
            X: Feature values (1D array)
            y: Target values (1D array, binary: 0 or 1)
        """
        # Initialize parameters
        self.theta0 = 0
        self.theta1 = 0
        
        # Gradient descent
        for i in range(self.epochs):
            z = self.theta0 + self.theta1 * X
            pred = self.sigmoid(z)
            theta0_gradient = np.mean(pred - y)
            theta1_gradient = np.mean((pred - y) * X)
            self.theta0 -= self.learning_rate * theta0_gradient
            self.theta1 -= self.learning_rate * theta1_gradient
        
        # Calculate final loss
        z = self.theta0 + self.theta1 * X
        pred = self.sigmoid(z)
        self.loss = -np.mean(y * np.log(pred + 1e-10) + (1 - y) * np.log(1 - pred + 1e-10))
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities for new data
        
        Args:
            X: Feature values to predict on
            
        Returns:
            probabilities: Predicted probabilities
        """
        z = self.theta0 + self.theta1 * X
        return self.sigmoid(z)
    
    def predict(self, X):
        """
        Make binary predictions on new data
        
        Args:
            X: Feature values to predict on
            
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)
    
    def get_parameters(self):
        """Get model parameters"""
        return {
            'theta0': self.theta0,
            'theta1': self.theta1,
            'loss': self.loss
        }


def logistic_regression_demo():
    """Demo function showing usage of the scratch implementation"""
    # Example usage
    n = int(input())
    data = []
    
    for z in range(n):
        line = input().strip().split()
        data.append([int(line[0]), int(line[1])])
    
    data = pd.DataFrame(data, columns=['exam_score', 'admitted'])
    print("First 5 rows:")
    print(data.head(5))
    
    print(f"Shape (N, d): {data.shape}")
    print()
    print("Summary statistics for exam_score:")
    exam_score_min = np.min(data['exam_score'])
    exam_score_max = np.max(data['exam_score'])
    exam_score_mean = np.mean(data['exam_score'])
    exam_score_std = np.std(data['exam_score'])
    
    print(f"Min: {exam_score_min}")
    print(f"Max: {exam_score_max}")
    print(f"Mean: {exam_score_mean:.2f}")
    print(f"Std: {exam_score_std:.2f}")
    print()
    
    # Train model
    X = data['exam_score'].values
    y = data['admitted'].values
    
    model = LogisticRegressionScratch(learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    
    params = model.get_parameters()
    print(f"Final theta0: {params['theta0']:.2f}")
    print(f"Final theta1: {params['theta1']:.2f}")
    print(f"Final Loss: {params['loss']:.2f}")
    print()
    
    # Make predictions
    pred_65 = model.predict_proba(np.array([65]))[0]
    pred_155 = model.predict_proba(np.array([155]))[0]
    print(f"Prediction for exam_score=65: {pred_65:.2f}")
    print(f"Prediction for exam_score=155: {pred_155:.2f}")


if __name__ == "__main__":
    logistic_regression_demo()
