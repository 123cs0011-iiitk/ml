# Source: course assignment — Author: Ankit Kumar
# Written by Ankit — course implementation — do not modify

import numpy as np


class LinearRegressionScratch:
    """Linear Regression implementation from scratch using gradient descent"""
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.feature_mean = 0.0
        self.feature_std = 1.0
        
    def normalize_features(self, X):
        """Normalize features using z-score normalization"""
        self.feature_mean = np.mean(X)
        self.feature_std = np.std(X)
        return (X - self.feature_mean) / self.feature_std
    
    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent
        
        Args:
            X: Feature values (1D array)
            y: Target values (1D array)
        """
        # Normalize features
        X_norm = self.normalize_features(X)
        
        # Initialize parameters
        self.theta0 = 0.0
        self.theta1 = 0.0
        
        # Gradient descent
        for i in range(self.epochs):
            pred = self.theta0 + (self.theta1) * X_norm
            errors = pred - y
            grad_theta0 = np.mean(errors)
            grad_theta1 = np.mean(errors * X_norm)
            self.theta0 = self.theta0 - (self.learning_rate * grad_theta0)
            self.theta1 = self.theta1 - (self.learning_rate * grad_theta1)
        
        # Calculate final MSE
        final_pred = self.theta0 + self.theta1 * X_norm
        self.mse = np.mean(((final_pred - y) ** 2) / 2)
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Feature values to predict on
            
        Returns:
            predictions: Predicted values
        """
        X_norm = (X - self.feature_mean) / self.feature_std
        return self.theta0 + self.theta1 * X_norm
    
    def get_parameters(self):
        """Get model parameters"""
        return {
            'theta0': self.theta0,
            'theta1': self.theta1,
            'mse': self.mse,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std
        }


def linear_regression_demo():
    """Demo function showing usage of the scratch implementation"""
    # Example usage
    n = int(input())
    data = []
    for i in range(n):
        line = input().strip().split()
        data.append([float(line[0]), float(line[1])])
    
    data = np.array(data)
    
    # Print first 5 rows
    for i in range(min(5, n)):
        print(f"{data[i, 0]:.1f} {data[i, 1]:.1f}")
    
    print(f"Shape: {data.shape}")
    
    # Print statistics
    for col in range(2):
        mean = np.mean(data[:, col])
        std = np.std(data[:, col])
        mini = np.min(data[:, col])
        maxi = np.max(data[:, col])
        print(f"{mean:.2f} {std:.2f} {mini:.2f} {maxi:.2f}")
    
    # Train model
    X = data[:, 0]
    y = data[:, 1]
    
    model = LinearRegressionScratch(learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    
    params = model.get_parameters()
    print(f"Final theta0={params['theta0']:.3f} | theta1={params['theta1']:.3f} | Final MSE={params['mse']:.2f}")
    
    # Make predictions
    pred_150 = model.predict(np.array([150]))
    pred_200 = model.predict(np.array([200]))
    print(f"Prediction for 150: {pred_150[0]:.2f}")
    print(f"Prediction for 200: {pred_200[0]:.2f}")


if __name__ == "__main__":
    linear_regression_demo()
