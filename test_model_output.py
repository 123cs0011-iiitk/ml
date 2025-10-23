import numpy as np
import joblib

# Load the model
model_data = joblib.load('backend/models/linear_regression/linear_regression_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

print("Model coefficients range:", np.min(model.coef_), "to", np.max(model.coef_))
print("Model intercept:", model.intercept_)
print("\nScaler mean:", scaler.mean_[:5], "...")
print("Scaler std:", scaler.scale_[:5], "...")

# Create a test input (random features)
X_test = np.random.randn(1, 37)
print("\nTest input (first 5 features):", X_test[0, :5])

# Scale and predict
X_scaled = scaler.transform(X_test)
print("Scaled input (first 5 features):", X_scaled[0, :5])

prediction = model.predict(X_scaled)
print("\nPrediction:", prediction[0])
print("Expected range: -11 to +11")

