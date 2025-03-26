import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("temperature_data.csv")
X = df[["Time"]].values
y = df["Temperature"].values

# Transform the features to polynomial features
degree = 3  # Polynomial degree
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
y_pred = model.predict(X_poly)

# Calculate accuracy (R² score)
r2 = r2_score(y, y_pred)
print(f"Model Accuracy (R² Score): {r2:.4f}")

# Save the model and polynomial transformer
joblib.dump(model, "model.pkl")
joblib.dump(poly, "poly_transform.pkl")
print("Model and transformer saved successfully.")
