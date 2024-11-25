import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Step 1: Generate sample stock data
np.random.seed(42)  # For reproducibility

# Generate dates
dates = pd.date_range(start="2023-01-01", periods=100)

# Simulate stock prices
close_prices = np.cumsum(np.random.normal(0, 1, size=100)) + 150  # Random walk around 150
open_prices = close_prices + np.random.normal(0, 0.5, size=100)   # Open prices slightly different
high_prices = close_prices + np.random.uniform(0.5, 2.0, size=100)  # High prices
low_prices = close_prices - np.random.uniform(0.5, 2.0, size=100)   # Low prices
volume = np.random.randint(100000, 1000000, size=100)               # Random volume

# Create a DataFrame
stock_data = pd.DataFrame({
    "Date": dates,
    "Open": open_prices,
    "High": high_prices,
    "Low": low_prices,
    "Close": close_prices,
    "Volume": volume
})

# Save to CSV (optional)
stock_data.to_csv("sample_stock_data.csv", index=False)
print("Sample stock data generated.")

# Step 2: Add technical indicators
stock_data['MA_10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['RSI'] = 100 - (100 / (1 + (stock_data['Close'].diff().clip(lower=0).rolling(window=14).mean() /
                                       stock_data['Close'].diff().clip(upper=0).abs().rolling(window=14).mean())))

# Drop rows with NaN values (caused by rolling calculations)
stock_data = stock_data.dropna()

# Step 3: Define features (X) and target (y)
X = stock_data[['MA_10', 'MA_50', 'RSI']]
y = stock_data['Close']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Step 8: Visualize Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price', color='orange')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Sample Index')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
