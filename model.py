import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load data
data = pd.read_csv('crypto_data.csv', parse_dates=['timestamp'])
dates = data['timestamp']  # Use timestamp instead of date

# Convert all columns to numeric, dropping any non-numeric columns
numeric_data = data.apply(pd.to_numeric, errors='coerce')
numeric_data = numeric_data.dropna(axis=1)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Update sequence creation and model parameters
seq_length = 100  # Looking back 100 days
pred_length = 10  # Predicting next 10 days

def create_sequences(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:(i + seq_length)])  # 100 days of data
        y.append(data[i + seq_length:i + seq_length + pred_length, 3])  # Next 10 days eth_close
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(scaled_data, seq_length, pred_length)

# Split into train and test sets (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Update model architecture
model = Sequential([
    LSTM(200, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),
    LSTM(100, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(pred_length)  # Output layer predicts 10 values
])

model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions
# Create empty array matching original data shape
pred_data = np.zeros((y_pred.shape[0], y_pred.shape[1], numeric_data.shape[1]))
pred_data[:, :, 3] = y_pred  # Fill eth_close column
y_pred_rescaled = scaler.inverse_transform(pred_data[:, 0, :])[:, 3]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test[0], label='Actual')
plt.plot(y_pred[0], label='Predicted')
plt.legend()
plt.title('ETH Price Prediction (10 days)')
plt.show()

# Print evaluation metrics
mse = mean_squared_error(y_test[0], y_pred[0])
print(f'MSE: {mse}')

# Function to make future predictions
def predict_future(model, last_sequence):
    future_pred = model.predict(last_sequence.reshape(1, seq_length, -1))
    return future_pred[0]

# Get last sequence from data
last_sequence = scaled_data[-seq_length:]
future_prices = predict_future(model, last_sequence)

# Inverse transform predictions
future_data = np.zeros((1, pred_length, numeric_data.shape[1]))
future_data[0, :, 3] = future_prices
future_prices_rescaled = scaler.inverse_transform(future_data[0])[:, 3]

print("\nPredicted ETH prices for next 10 days:")
for i, price in enumerate(future_prices_rescaled, 1):
    print(f"Day {i}: ${price:.2f}")

# Get last 100 days of real prices and dates for visualization
real_prices = numeric_data.iloc[-seq_length:, 3].values  # Column 3 is eth_close
historical_dates = dates[-seq_length:]

# Create future dates
last_date = dates.iloc[-1]
future_dates = [last_date + timedelta(days=x) for x in range(1, pred_length + 1)]

# Update visualization
plt.figure(figsize=(15, 7))
plt.plot(historical_dates[-100:], real_prices[-100:],
         color='blue', label='Historical (100 days)', linewidth=2)
plt.plot(future_dates, future_prices_rescaled,
         color='red', label='Predicted (10 days)', linewidth=2, linestyle='--')

# Add styling
plt.title('ETH Price Prediction for Next 10 Days', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Rotate x-axis labels
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()