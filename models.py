#This script handles the time-series forecasting, training both LSTM and CNN models, and generating the prediction plots.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Input

# --- 1. Data Prep for Time Series ---
df = pd.read_csv('missile_attacks_daily.csv')
df['time_start'] = pd.to_datetime(df['time_start'], errors='coerce')
df = df.dropna(subset=['time_start'])
df['date'] = df['time_start'].dt.date

# Aggregate daily launches
daily_counts = df.groupby('date')['launched'].sum().reset_index()
daily_counts.set_index('date', inplace=True)
daily_counts.index = pd.to_datetime(daily_counts.index)
# Resample to ensure consecutive days (fill missing with 0)
daily_counts = daily_counts.resample('D').sum().fillna(0)

# Scaling
data = daily_counts.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Helper: Create Sequences
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 30
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train/Test Split (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# --- 2. LSTM Model ---
print("Training LSTM...")
model_lstm = Sequential()
model_lstm.add(Input(shape=(look_back, 1)))
model_lstm.add(LSTM(50, return_sequences=True))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

lstm_pred = model_lstm.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)

# --- 3. CNN Model ---
print("Training CNN...")
model_cnn = Sequential()
model_cnn.add(Input(shape=(look_back, 1)))
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(1))
model_cnn.compile(optimizer='adam', loss='mse')
model_cnn.fit(X_train, y_train, epochs=20, verbose=0)

cnn_pred = model_cnn.predict(X_test)
cnn_pred = scaler.inverse_transform(cnn_pred)

# --- 4. Plots ---

# Align dates for test data
test_dates = daily_counts.index[look_back + len(X_train):]

# Plot A: Standalone LSTM Forecast
plt.figure(figsize=(12, 6))
plt.plot(daily_counts.index, scaler.inverse_transform(scaled_data), label='Actual Data', alpha=0.5)
plt.plot(test_dates, lstm_pred, label='LSTM Prediction', color='orange')
plt.title('Missile Attacks Forecasting (LSTM Only)')
plt.xlabel('Date')
plt.ylabel('Launched Missiles')
plt.legend()
plt.savefig('lstm_forecast.png')
print("Generated lstm_forecast.png")
plt.close()

# Plot B: Model Comparison (LSTM vs CNN)
y_test_inv = scaler.inverse_transform([y_test])

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_inv.flatten(), label='Actual Data (Test)', color='blue')
plt.plot(test_dates, lstm_pred.flatten(), label='LSTM Prediction', color='orange')
plt.plot(test_dates, cnn_pred.flatten(), label='CNN Prediction', color='green')
plt.title('LSTM vs CNN Forecasting on Test Data')
plt.xlabel('Date')
plt.ylabel('Missiles Launched')
plt.legend()
plt.savefig('model_comparison.png')
print("Generated model_comparison.png")
plt.close()
