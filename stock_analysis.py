# Script to analyze Netflix stock price dataset and train a simple LSTM model
# Usage: python3 stock_analysis.py --data netflix_stock_price.csv

import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def load_data(path: str) -> pd.DataFrame:
    """Load stock price CSV."""
    df = pd.read_csv(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    return df


def plot_stock(df: pd.DataFrame) -> None:
    """Plot closing price over time."""
    plt.figure(figsize=(10, 4))
    plt.plot(df['Date'], df['Close'], label='Close')
    plt.title('Netflix Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


def prepare_data(series: np.ndarray, look_back: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sequences for LSTM."""
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_lstm(train_data: np.ndarray, look_back: int = 60) -> Sequential:
    """Build and train LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    X, y = prepare_data(train_data, look_back)
    model.fit(X, y, epochs=10, batch_size=32)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to Netflix stock CSV')
    args = parser.parse_args()

    df = load_data(args.data)

    # Visualization
    plot_stock(df)

    # Use closing prices for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    model = train_lstm(close_scaled)

    # Predict next day price
    last_seq = close_scaled[-60:]
    pred = model.predict(np.array([last_seq]))
    next_price = scaler.inverse_transform(pred)[0][0]

    print(f'Next predicted closing price: {next_price:.2f}')


if __name__ == '__main__':
    main()
