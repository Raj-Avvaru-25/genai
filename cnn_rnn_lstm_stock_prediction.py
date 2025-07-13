"""Predict Netflix stock price using a hybrid CNN/RNN/LSTM model.

Usage:
    python3 cnn_rnn_lstm_stock_prediction.py --data NFLX.csv --date dd/mm/yyyy

The script trains a small neural network that combines a 1D convolutional
layer, an LSTM layer, and a SimpleRNN layer to forecast the closing price
on a specific future date. If the date is present in the dataset, the
actual closing price is shown instead.
"""

import argparse
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, SimpleRNN, Dense
from tensorflow.keras.models import Sequential


LOOK_BACK = 60


def load_data(path: str) -> pd.DataFrame:
    """Load stock price CSV sorted by date."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df


def plot_stock(df: pd.DataFrame) -> None:
    """Plot closing price history."""
    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["Close"], label="Close")
    plt.title("Netflix Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def prepare_data(series: np.ndarray, look_back: int = LOOK_BACK) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series modeling."""
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def build_model(look_back: int = LOOK_BACK) -> Sequential:
    """Construct a simple CNN/RNN/LSTM hybrid model."""
    model = Sequential(
        [
            Conv1D(32, kernel_size=3, activation="relu", input_shape=(look_back, 1)),
            MaxPooling1D(pool_size=2),
            LSTM(32, return_sequences=True),
            SimpleRNN(32),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(train_data: np.ndarray, epochs: int = 10, look_back: int = LOOK_BACK) -> Sequential:
    """Train the hybrid model."""
    X, y = prepare_data(train_data, look_back)
    model = build_model(look_back)
    early = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=32, callbacks=[early])
    return model


def forecast(model: Sequential, last_seq: np.ndarray, days: int, scaler: MinMaxScaler) -> float:
    """Forecast a number of days ahead and return the final predicted price."""
    seq = last_seq.copy()
    for _ in range(days):
        pred = model.predict(np.array([seq]), verbose=0)
        seq = np.vstack([seq[1:], pred])
    return scaler.inverse_transform(pred)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Netflix stock CSV")
    parser.add_argument("--date", required=True, help="Target date dd/mm/yyyy")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()

    df = load_data(args.data)
    plot_stock(df)

    target_date = datetime.strptime(args.date, "%d/%m/%Y")
    last_date = df["Date"].max()

    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    model = train_model(close_scaled, epochs=args.epochs)

    if target_date <= last_date:
        row = df[df["Date"] == target_date]
        if not row.empty:
            actual = row["Close"].iloc[0]
            print(f"Actual closing price on {target_date.date()}: {actual:.2f}")
        else:
            print("Date is within dataset but missing from file.")
        return

    days_ahead = (target_date - last_date).days
    last_seq = close_scaled[-LOOK_BACK:]
    predicted = forecast(model, last_seq, days_ahead, scaler)
    print(f"Predicted closing price on {target_date.date()}: {predicted:.2f}")


if __name__ == "__main__":
    main()