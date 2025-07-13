"""Generate a year-long prediction graph using a CNN/RNN/LSTM hybrid model.

The script trains a small network combining convolutional and recurrent layers
on Netflix stock closing prices. If the requested year is contained within the
CSV data, the script simply plots the actual closing prices for that year.
Otherwise it forecasts closing prices for each trading day until the end of the
target year and displays the predicted series.

Usage:
    python3 cnn_rnn_lstm_year_prediction.py --data NFLX.csv --year 2027
"""

import argparse
from datetime import datetime, timedelta
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, SimpleRNN, Dense
from tensorflow.keras.models import Sequential

LOOK_BACK = 60


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df


def prepare_data(series: np.ndarray, look_back: int = LOOK_BACK) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def build_model(look_back: int = LOOK_BACK) -> Sequential:
    model = Sequential([
        Conv1D(32, kernel_size=3, activation="relu", input_shape=(look_back, 1)),
        MaxPooling1D(pool_size=2),
        LSTM(32, return_sequences=True),
        SimpleRNN(32),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(train_data: np.ndarray, epochs: int = 10, look_back: int = LOOK_BACK) -> Sequential:
    X, y = prepare_data(train_data, look_back)
    model = build_model(look_back)
    early = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=32, callbacks=[early])
    return model


def forecast_dates(model: Sequential, last_seq: np.ndarray, start_date: datetime, end_date: datetime, scaler: MinMaxScaler) -> List[Tuple[datetime, float]]:
    seq = last_seq.copy()
    current = start_date
    preds: List[Tuple[datetime, float]] = []
    while current < end_date:
        current += timedelta(days=1)
        if current.weekday() >= 5:
            continue
        pred = model.predict(np.array([seq]), verbose=0)
        seq = np.vstack([seq[1:], pred])
        price = scaler.inverse_transform(pred)[0][0]
        preds.append((current, price))
    return preds


def plot_series(dates: List[datetime], prices: List[float], title: str, label: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(dates, prices, label=label)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Netflix stock CSV")
    parser.add_argument("--year", type=int, required=True, help="Target year")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()

    df = load_data(args.data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    model = train_model(scaled_close, epochs=args.epochs)

    year_start = datetime(args.year, 1, 1)
    year_end = datetime(args.year, 12, 31)
    last_date = df["Date"].max()

    if year_end <= last_date:
        subset = df[(df["Date"] >= year_start) & (df["Date"] <= year_end)]
        if subset.empty:
            print("No data for that year in the dataset.")
            return
        plot_series(subset["Date"].tolist(), subset["Close"].tolist(), f"Netflix Closing Price {args.year}", "Actual")
        return

    last_seq = scaled_close[-LOOK_BACK:]
    preds = forecast_dates(model, last_seq, last_date, year_end, scaler)

    dates = [d for d, _ in preds if d.year == args.year]
    prices = [p for d, p in preds if d.year == args.year]
    if not dates:
        print("No predictions generated for that year.")
        return

    plot_series(dates, prices, f"Predicted Netflix Closing Price {args.year}", "Predicted")


if __name__ == "__main__":
    main()