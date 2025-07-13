"""Generate a stock prediction graph for a given year.

This script loads the Netflix stock price history, trains an LSTM model,
and forecasts closing prices for a future year if the data is not
available. If the selected year is within the dataset, it simply plots
the actual closing prices for that year.

Usage:
    python3 stock_year_prediction.py --data NFLX.csv --year 2026
"""

import argparse
from datetime import datetime, timedelta
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


LOOK_BACK = 60


def load_data(path: str) -> pd.DataFrame:
    """Load the stock price CSV file sorted by date."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df


def prepare_data(series: np.ndarray, look_back: int = LOOK_BACK) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences of previous closing prices for LSTM."""
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_lstm(train_data: np.ndarray, *, epochs: int = 10) -> Sequential:
    """Build and train a simple LSTM model."""
    model = Sequential(
        [LSTM(50, return_sequences=True, input_shape=(LOOK_BACK, 1)), LSTM(50), Dense(1)]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    X, y = prepare_data(train_data, LOOK_BACK)
    early = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=32, callbacks=[early])
    return model


def forecast_dates(
    model: Sequential,
    last_seq: np.ndarray,
    start_date: datetime,
    end_date: datetime,
    scaler: MinMaxScaler,
) -> List[Tuple[datetime, float]]:
    """Forecast closing prices from the day after ``start_date`` until ``end_date``."""

    seq = last_seq.copy()
    current = start_date
    preds: List[Tuple[datetime, float]] = []

    while current < end_date:
        current += timedelta(days=1)
        if current.weekday() >= 5:  # Skip weekends
            continue
        pred = model.predict(np.array([seq]), verbose=0)
        seq = np.vstack([seq[1:], pred])
        price = scaler.inverse_transform(pred)[0][0]
        preds.append((current, price))

    return preds


def plot_series(dates: List[datetime], values: List[float], title: str, label: str) -> None:
    """Helper to plot a date series."""
    plt.figure(figsize=(10, 4))
    plt.plot(dates, values, label=label)
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
    close_scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    model = train_lstm(close_scaled, epochs=args.epochs)

    year_start = datetime(args.year, 1, 1)
    year_end = datetime(args.year, 12, 31)
    last_date = df["Date"].max()

    if year_end <= last_date:
        # Year fully contained in dataset
        subset = df[(df["Date"] >= year_start) & (df["Date"] <= year_end)]
        if subset.empty:
            print("No data for that year in the dataset.")
            return
        plot_series(subset["Date"].tolist(), subset["Close"].tolist(), f"Netflix Closing Price {args.year}", "Actual")
        return

    # Forecast future dates until the end of the requested year
    last_seq = close_scaled[-LOOK_BACK:]
    preds = forecast_dates(model, last_seq, last_date, year_end, scaler)

    # Only keep predictions for the target year
    dates = [d for d, _ in preds if d.year == args.year]
    prices = [p for d, p in preds if d.year == args.year]

    if not dates:
        print("No predictions generated for that year.")
        return

    plot_series(dates, prices, f"Predicted Netflix Closing Price {args.year}", "Predicted")


if __name__ == "__main__":
    main()