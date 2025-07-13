import argparse
from datetime import datetime, timedelta
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

LOOK_BACK = 60


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df


def prepare_sequences(data: np.ndarray, look_back: int = LOOK_BACK):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_model(series: np.ndarray, epochs: int = 10) -> Sequential:
    X, y = prepare_sequences(series)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    early = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=32, callbacks=[early])
    return model


def forecast_year(model: Sequential, last_seq: np.ndarray, start: datetime, end: datetime, scaler: MinMaxScaler) -> List[float]:
    seq = last_seq.copy()
    current = start
    preds: List[float] = []
    while current < end:
        current += timedelta(days=1)
        if current.weekday() >= 5:
            continue
        pred = model.predict(np.array([seq]), verbose=0)
        seq = np.vstack([seq[1:], pred])
        preds.append(scaler.inverse_transform(pred)[0][0])
    return preds


def plot_year(dates: List[datetime], prices: List[float], year: int) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(dates, prices, label="Predicted")
    plt.title(f"Predicted Netflix Closing Price {year}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Netflix closing prices for a given year")
    parser.add_argument("--data", required=True, help="Path to Netflix CSV data")
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
        plot_year(subset["Date"].tolist(), subset["Close"].tolist(), args.year)
        return

    last_seq = scaled_close[-LOOK_BACK:]
    preds = forecast_year(model, last_seq, last_date, year_end, scaler)
    dates = [d for d in pd.date_range(last_date + timedelta(days=1), year_end) if d.weekday() < 5]
    dates = [d for d in dates if d.year == args.year]
    plot_year(dates, preds[: len(dates)], args.year)


if __name__ == "__main__":
    main()