This repository contains a simple example script for analyzing the [Netflix Stock Price History](https://www.kaggle.com/datasets/adilshamim8/netflix-stock-price-history) dataset. The script loads the CSV data, plots the historical closing prices, and trains a basic LSTM model to predict the next closing price. Training uses early stopping to prevent overfitting.

## Prerequisites

Install the required Python packages:

```bash
pip install pandas matplotlib scikit-learn tensorflow
```

## Usage

1. Download `NFLX.csv` from Kaggle and place it in this repository.
2. Run the analysis script with a target date. You can also set the number of training epochs:

```bash
python3 cnn_rnn_lstm_stock_prediction.py --data NFLX.csv --date 01/12/2025 --epochs 15
```

This will plot the closing price history and output a predicted closing price for the specified date (or the actual price if the date already exists in the file).

### Predicting an Entire Year

To generate a prediction graph for a future year, use the `stock_year_prediction.py` script. If the year is already contained in the dataset, the script simply plots the actual closing prices for that year.

```bash
python3 stock_year_prediction.py --data NFLX.csv --year 2026
```

The script trains the same LSTM model and forecasts closing prices for each trading day until the end of the requested year, then displays the resulting plot.

### Alternate Year Forecast Script

`year_prediction_graph.py` provides a similar forecast in a standalone example.
Use it like so:

```bash
python3 year_prediction_graph.py --data NFLX.csv --year 2027
```

It will train an LSTM model and plot predicted closing prices for each trading day in the chosen year.

### Hybrid CNN/RNN/LSTM Predictor

For a model that combines convolutional and recurrent layers, run the
`cnn_rnn_lstm_stock_prediction.py` script. Provide a target date to forecast
the closing price on that day:

```bash
python3 cnn_rnn_lstm_stock_prediction.py --data NFLX.csv --date 01/12/2025
```

The script trains a simple CNN/RNN/LSTM network and outputs the predicted price
for the requested date.


### Hybrid Year Prediction Graph

`cnn_rnn_lstm_year_prediction.py` extends the hybrid model to forecast an entire year. Run it with a target year:

```bash
python3 cnn_rnn_lstm_year_prediction.py --data NFLX.csv --year 2027
```

If the chosen year already exists in the dataset, the script displays the actual prices for that year. Otherwise it predicts the closing price for each trading day and plots the results.
