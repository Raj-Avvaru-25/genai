# GenAI Stock Analysis Example

This repository contains a simple example script for analyzing the [Netflix Stock Price History](https://www.kaggle.com/datasets/adilshamim8/netflix-stock-price-history) dataset. The script loads the CSV data, plots the historical closing prices, and trains a basic LSTM model to predict the next closing price.

## Prerequisites

Install the required Python packages:

```bash
pip install pandas matplotlib scikit-learn tensorflow
```

## Usage

1. Download `NFLX.csv` from Kaggle and place it in this repository.
2. Run the analysis script:

```bash
python3 stock_analysis.py --data NFLX.csv
```

This will plot the closing price history and output a predicted next-day closing price.
