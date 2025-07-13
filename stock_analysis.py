{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM84xFl5LvimRVVa6zeY2US",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Raj-Avvaru-25/genai/blob/main/stock_analysis.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zwqOrPs5uKtt"
      },
      "outputs": [],
      "source": [
        "\n",
        "import argparse\n",
        "from typing import Tuple\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import LSTM, Dense\n",
        "\n",
        "\n",
        "def load_data(path: str) -> pd.DataFrame:\n",
        "    \"\"\"Load stock price CSV.\"\"\"\n",
        "    df = pd.read_csv(path, parse_dates=['Date'])\n",
        "    df.sort_values('Date', inplace=True)\n",
        "    return df\n",
        "\n",
        "\n",
        "def plot_stock(df: pd.DataFrame) -> None:\n",
        "    \"\"\"Plot closing price over time.\"\"\"\n",
        "    plt.figure(figsize=(10, 4))\n",
        "    plt.plot(df['Date'], df['Close'], label='Close')\n",
        "    plt.title('Netflix Closing Price')\n",
        "    plt.xlabel('Date')\n",
        "    plt.ylabel('Price')\n",
        "    plt.legend()\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "\n",
        "def prepare_data(series: np.ndarray, look_back: int = 60) -> Tuple[np.ndarray, np.ndarray]:\n",
        "    \"\"\"Prepare sequences for LSTM.\"\"\"\n",
        "    X, y = [], []\n",
        "    for i in range(look_back, len(series)):\n",
        "        X.append(series[i - look_back:i])\n",
        "        y.append(series[i])\n",
        "    return np.array(X), np.array(y)\n",
        "\n",
        "\n",
        "def train_lstm(train_data: np.ndarray, look_back: int = 60) -> Sequential:\n",
        "    \"\"\"Build and train LSTM model.\"\"\"\n",
        "    model = Sequential([\n",
        "        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),\n",
        "        LSTM(50),\n",
        "        Dense(1)\n",
        "    ])\n",
        "    model.compile(optimizer='adam', loss='mean_squared_error')\n",
        "    X, y = prepare_data(train_data, look_back)\n",
        "    model.fit(X, y, epochs=10, batch_size=32)\n",
        "    return model\n",
        "\n",
        "\n",
        "def main() -> None:\n",
        "    parser = argparse.ArgumentParser()\n",
        "    parser.add_argument('--data', required=True, help='Path to Netflix stock CSV')\n",
        "    args = parser.parse_args()\n",
        "\n",
        "    df = load_data(args.data)\n",
        "\n",
        "    # Visualization\n",
        "    plot_stock(df)\n",
        "\n",
        "    # Use closing prices for prediction\n",
        "    scaler = MinMaxScaler(feature_range=(0, 1))\n",
        "    close_scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))\n",
        "\n",
        "    model = train_lstm(close_scaled)\n",
        "\n",
        "    # Predict next day price\n",
        "    last_seq = close_scaled[-60:]\n",
        "    pred = model.predict(np.array([last_seq]))\n",
        "    next_price = scaler.inverse_transform(pred)[0][0]\n",
        "\n",
        "    print(f'Next predicted closing price: {next_price:.2f}')\n",
        "\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    main()"
      ]
    }
  ]
}