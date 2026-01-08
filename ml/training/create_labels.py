import pandas as pd
import numpy as np

INPUT_FILE = "ml/data/nifty_5m_12m_clean.csv"
OUTPUT_FILE = "ml/training/labels.csv"

THRESHOLD = 0.0005  # 0.05%

def compute_trend(current_close, future_close):
    ret = (future_close - current_close) / current_close

    if ret > THRESHOLD:
        return 2   # Up
    elif ret < -THRESHOLD:
        return 0   # Down
    else:
        return 1   # Sideways

def create_labels():
    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])

    closes = df["close"].values

    labels = {
        "datetime":[],
        "y_dir": [],
        "y_trend_5m": [],
        "y_trend_10m": [],
        "y_trend_15m": []
    }

    max_horizon = 3  # 15 minutes = 3 candles

    for i in range(len(df) - max_horizon):
        # ----- Next candle direction -----
        labels["y_dir"].append(
            1 if closes[i + 1] > closes[i] else 0
        )

        # ----- Trend labels -----
        labels["y_trend_5m"].append(
            compute_trend(closes[i], closes[i + 1])
        )
        labels["y_trend_10m"].append(
            compute_trend(closes[i], closes[i + 2])
        )
        labels["y_trend_15m"].append(
            compute_trend(closes[i], closes[i + 3])
        )
        labels["datetime"].append(df.loc[i, "datetime"])


    label_df = pd.DataFrame(labels)

    label_df.to_csv(OUTPUT_FILE, index=False)
    print("Labels created:", label_df.shape)

if __name__ == "__main__":
    create_labels()
