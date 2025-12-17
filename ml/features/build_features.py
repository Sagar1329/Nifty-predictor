import pandas as pd
import numpy as np

INPUT_FILE = "ml/data/nifty_5m_clean.csv"
OUTPUT_FILE = "ml/features/features.csv"

LOOKBACK = 10

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_features():
    df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])

    # ----- Base features -----
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["candle_body"] = df["close"] - df["open"]
    df["hl_range"] = df["high"] - df["low"]

    # ----- Indicators -----
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["volatility_10"] = df["log_return"].rolling(10).std()

    # ----- Drop initial NaNs -----
    df = df.dropna().reset_index(drop=True)

    # ----- Build lookback window -----
    feature_cols = [
        "log_return",
        "candle_body",
        "hl_range",
        "ema_9",
        "ema_21",
        "rsi_14",
        "volatility_10"
    ]

    X = []

    for i in range(LOOKBACK, len(df)):
        window = df.loc[i - LOOKBACK:i - 1, feature_cols].values.flatten()
        X.append(window)

    feature_df = pd.DataFrame(X)

    feature_df.to_csv(OUTPUT_FILE, index=False)
    print("Features built:", feature_df.shape)

if __name__ == "__main__":
    build_features()
