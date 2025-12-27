"""
Evaluate 15-minute trend model using replay predictions.

Phase A: Dataset construction ONLY
"""

from typing import Optional
import pandas as pd
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------

HORIZON_MINUTES = 15
THRESHOLD_PCT = 0.0005  # must match training
TIMEZONE = "Asia/Kolkata"


# ----------------------------
# Load replay predictions
# ----------------------------

def load_replay_predictions(path: Path) -> pd.DataFrame:
    """
    Expected input columns:
    - timestamp
    - prediction
    - confidence_level
    - probabilities (dict-like string) OR p_up/p_sideways/p_down
    """

    df = pd.read_csv(path)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(TIMEZONE)

    # Expand probabilities if needed
    if "probabilities" in df.columns:
        probs = df["probabilities"].apply(eval)
        df["p_down"] = probs.apply(lambda x: x.get("DOWN"))
        df["p_sideways"] = probs.apply(lambda x: x.get("SIDEWAYS"))
        df["p_up"] = probs.apply(lambda x: x.get("UP"))

    required_cols = {
        "timestamp",
        "prediction",
        "confidence_level",
        "p_up",
        "p_sideways",
        "p_down",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in replay predictions: {missing}")

    return df.sort_values("timestamp").reset_index(drop=True)


# ----------------------------
# Load candle data
# ----------------------------

def load_candles(path: Path) -> pd.DataFrame:
    """
    Expected columns:
    - datetime
    - open
    - high
    - low
    - close
    """

    df = pd.read_csv(path)

    df["datetime"] = (
        pd.to_datetime(df["datetime"])
        .dt.tz_localize(TIMEZONE)
    )

    df = df.sort_values("datetime").reset_index(drop=True)
    df.set_index("datetime", inplace=True)

    return df


# ----------------------------
# Compute actual future trend
# ----------------------------

def compute_actual_trend(
    candles: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> Optional[str]:
    """
    Returns UP / DOWN / SIDEWAYS
    or None if future candle not available
    """

    future_time = timestamp + pd.Timedelta(minutes=HORIZON_MINUTES)

    if timestamp not in candles.index or future_time not in candles.index:
        return None

    price_now = candles.loc[timestamp, "close"]
    price_future = candles.loc[future_time, "close"]

    pct_change = (price_future - price_now) / price_now

    if pct_change > THRESHOLD_PCT:
        return "UP"
    elif pct_change < -THRESHOLD_PCT:
        return "DOWN"
    else:
        return "SIDEWAYS"


# ----------------------------
# Build evaluation dataset
# ----------------------------

def build_evaluation_dataset(
    predictions: pd.DataFrame,
    candles: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in predictions.iterrows():
        ts = row["timestamp"]

        actual = compute_actual_trend(candles, ts)
        if actual is None:
            continue  # drop rows without future truth

        rows.append({
            "timestamp": ts,
            "predicted": row["prediction"],
            "actual": actual,
            "confidence_level": row["confidence_level"],
            "p_up": row["p_up"],
            "p_sideways": row["p_sideways"],
            "p_down": row["p_down"],
        })

    return pd.DataFrame(rows)


# ----------------------------
# Metrics (NOT IMPLEMENTED YET)
# ----------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    raise NotImplementedError


def compute_metrics_by_confidence(df: pd.DataFrame) -> dict:
    raise NotImplementedError


def run_evaluation(
    predictions_path: Path,
    candles_path: Path,
):
    raise NotImplementedError


# ----------------------------
# Debug run (dataset inspection only)
# ----------------------------

if __name__ == "__main__":
    predictions_path = Path("data/replay_predictions.csv")
    candles_path = Path("ml/data/nifty_5m_clean.csv")

    preds = load_replay_predictions(predictions_path)
    candles = load_candles(candles_path)

    eval_df = build_evaluation_dataset(preds, candles)

    print("Evaluation dataset preview:")
    print(eval_df.head())
    print("\nCounts:")
    print(eval_df["actual"].value_counts())
