"""
Evaluate 15-minute trend model using replay predictions.

Phase A: Dataset construction ONLY
"""

from typing import Optional
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import pandas as pd
# ----------------------------
# Configuration
# ----------------------------

HORIZON_MINUTES = 15
THRESHOLD_PCT = 0.0005  # must match training
TIMEZONE = "Asia/Kolkata"




def tag_volatility(vol: float) -> str:
    if vol < 0.0008:
        return "LOW_VOL"
    elif vol < 0.0015:
        return "MEDIUM_VOL"
    else:
        return "HIGH_VOL"

def tag_rsi(rsi: float) -> str:
    if rsi < 30:
        return "OVERSOLD"
    elif rsi > 70:
        return "OVERBOUGHT"
    else:
        return "NEUTRAL"

def tag_time_of_day(ts: pd.Timestamp) -> str:
    hour = ts.hour
    minute = ts.minute

    if hour == 9 and minute < 45:
        return "OPEN"
    elif hour < 14 or (hour == 14 and minute < 30):
        return "MID"
    else:
        return "CLOSE"

def tag_trend(ema9: float, ema21: float) -> str:
    if ema9 > ema21:
        return "UP_TREND"
    else:
        return "DOWN_TREND"



def compute_coverage_metrics(df: pd.DataFrame) -> None:
    total = len(df)

    def summarize(name, subset):
        if subset.empty:
            return

        coverage = len(subset) / total
        accuracy = (subset["predicted"] == subset["actual"]).mean()

        print(f"\n{name}")
        print(f"  coverage: {coverage:.2f}")
        print(f"  accuracy: {accuracy:.2f}")
        print(f"  trades: {len(subset)}")

    summarize(
        "HIGH only",
        df[df["confidence_level"] == "HIGH"]
    )

    summarize(
        "HIGH + MEDIUM",
        df[df["confidence_level"].isin(["HIGH", "MEDIUM"])]
    )

    summarize(
        "ALL",
        df
    )


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
    df = add_indicators(df)
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_10"] = df["log_return"].rolling(10).std()

    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


# ----------------------------
# Build evaluation dataset
# ----------------------------

def build_evaluation_dataset(
    predictions: pd.DataFrame,
    candles: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # candles already indexed by datetime
    for _, row in predictions.iterrows():
        ts = pd.to_datetime(row["timestamp"])

        if ts not in candles.index:
            continue

        actual = compute_actual_trend(candles, ts)
        if actual is None:
            continue

        candle = candles.loc[ts]

        rows.append({
            "timestamp": ts,
            "predicted": row["prediction"],
            "actual": actual,
            "confidence_level": row["confidence_level"],

            # probabilities
            "p_up": row["p_up"],
            "p_sideways": row["p_sideways"],
            "p_down": row["p_down"],

            # -------- REGIMES --------
            "volatility_regime": tag_volatility(candle["volatility_10"]),
            "rsi_regime": tag_rsi(candle["rsi_14"]),
            "time_regime": tag_time_of_day(ts),
            "trend_regime": tag_trend(candle["ema_9"], candle["ema_21"]),
        })

    return pd.DataFrame(rows)




def add_confusion_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds TP / FP / TN / FN bucket column.
    Assumes:
    - predicted ∈ {UP, DOWN}
    - actual ∈ {UP, DOWN}
    """
    df = df.copy()

    def classify(row):
        if row["predicted"] == "UP" and row["actual"] == "UP":
            return "TP"
        if row["predicted"] == "UP" and row["actual"] == "DOWN":
            return "FP"
        if row["predicted"] == "DOWN" and row["actual"] == "DOWN":
            return "TN"
        if row["predicted"] == "DOWN" and row["actual"] == "UP":
            return "FN"
        return None  # should never happen

    df["bucket"] = df.apply(classify, axis=1)
    return df

# ----------------------------
# Metrics (NOT IMPLEMENTED YET)
# ----------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute baseline classification metrics for 15-minute trend prediction.

    Expected df columns:
    - actual
    - predicted
    """

    y_true = df["actual"]
    y_pred = df["predicted"]

    labels = ["DOWN", "SIDEWAYS", "UP"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),

        "recall_macro": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),

        "f1_macro": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }

    cm = confusion_matrix(
        y_true, y_pred, labels=labels
    )

    metrics["confusion_matrix"] = pd.DataFrame(
        cm,
        index=[f"actual_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )

    return metrics


def compute_metrics_by_confidence(df: pd.DataFrame) -> dict:
    """
    Compute metrics separately for each confidence level.
    """
    results = {}

    for level in ["HIGH", "MEDIUM", "LOW"]:
        subset = df[df["confidence_level"] == level]

        if subset.empty:
            results[level] = {"note": "no samples"}
            continue

        y_true = subset["actual"]
        y_pred = subset["predicted"]

        results[level] = {
            "count": len(subset),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred,
                labels=["DOWN", "SIDEWAYS", "UP"]
            ),
        }

    return results

def run_evaluation(
    predictions_path: Path,
    candles_path: Path,
):
    metrics = compute_metrics(eval_df)

    print("\nOverall Metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\n=== Metrics by Confidence Level ===")
    confidence_metrics = compute_metrics_by_confidence(eval_df)

    for level, metrics in confidence_metrics.items():
        print(f"\n[{level}]")
        for k, v in metrics.items():
            if k == "confusion_matrix":
                print("confusion_matrix:")
                print(pd.DataFrame(
                    v,
                    index=["actual_DOWN", "actual_SIDEWAYS", "actual_UP"],
                    columns=["pred_DOWN", "pred_SIDEWAYS", "pred_UP"]
                ))
            else:
                print(f"{k}: {v}")


# ----------------------------
# Debug run (dataset inspection only)
# ----------------------------

if __name__ == "__main__":
    predictions_path = Path("data/replay_predictions.csv")
    candles_path = Path("ml/data/nifty_5m_clean.csv")

    preds = load_replay_predictions(predictions_path)
    candles = load_candles(candles_path)

    eval_df = build_evaluation_dataset(preds, candles)
    eval_df = add_confusion_buckets(eval_df)

    # assert set(eval_df["bucket"].unique()) == {"TP", "FP", "TN", "FN"}
    # assert len(eval_df) == eval_df["bucket"].value_counts().sum()
    print(
    eval_df[
        ["volatility_regime", "rsi_regime", "time_regime", "trend_regime"]
    ].value_counts()
)

    print("\nBucket counts:")
    print(eval_df["bucket"].value_counts())

    print("Evaluation dataset preview:")
    print(eval_df.head())
    print("\nCounts:")
    print(eval_df["actual"].value_counts())

    run_evaluation(predictions_path, candles_path)

    fp_df = eval_df[eval_df["bucket"] == "FP"]
    fn_df = eval_df[eval_df["bucket"] == "FN"]

    print("\nFP by volatility:")
    print(fp_df["volatility_regime"].value_counts())

    print("\nFP by RSI:")
    print(fp_df["rsi_regime"].value_counts())

    print("\nFP by time of day:")
    print(fp_df["time_regime"].value_counts())

    print("\nFP by trend regime:")
    print(fp_df["trend_regime"].value_counts())

    print("\nFP combinations:")
    print(
        fp_df[
            ["volatility_regime", "rsi_regime", "time_regime", "trend_regime"]
        ].value_counts().head(10)
    )

    print("\nFN by volatility:")
    print(fn_df["volatility_regime"].value_counts())

    print("\nFN by RSI:")
    print(fn_df["rsi_regime"].value_counts())

    print("\nFN by time of day:")
    print(fn_df["time_regime"].value_counts())

    print("\nFN by trend regime:")
    print(fn_df["trend_regime"].value_counts())

    print("\nFN combinations:")
    print(
        fn_df[
            ["volatility_regime", "rsi_regime", "time_regime", "trend_regime"]
        ].value_counts().head(10)
    )
    print("\n=== Coverage-based Evaluation ===")
    compute_coverage_metrics(eval_df)



    




