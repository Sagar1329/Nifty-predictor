"""
Evaluate 15-minute trend model using replay predictions.

Phase A: Dataset construction ONLY
"""

from typing import Optional
import pandas as pd
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

        # Compute actual future trend
        actual = compute_actual_trend(candles, ts)
        if actual is None:
            continue

        predicted = row["prediction"]

        # ----------------------------
        # FILTERING LOGIC (CRITICAL)
        # ----------------------------

        # Drop SIDEWAYS from ground truth
        if actual not in {"UP", "DOWN"}:
            continue

        # Drop abstained predictions
        if predicted not in {"UP", "DOWN"}:
            continue

        rows.append({
            "timestamp": ts,
            "predicted": predicted,
            "actual": actual,
            "confidence_level": row["confidence_level"],
            "p_up": row["p_up"],
            "p_sideways": row["p_sideways"],
            "p_down": row["p_down"],
        })

    df = pd.DataFrame(rows)

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

    print("Evaluation dataset preview:")
    print(eval_df.head())
    print("\nCounts:")
    print(eval_df["actual"].value_counts())

    run_evaluation(predictions_path, candles_path)
