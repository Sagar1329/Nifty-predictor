import pandas as pd
import joblib
import numpy as np

MODEL_FILE = "ml/models/y_trend_15m_rf.pkl"
FEATURE_FILE = "ml/training/X.csv"

LOOKBACK = 10

FEATURE_NAMES = [
    "log_return",
    "candle_body",
    "hl_range",
    "ema_9",
    "ema_21",
    "rsi_14",
    "volatility_10"
]

def build_feature_labels():
    labels = []
    for i in range(LOOKBACK):
        for name in FEATURE_NAMES:
            labels.append(f"{name}_t-{LOOKBACK - i}")
    return labels

def main():
    model = joblib.load(MODEL_FILE)
    X = pd.read_csv(FEATURE_FILE)

    importances = model.feature_importances_
    feature_labels = build_feature_labels()

    importance_df = pd.DataFrame({
        "feature": feature_labels,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        by="importance", ascending=False
    )

    print("\nTop 20 Important Features:\n")
    print(importance_df.head(20))

    print("\nGrouped Importance by Feature Type:\n")
    grouped = (
        importance_df
        .assign(base=lambda x: x["feature"].str.split("_t-").str[0])
        .groupby("base")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    print(grouped)

if __name__ == "__main__":
    main()
