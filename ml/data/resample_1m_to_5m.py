import pandas as pd
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------

INPUT_FILE = Path("ml/data/nifty_50_minute_data.csv")   # <-- update if needed
OUTPUT_FILE = Path("ml/data/nifty_5m_12m_clean.csv")

TIMEZONE = "Asia/Kolkata"

MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)

END_DATE = pd.Timestamp("2025-02-07", tz=TIMEZONE)
START_DATE = END_DATE - pd.DateOffset(months=12)


# ----------------------------
# Helpers
# ----------------------------

def is_market_time(ts: pd.Timestamp) -> bool:
    h, m = ts.hour, ts.minute
    return (
        (h > MARKET_OPEN[0] or (h == MARKET_OPEN[0] and m >= MARKET_OPEN[1]))
        and
        (h < MARKET_CLOSE[0] or (h == MARKET_CLOSE[0] and m <= MARKET_CLOSE[1]))
    )


# ----------------------------
# Main processing
# ----------------------------

def resample_1m_to_5m():
    print("Loading 1-minute data...")
    df = pd.read_csv(INPUT_FILE)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["date"])

    # Localize to IST (dataset already IST but treat explicitly)
    df["datetime"] = df["datetime"].dt.tz_localize(TIMEZONE)

    # Sort & filter time range
    df = df.sort_values("datetime")
    df = df[(df["datetime"] >= START_DATE) & (df["datetime"] <= END_DATE)]

    # Filter market hours only
    df = df[df["datetime"].apply(is_market_time)]

    # Set index for resampling
    df = df.set_index("datetime")

    print("Resampling to 5-minute candles...")

    df_5m = (
        df
        .resample("5T", label="right", closed="right")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })
        .dropna()
    )

    # Reset index and remove timezone (keep naive IST)
    df_5m = df_5m.reset_index()
    df_5m["datetime"] = df_5m["datetime"].dt.tz_localize(None)

    # ----------------------------
    # Sanity checks
    # ----------------------------
    if df_5m.isnull().any().any():
        raise RuntimeError("NaNs detected after resampling")

    # Rough candle count check (≈75 candles per day)
    days = df_5m["datetime"].dt.date.nunique()
    avg_candles = len(df_5m) / max(days, 1)

    print(f"Days covered: {days}")
    print(f"Total 5m candles: {len(df_5m)}")
    print(f"Avg candles/day: {avg_candles:.1f}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_5m.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    resample_1m_to_5m()
