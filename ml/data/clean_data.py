import pandas as pd

INPUT_FILE = "ml/data/nifty_5m_raw_12m.csv"
OUTPUT_FILE = "ml/data/nifty_5m_clean_12m.csv"

def clean_data():
    """
    Clean raw Yahoo 5m NIFTY data into model-ready format
    """

    # Yahoo CSV has multi-index header
    df = pd.read_csv(INPUT_FILE, header=[0, 1])

    # Flatten multi-level columns
    df.columns = [c[0].lower() for c in df.columns]

    # Expected columns now:
    # datetime, open, high, low, close, volume

    df = df.rename(columns={"datetime": "datetime"})

    # Keep only required columns
    df = df[["datetime", "open", "high", "low", "close"]]

    # Convert datetime to pandas datetime (UTC)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Convert UTC → IST
    df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata")

    # Remove timezone info (keep naive IST)
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    # Sort by time
    df = df.sort_values("datetime").reset_index(drop=True)

    # Sanity checks
    if df.isnull().any().any():
        raise RuntimeError("NaNs found after cleaning")

    if not df["datetime"].is_monotonic_increasing:
        raise RuntimeError("Datetime not sorted correctly")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Clean data saved: {OUTPUT_FILE}, rows={len(df)}")


if __name__ == "__main__":
    clean_data()
