import pandas as pd

INPUT_FILE = "ml/data/nifty_5m.csv"
OUTPUT_FILE = "ml/data/nifty_5m_clean.csv"

def clean_data():
    # Read CSV, skip first 2 header rows
    df = pd.read_csv(
        INPUT_FILE,
        skiprows=[0, 1]
    )

    # Rename columns
    df.columns = ["datetime", "close", "high", "low", "open", "volume"]

    # Drop volume (not useful for index)
    df = df.drop(columns=["volume"])

    # Convert datetime to pandas datetime (UTC)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Convert UTC → IST
    df["datetime"] = df["datetime"].dt.tz_convert("Asia/Kolkata")

    # Remove timezone info (keep naive IST)
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    # Sort by time (important)
    df = df.sort_values("datetime")

    # Reset index
    df = df.reset_index(drop=True)

    # Basic sanity check
    if df.isnull().any().any():
        raise RuntimeError("NaNs found after cleaning")

    df.to_csv(OUTPUT_FILE, index=False)
    print("Clean data saved:", df.shape)

if __name__ == "__main__":
    clean_data()
