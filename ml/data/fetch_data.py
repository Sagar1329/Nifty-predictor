import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

OUTPUT_FILE = "ml/data/nifty_5m_raw_12m.csv"

SYMBOL = "^NSEI"
INTERVAL = "5m"
WINDOW_DAYS = 55        # Safe buffer under Yahoo 60-day limit
TOTAL_MONTHS = 12


def fetch_data():
    all_chunks = []

    end = datetime.utcnow()

    for _ in range(TOTAL_MONTHS):
        start = end - timedelta(days=WINDOW_DAYS)

        print(f"Fetching {start.date()} to {end.date()}")

        df = yf.download(
            SYMBOL,
            interval=INTERVAL,
            start=start,
            end=end,
            progress=False,
        )

        if not df.empty:
            all_chunks.append(df)
        else:
            print("Warning Empty chunk returned")

        end = start  # move window backward

    if not all_chunks:
        raise RuntimeError("No data fetched from Yahoo")

    full_df = pd.concat(all_chunks)
    full_df = full_df.sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="first")]

    full_df.to_csv(OUTPUT_FILE)
    print(f"Saved raw data: {OUTPUT_FILE}, rows={len(full_df)}")


if __name__ == "__main__":
    fetch_data()
