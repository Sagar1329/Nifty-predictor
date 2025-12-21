import pandas as pd
import yfinance as yf
from typing import List


class YahooLiveDataProvider:
    """
    Fetches recent intraday candles from Yahoo Finance.
    Designed for polling-based live ingestion.
    """

    def __init__(
        self,
        ticker: str = "^NSEI",
        interval: str = "5m",
        lookback_period: str = "1d",
    ):
        self.ticker = ticker
        self.interval = interval
        self.lookback_period = lookback_period

    def fetch_recent_candles(self) -> pd.DataFrame:
        """
        Fetch recent intraday candles and return normalized OHLC dataframe.
        Columns: datetime, open, high, low, close
        """

        df = yf.download(
            tickers=self.ticker,
            period=self.lookback_period,
            interval=self.interval,
            progress=False,
        )

        if df.empty:
            return pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close"]
            )

        # Handle multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Normalize column names
        df = df.rename(
            columns={
                "Datetime": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
            }
        )

        df = df[["datetime", "open", "high", "low", "close"]]
        df = df.sort_values("datetime").reset_index(drop=True)

        return df
