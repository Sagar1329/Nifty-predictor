import pandas as pd


class ReplayDataProvider:
    """
    Supplies candles sequentially from historical data.
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, parse_dates=["datetime"])
        self.df = self.df.sort_values("datetime").reset_index(drop=True)
        self.cursor = 0

    def has_next(self) -> bool:
        return self.cursor < len(self.df)

    def next_candle(self) -> pd.Series:
        if not self.has_next():
            raise StopIteration("Replay data exhausted")

        candle = self.df.iloc[self.cursor]
        self.cursor += 1
        return candle
