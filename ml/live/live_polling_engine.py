import time
import pandas as pd
from typing import Optional

from ml.data.yahoo_live_provider import YahooLiveDataProvider
from ml.inference.predictor import TrendPredictor
from ml.state.global_state import current_state_store, signal_history_store


class LivePollingEngine:
    def __init__(
        self,
        provider: YahooLiveDataProvider,
        predictor: TrendPredictor,
        window_size: int = 60,
        poll_seconds: int = 60,
    ):
        self.provider = provider
        self.predictor = predictor
        self.window_size = window_size
        self.poll_seconds = poll_seconds

        self.running = False
        self.last_seen_timestamp: Optional[pd.Timestamp] = None
        self.buffer = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close"]
        )

    def _warm_start(self):
        df = self.provider.fetch_recent_candles()

        if df.empty:
            return

        df = df.sort_values("datetime")
        self.buffer = df.tail(self.window_size).reset_index(drop=True)
        self.last_seen_timestamp = self.buffer["datetime"].iloc[-1]

    def start(self):
        self.running = True
        self._warm_start()

        while self.running:
            try:
                df = self.provider.fetch_recent_candles()

                if not df.empty:
                    df = df.sort_values("datetime")

                    new_candles = df[
                        df["datetime"] > self.last_seen_timestamp
                    ]

                    for _, candle in new_candles.iterrows():
                        self._process_candle(candle)
                        self.last_seen_timestamp = candle["datetime"]

            except Exception as e:
                # Fail-safe: log and continue next poll
                print(f"[LivePollingEngine] Error: {e}")

            time.sleep(self.poll_seconds)

    def stop(self):
        self.running = False

    def _process_candle(self, candle: pd.Series):
        self.buffer = pd.concat(
            [self.buffer, candle.to_frame().T],
            ignore_index=True,
        )

        self.buffer = self.buffer.tail(self.window_size)

        if len(self.buffer) < self.window_size:
            return

        result = self.predictor.predict(self.buffer)

        state = {
            "timestamp": candle["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            **result,
        }

        current_state_store.update(state)
        signal_history_store.append(state)
