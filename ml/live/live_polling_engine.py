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
        self.last_seen_timestamp = None

        self.buffer = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close"]
        )

    # ----------------------------
    # Warm start
    # ----------------------------
    def _warm_start(self):
        df = self.provider.fetch_recent_candles()

        if df is None or df.empty:
            current_state_store.update({
                "status": "live",
                "phase": "no_data",
                "message": "No data returned from Yahoo during warm start.",
                "last_candle_time": None
            })
            return

        df = df.sort_values("datetime")
        self.buffer = df.tail(self.window_size).reset_index(drop=True)
        self.last_seen_timestamp = self.buffer["datetime"].iloc[-1]

        current_state_store.update({
            "status": "live",
            "phase": "market_closed",
            "message": "Market is closed. Waiting for new candles.",
            "last_candle_time": str(self.last_seen_timestamp)
        })

    # ----------------------------
    # Main loop
    # ----------------------------
    def start(self):
        self.running = True
        self._warm_start()

        while self.running:
            try:
                df = self.provider.fetch_recent_candles()

                # ----------------------------
                # No data returned
                # ----------------------------
                if df is None or df.empty:
                    current_state_store.update({
                        "status": "live",
                        "phase": "no_data",
                        "message": "No data returned from Yahoo.",
                        "last_candle_time": None
                    })
                    time.sleep(self.poll_seconds)
                    continue

                df = df.sort_values("datetime")
                latest_ts = df.iloc[-1]["datetime"]

                # ----------------------------
                # Market closed (stale candle)
                # ----------------------------
                if self.last_seen_timestamp is not None and latest_ts <= self.last_seen_timestamp:
                    current_state_store.update({
                        "status": "live",
                        "phase": "market_closed",
                        "message": "Market is closed. Waiting for new candles.",
                        "last_candle_time": str(latest_ts)
                    })
                    time.sleep(self.poll_seconds)
                    continue

                # ----------------------------
                # New candle(s)
                # ----------------------------
                new_candles = df[
                    df["datetime"] > (self.last_seen_timestamp or df.iloc[0]["datetime"])
                ]

                for _, candle in new_candles.iterrows():
                    self._process_candle(candle)
                    self.last_seen_timestamp = candle["datetime"]

            except Exception as e:
                current_state_store.update({
                    "status": "live",
                    "phase": "engine_error",
                    "message": f"Live engine error: {e}",
                    "last_candle_time": str(self.last_seen_timestamp)
                })

            time.sleep(self.poll_seconds)

    # ----------------------------
    # Stop
    # ----------------------------
    def stop(self):
        self.running = False
        current_state_store.update({
            "status": "stopped",
            "message": "Live mode stopped"
        })

    # ----------------------------
    # Candle processing
    # ----------------------------
    def _process_candle(self, candle: pd.Series):
        self.buffer = pd.concat(
            [self.buffer, candle.to_frame().T],
            ignore_index=True,
        ).tail(self.window_size)

        if len(self.buffer) < self.window_size:
            return

        try:
            result = self.predictor.predict(self.buffer)
        except Exception as e:
            current_state_store.update({
                "status": "live",
                "phase": "inference_error",
                "message": f"Inference failed: {e}",
                "last_candle_time": str(candle['datetime'])
            })
            return

        state = {
            "timestamp": candle["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            **result,
        }

        current_state_store.update(state)
        signal_history_store.append(state)
