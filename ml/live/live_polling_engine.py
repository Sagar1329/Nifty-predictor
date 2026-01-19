import time
import pandas as pd
from datetime import timedelta
import pytz

from ml.data.yahoo_live_provider import YahooLiveDataProvider
from ml.inference.predictor import TrendPredictor
from ml.state.global_state import current_state_store, signal_history_store

IST = pytz.timezone("Asia/Kolkata")
MIN_BUFFER_FOR_PREDICTION = 60  # MUST match model requirement


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

        self.last_seen_candle_ts = None
        self.last_predicted_candle_ts = None

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

        # Normalize to IST
        df["datetime"] = (
            pd.to_datetime(df["datetime"], utc=True)
            .dt.tz_convert(IST)
        )

        df = df.sort_values("datetime")

        self.buffer = df.tail(self.window_size).reset_index(drop=True)
        self.last_seen_candle_ts = self.buffer["datetime"].iloc[-1]

        # NEW: explicit warm-up state
        if len(self.buffer) < MIN_BUFFER_FOR_PREDICTION:
            current_state_store.update({
                "status": "live",
                "phase": "warming_up",
                "message": "Warming up buffer with historical candles",
                "buffer_size": len(self.buffer),
                "required": MIN_BUFFER_FOR_PREDICTION,
                "last_candle_time": self.last_seen_timestamp.strftime("%Y-%m-%d %H:%M:%S")
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
                # No data
                # ----------------------------
                if df is None or df.empty:
                    time.sleep(self.poll_seconds)
                    continue

                # ----------------------------
                # Normalize datetime
                # ----------------------------
                df["datetime"] = (
                    pd.to_datetime(df["datetime"], utc=True)
                    .dt.tz_convert(IST)
                )
                df = df.sort_values("datetime")

                latest_candle_ts = df.iloc[-1]["datetime"]

                # ----------------------------
                # FIRST: detect new candles safely
                # ----------------------------
                if self.last_seen_candle_ts is not None:
                    new_candles = df[df["datetime"] > self.last_seen_candle_ts]
                else:
                    # Safety on first iteration / bad warm start
                    new_candles = df.tail(0)

                if not new_candles.empty:
                    for _, candle in new_candles.iterrows():
                        self._process_candle(candle)
                        self.last_seen_candle_ts = candle["datetime"]

                    time.sleep(self.poll_seconds)
                    continue

                # ----------------------------
                # No new candle → update status
                # ----------------------------
                if self.last_seen_candle_ts is not None:
                    current_state_store.update({
                        "status": "live",
                        "phase": "waiting_for_next_candle",
                        "last_candle_time": self.last_seen_candle_ts.strftime(
                            "%Y-%m-%d %H:%M:%S IST"
                        )
                    })

            except Exception as e:
                current_state_store.update({
                    "status": "error",
                    "message": str(e)
                })

            time.sleep(self.poll_seconds)

    

    # ----------------------------
    # Stop
    # ----------------------------
    def stop(self):
        self.running = False
        current_state_store.update({
            "status": "stopped"
        })

    # ----------------------------
    # Candle processing (SINGLE SHOT)
    # ----------------------------
    def _process_candle(self, candle: pd.Series):
        print("\n[DEBUG] ENTER _process_candle()")
        print("[DEBUG] candle type:", type(candle))
        print("[DEBUG] buffer BEFORE append shape:", self.buffer.shape)

        

        print("[DEBUG] buffer BEFORE predict shape:", self.buffer.shape)
        print("[DEBUG] buffer dtypes:\n", self.buffer.dtypes)

        candle_ts = candle["datetime"]

        # Safety: never predict twice for same candle
        if self.last_predicted_candle_ts == candle_ts:
            return

        print(f"[Live] Processing candle {candle_ts}")

        self.buffer = pd.concat(
            [self.buffer, candle.to_frame().T],
            ignore_index=True,
        ).tail(self.window_size)

        if len(self.buffer) < MIN_BUFFER_FOR_PREDICTION:
            current_state_store.update({
                "status": "live",
                "phase": "warming_up",
                "message": "Collecting candles before first prediction",
                "buffer_size": len(self.buffer),
                "required": MIN_BUFFER_FOR_PREDICTION,
                "last_candle_time": candle["datetime"].strftime("%Y-%m-%d %H:%M:%S")
            })
            return


        result = self.predictor.predict(self.buffer)

        state = {
            "status": "live",
            "phase": "prediction",
            "timestamp": candle_ts.strftime("%Y-%m-%d %H:%M:%S IST"),
            **result,
        }

        current_state_store.update(state)
        signal_history_store.append(state)

        self.last_predicted_candle_ts = candle_ts

        print(f"[Live] Prediction emitted: {result['signal']} ({result['confidence_level']})")
