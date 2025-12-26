import time
import pandas as pd
from datetime import datetime, timedelta
import pytz

from ml.data.yahoo_live_provider import YahooLiveDataProvider
from ml.inference.predictor import TrendPredictor
from ml.state.global_state import current_state_store, signal_history_store

IST = pytz.timezone("Asia/Kolkata")

CANDLE_INTERVAL = timedelta(minutes=5)
WAITING_TOLERANCE = timedelta(minutes=8)  # 1.5x candle duration


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
        self.prediction_emitted_for_timestamp = None


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
        self.last_seen_timestamp = self.buffer["datetime"].iloc[-1]

    # ----------------------------
    # Main loop
    # # ----------------------------
    # def start(self):
    #     self.running = True
    #     self._warm_start()

    #     while self.running:
    #         try:
    #             df = self.provider.fetch_recent_candles()

    #             if df is None or df.empty:
    #                 current_state_store.update({
    #                     "status": "live",
    #                     "phase": "no_data",
    #                     "message": "No data returned from Yahoo.",
    #                     "last_candle_time": None
    #                 })
    #                 time.sleep(self.poll_seconds)
    #                 continue

    #             # Normalize to IST
    #             df["datetime"] = (
    #                 pd.to_datetime(df["datetime"], utc=True)
    #                 .dt.tz_convert(IST)
    #             )

    #             df = df.sort_values("datetime")
    #             latest_ts = df.iloc[-1]["datetime"]
    #             now_ist = datetime.now(IST)

    #             # ----------------------------
    #             # Market open — waiting for next candle
    #             # ----------------------------
    #             if now_ist - latest_ts <= WAITING_TOLERANCE:
    #                 if not current_state_store.has_prediction():
    

    #                     current_state_store.update({
    #                         "status": "live",
    #                         "phase": "waiting_for_next_candle",
    #                         "message": "Market is open. Waiting for next candle close.",
    #                         "last_candle_time": latest_ts.strftime("%Y-%m-%d %H:%M:%S %Z")
    #                     })
    #                 time.sleep(self.poll_seconds)
    #                 continue

    #             # ----------------------------
    #             # Market likely closed
    #             # ----------------------------
    #             if now_ist - latest_ts > WAITING_TOLERANCE:
    #                 current_state_store.update({
    #                     "status": "live",
    #                     "phase": "market_closed",
    #                     "message": "Market appears closed. Waiting for new candles.",
    #                     "last_candle_time": latest_ts.strftime("%Y-%m-%d %H:%M:%S %Z")
    #                 })
    #                 time.sleep(self.poll_seconds)
    #                 continue

    #             # ----------------------------
    #             # New candle(s) detected
    #             # ----------------------------
    #             new_candles = df[
    #                 df["datetime"] > (self.last_seen_timestamp or df.iloc[0]["datetime"])
    #             ]

    #             for _, candle in new_candles.iterrows():
    #                 self._process_candle(candle)
    #                 self.last_seen_timestamp = candle["datetime"]

    #         except Exception as e:
    #             current_state_store.update({
    #                 "status": "live",
    #                 "phase": "engine_error",
    #                 "message": f"Live engine error: {e}",
    #                 "last_candle_time": str(self.last_seen_timestamp)
    #             })

    #         time.sleep(self.poll_seconds)


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

                # ----------------------------
                # Normalize datetime to IST
                # ----------------------------
                if "datetime" in df.columns:
                    df["datetime"] = (
                        pd.to_datetime(df["datetime"], utc=True)
                        .dt.tz_convert(IST)
                    )

                df = df.sort_values("datetime")
                latest_ts = df.iloc[-1]["datetime"]

                # ----------------------------
                #  Detect & process NEW candles FIRST
                # ----------------------------
                if self.last_seen_timestamp is None:
                    self.last_seen_timestamp = latest_ts

                new_candles = df[df["datetime"] > self.last_seen_timestamp]

                if not new_candles.empty:
                    for _, candle in new_candles.iterrows():
                        self._process_candle(candle)
                        self.last_seen_timestamp = candle["datetime"]

                    # After processing new candles, go to next poll
                    time.sleep(self.poll_seconds)
                    continue

                # ----------------------------
                # 2No new candle → update STATUS
                # ----------------------------
                now_ist = datetime.now(IST)

                market_open = (
                    (now_ist.hour > 9 or (now_ist.hour == 9 and now_ist.minute >= 15))
                    and
                    (now_ist.hour < 15 or (now_ist.hour == 15 and now_ist.minute <= 30))
                )

                if market_open:
                    # Do NOT overwrite an existing prediction
                    if self.prediction_emitted_for_timestamp is None:
                        current_state_store.update({
                            "status": "live",
                            "phase": "waiting_for_next_candle",
                            "message": "Market is open. Waiting for next candle close.",
                            "last_candle_time": latest_ts.strftime("%Y-%m-%d %H:%M:%S %Z")
                        })
                else:
                    current_state_store.update({
                        "status": "live",
                        "phase": "market_closed",
                        "message": "Market is closed. Waiting for next session.",
                        "last_candle_time": latest_ts.strftime("%Y-%m-%d %H:%M:%S %Z")
                    })

            except Exception as e:
                current_state_store.update({
                    "status": "live",
                    "phase": "engine_error",
                    "message": f"Live engine error: {e}",
                    "last_candle_time": (
                        self.last_seen_timestamp.strftime("%Y-%m-%d %H:%M:%S %Z")
                        if self.last_seen_timestamp else None
                    )
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
    # def _process_candle(self, candle: pd.Series):
    #     print(f"Processing candle: {candle['datetime']}")
    #     self.buffer = pd.concat(
    #         [self.buffer, candle.to_frame().T],
    #         ignore_index=True,
    #     ).tail(self.window_size)

    #     if len(self.buffer) < self.window_size:
    #         return

    #     result = self.predictor.predict(self.buffer)

    #     state = {
    #         "timestamp": candle["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
    #         **result,
    #     }

    #     current_state_store.update(state)
    #     signal_history_store.append(state)
    #     self.prediction_emitted_for_timestamp = candle["datetime"]

    def _process_candle(self, candle: pd.Series):
        print(f"\n[LivePollingEngine] Processing candle:")
        print(f"  Candle time : {candle['datetime']}")

        self.buffer = pd.concat(
            [self.buffer, candle.to_frame().T],
            ignore_index=True,
        ).tail(self.window_size)

        if len(self.buffer) < self.window_size:
            print("  Buffer not ready yet, skipping prediction")
            return

        result = self.predictor.predict(self.buffer)

        print("  ✅ Prediction generated:")
        print(f"    Signal           : {result.get('signal')}")
        print(f"    Confidence level : {result.get('confidence_level')}")
        print(f"    Probabilities    : {result.get('probabilities')}")

        state = {
            "timestamp": candle["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            **result,
        }

        # TEMP: keep state update for now
        current_state_store.update(state)
        signal_history_store.append(state)

        self.prediction_emitted_for_timestamp = candle["datetime"]

        print("  🟢 Prediction stored in state store\n")

