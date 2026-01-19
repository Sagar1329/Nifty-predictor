import threading
from typing import Optional

from ml.live.live_polling_engine import LivePollingEngine
from ml.data.yahoo_live_provider import YahooLiveDataProvider
from ml.inference.predictor import TrendPredictor
from ml.state.runtime_mode import RuntimeMode
from ml.state.global_state import current_state_store

import ml.state.runtime_mode as runtime_mode


class LiveController:
    def __init__(self, poll_seconds: int = 60):
        self.poll_seconds = poll_seconds
        self._engine: Optional[LivePollingEngine] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()



    def start(self):
        with self._lock:
            if runtime_mode.current_mode == RuntimeMode.LIVE:
                return {
                    "status": "already_running",
                    "mode": runtime_mode.current_mode,
                }

            if runtime_mode.current_mode == RuntimeMode.REPLAY:
                return {
                    "status": "conflict",
                    "message": "Replay is running. Stop replay first.",
                    "mode": runtime_mode.current_mode,
                }

            provider = YahooLiveDataProvider()
            predictor = TrendPredictor()

            self._engine = LivePollingEngine(
                provider=provider,
                predictor=predictor,
                poll_seconds=self.poll_seconds,
            )
            current_state_store.update({
                "status": "live",
                "phase": "warming_up",
                "message": "Live mode started. Waiting for first candle.",
                "last_candle_time": None
            })

            self._thread = threading.Thread(
                target=self._engine.start,
                daemon=True
            )
            self._thread.start()

            # Set runtime mode
            runtime_mode.current_mode = RuntimeMode.LIVE

            return {
                "status": "started",
                "mode": runtime_mode.current_mode,
            }

    def stop(self):
        with self._lock:
            if runtime_mode.current_mode != RuntimeMode.LIVE:
                return {
                    "status": "not_running",
                    "mode": runtime_mode.current_mode,
                }

            self._engine.stop()
            self._engine = None
            self._thread = None

            # Reset runtime mode
            runtime_mode.current_mode = RuntimeMode.NONE

            return {
                "status": "stopped",
                "mode": runtime_mode.current_mode,
                  "message": "Live mode stopped"

            }
        
    def debug_buffer(self):
        engine = self._engine  # whatever variable holds LivePollingEngine

        if engine.buffer is None or engine.buffer.empty:
            return {
                "buffer_size": 0,
                "message": "Buffer is empty"
            }

        return {
            "buffer_size": len(engine.buffer),
            "columns": engine.buffer.columns.tolist(),
            "dtypes": engine.buffer.dtypes.astype(str).to_dict(),
            "min_datetime": str(engine.buffer["datetime"].min()),
            "max_datetime": str(engine.buffer["datetime"].max()),
            "last_5_rows": engine.buffer.tail(5).to_dict(orient="records")
        }

