import threading
from typing import Optional

from ml.live.live_polling_engine import LivePollingEngine
from ml.data.yahoo_live_provider import YahooLiveDataProvider
from ml.inference.predictor import TrendPredictor
from ml.state.runtime_mode import RuntimeMode
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
            }
