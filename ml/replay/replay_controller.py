import threading
from typing import Optional

from ml.replay.replay_engine import ReplayEngine
from ml.state.global_state import current_state_store, signal_history_store
from ml.state.runtime_mode import RuntimeMode, current_mode


class ReplayController:
    def __init__(self, csv_path: str, tick_seconds: float = 1.0):
        self.csv_path = csv_path
        self.tick_seconds = tick_seconds
        self._thread: Optional[threading.Thread] = None
        self._engine: Optional[ReplayEngine] = None
        self._lock = threading.Lock()

    def start(self):
        global current_mode

        with self._lock:
            # ❌ Block replay if live mode is running
            if current_mode == RuntimeMode.LIVE:
                return {
                    "status": "conflict",
                    "message": "Live mode is running. Stop live first.",
                    "mode": current_mode,
                }

            # Already running replay
            if self._thread and self._thread.is_alive():
                return {
                    "status": "already_running",
                    "mode": current_mode,
                }

            self._engine = ReplayEngine(
                csv_path=self.csv_path,
                current_state_store=current_state_store,
                signal_history_store=signal_history_store,
                tick_seconds=self.tick_seconds,
            )

            self._thread = threading.Thread(
                target=self._engine.start,
                daemon=True
            )
            self._thread.start()

            # ✅ Set runtime mode
            current_mode = RuntimeMode.REPLAY

            return {
                "status": "started",
                "mode": current_mode,
            }

    def stop(self):
        global current_mode

        with self._lock:
            if not self._engine:
                return {
                    "status": "not_running",
                    "mode": current_mode,
                }

            self._engine.running = False
            self._engine = None
            self._thread = None

            # ✅ Reset runtime mode
            current_mode = RuntimeMode.NONE

            return {
                "status": "stopped",
                "mode": current_mode,
            }

    def reset(self):
        global current_mode

        with self._lock:
            # Stop if running
            if self._engine:
                self._engine.running = False

            # Clear state stores
            current_state_store.update(None)
            signal_history_store.get_all().clear()

            self._engine = None
            self._thread = None

            # ✅ Reset runtime mode
            current_mode = RuntimeMode.NONE

            return {
                "status": "reset",
                "mode": current_mode,
            }
