import threading
from typing import Optional

from ml.replay.replay_engine import ReplayEngine
from ml.state.global_state import current_state_store, signal_history_store


class ReplayController:
    def __init__(self, csv_path: str, tick_seconds: float = 1.0):
        self.csv_path = csv_path
        self.tick_seconds = tick_seconds
        self._thread: Optional[threading.Thread] = None
        self._engine: Optional[ReplayEngine] = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"status": "already_running"}

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

            return {"status": "started"}

    def stop(self):
        with self._lock:
            if not self._engine:
                return {"status": "not_running"}

            self._engine.running = False
            return {"status": "stopped"}

    def reset(self):
        with self._lock:
            # Stop if running
            if self._engine:
                self._engine.running = False

            # Clear state
            current_state_store.update(None)
            signal_history_store.get_all().clear()

            self._engine = None
            self._thread = None

            return {"status": "reset"}
