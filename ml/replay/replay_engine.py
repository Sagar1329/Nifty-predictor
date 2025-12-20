import time
import pandas as pd

from ml.inference.predictor import TrendPredictor
from ml.data.replay_provider import ReplayDataProvider
from ml.state.state_store import CurrentStateStore, SignalHistoryStore


class ReplayEngine:
    """
    Drives simulated time and triggers inference.
    """

    def __init__(
        self,
        csv_path: str,
        window_size: int = 60,
        tick_seconds: float = 1.0,
    ):
        self.provider = ReplayDataProvider(csv_path)
        self.window_size = window_size
        self.tick_seconds = tick_seconds

        self.buffer = []
        self.predictor = TrendPredictor()

        self.current_state = CurrentStateStore()
        self.history = SignalHistoryStore()

        self.running = False

    def _buffer_ready(self) -> bool:
        return len(self.buffer) >= self.window_size

    def start(self):
        print("Starting replay engine...")
        self.running = True

        while self.running and self.provider.has_next():
            candle = self.provider.next_candle()
            self.buffer.append(candle)

            if len(self.buffer) > self.window_size:
                self.buffer.pop(0)

            if self._buffer_ready():
                df = pd.DataFrame(self.buffer)
                result = self.predictor.predict(df)

                state = {
                    "timestamp": str(candle["datetime"]),
                    **result
                }

                self.current_state.update(state)
                self.history.append(state)

                print(state)

            time.sleep(self.tick_seconds)

        print("Replay finished.")
        self.running = False
