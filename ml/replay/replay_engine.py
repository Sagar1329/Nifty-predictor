import time
import pandas as pd

from ml.inference.predictor import TrendPredictor
from ml.data.replay_provider import ReplayDataProvider
from ml.state.state_store import CurrentStateStore, SignalHistoryStore
from pathlib import Path
from ml.replay.replay_writer import ReplayPredictionWriter


class ReplayEngine:
    """
    Drives simulated time and triggers inference.
    """

    def __init__(
        self,
        csv_path: str,
        current_state_store,
        signal_history_store,
        window_size: int = 60,
        tick_seconds: float = 1.0,
    ):
        self.provider = ReplayDataProvider(csv_path)
        self.window_size = window_size
        self.tick_seconds = tick_seconds

        self.buffer = []
        self.predictor = TrendPredictor()

        self.current_state = current_state_store
        self.history = signal_history_store
        self.prediction_writer = ReplayPredictionWriter(
                                    Path("data/replay_predictions.csv")
                                )

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

                self.prediction_writer.write(state)


                print(state)

            time.sleep(self.tick_seconds)

        print("Replay finished.")
        self.running = False
