from typing import Dict, List, Optional
from threading import Lock


class CurrentStateStore:
    """
    Holds the latest system state.
    Overwritten on every replay step.
    """

    def __init__(self):
        self._state: Optional[Dict] = None
        self._lock = Lock()

    def update(self, state: Dict):
        with self._lock:
            self._state = state

    def get(self) -> Optional[Dict]:
        with self._lock:
            return self._state
        

    def has_prediction(self) -> bool:
        """
        Returns True if the current state represents
        a prediction emitted by the model.
        """
        
        with self._lock:
            print("Are we coming here")
            if not self._state:
                return False

            return (
                "signal" in self._state
                or "prediction" in self._state
                or "probabilities" in self._state
            )


class SignalHistoryStore:
    """
    Stores recent signal history (optional).
    """

    def __init__(self, max_length: int = 500):
        self._history: List[Dict] = []
        self._max_length = max_length
        self._lock = Lock()

    def append(self, state: Dict):
        with self._lock:
            self._history.append(state)
            if len(self._history) > self._max_length:
                self._history.pop(0)

    def get_all(self) -> List[Dict]:
        with self._lock:
            return list(self._history)
