from ml.replay.replay_engine import ReplayEngine
from ml.state.state_store import CurrentStateStore, SignalHistoryStore

current_state_store = CurrentStateStore()
signal_history_store = SignalHistoryStore()

engine = ReplayEngine(
    csv_path="ml/data/nifty_5m_clean.csv",
    current_state_store=current_state_store,
    signal_history_store=signal_history_store,
    tick_seconds=0.5
)

engine.start()
