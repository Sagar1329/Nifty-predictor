from ml.replay.replay_engine import ReplayEngine

engine = ReplayEngine(
    csv_path="ml/data/nifty_5m_clean.csv",
    tick_seconds=0.5
)

engine.start()
