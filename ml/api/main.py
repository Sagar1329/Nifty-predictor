from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

from ml.inference.predictor import TrendPredictor
from ml.state.global_state import current_state_store, signal_history_store
from ml.replay.replay_controller import ReplayController
from ml.live.live_controller import LiveController
import ml.state.runtime_mode as runtime_mode


app = FastAPI(
    title="NIFTY Trend Prediction API",
    description="15-minute intraday trend prediction using ML",
    version="1.0.0"
)

replay_controller = ReplayController(
    csv_path="ml/data/nifty_5m_clean.csv",
    tick_seconds=0.5
)

# Load model once at startup
predictor = TrendPredictor()
live_controller = LiveController(poll_seconds=60)

# ----------------------------
# Request / Response Schemas
# ----------------------------

class Candle(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float


class PredictionResponse(BaseModel):
    signal: str
    confidence_level: str
    prediction: str
    probabilities: dict


# ----------------------------
# Health Check
# ----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Prediction Endpoint
# ----------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict_trend(candles: List[Candle]):
    if len(candles) < 60:
        raise HTTPException(
            status_code=400,
            detail="At least 60 candles are required for prediction"
        )

    df = pd.DataFrame([c.dict() for c in candles])

    try:
        result = predictor.predict(df)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ----------------------------
# State Endpoints
# ----------------------------

@app.get("/state")
def get_current_state():
    state = current_state_store.get()
    if state is None:
        return {
            "status": "initializing",
            "message": "No state available yet"
        }
    return state


@app.get("/history")
def get_signal_history(limit: int = 100):
    history = signal_history_store.get_all()
    return history[-limit:]


# ----------------------------
# Replay Controls
# ----------------------------

@app.post("/replay/start")
def start_replay():
    return replay_controller.start()


@app.post("/replay/stop")
def stop_replay():
    return replay_controller.stop()


@app.post("/replay/reset")
def reset_replay():
    return replay_controller.reset()


# ----------------------------
# Live Controls
# ----------------------------

@app.post("/live/start")
def start_live():
    return live_controller.start()


@app.post("/live/stop")
def stop_live():
    return live_controller.stop()

@app.get("/debug/buffer")
def stop_live():
    return live_controller.debug_buffer()
# ----------------------------
# Runtime Mode
# ----------------------------

@app.get("/mode")
def get_mode():
    return {
        "mode": runtime_mode.current_mode
    }
