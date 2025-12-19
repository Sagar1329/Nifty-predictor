from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

from ml.inference.predictor import TrendPredictor

app = FastAPI(
    title="NIFTY Trend Prediction API",
    description="15-minute intraday trend prediction using ML",
    version="1.0.0"
)

# Load model once at startup
predictor = TrendPredictor()

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

    # Convert request to DataFrame
    df = pd.DataFrame([c.dict() for c in candles])

    try:
        result = predictor.predict(df)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
