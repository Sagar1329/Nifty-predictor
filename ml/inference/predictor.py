import pandas as pd
import numpy as np
import joblib
from pathlib import Path

LOOKBACK = 10

FEATURE_NAMES = [
    "log_return",
    "candle_body",
    "hl_range",
    "ema_9",
    "ema_21",
    "rsi_14",
    "volatility_10"
]

MODEL_PATH = Path("ml/models/y_trend_15m_rf.pkl")


class TrendPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)

    # ----------------------------
    # Indicator helpers
    # ----------------------------
    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    

    @staticmethod
    def tag_volatility(vol: float) -> str:
        if vol < 0.0008:
            return "LOW_VOL"
        elif vol < 0.0015:
            return "MEDIUM_VOL"
        else:
            return "HIGH_VOL"

    @staticmethod
    def tag_rsi(rsi: float) -> str:
        if rsi < 30:
            return "OVERSOLD"
        elif rsi > 70:
            return "OVERBOUGHT"
        else:
            return "NEUTRAL"

    @staticmethod
    def tag_time_of_day(ts) -> str:
        hour = ts.hour
        minute = ts.minute

        if hour == 9 and minute < 45:
            return "OPEN"
        elif hour < 14 or (hour == 14 and minute < 30):
            return "MID"
        else:
            return "CLOSE"


    # ----------------------------
    # Feature builder (MUST MATCH TRAINING)
    # ----------------------------
    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) < LOOKBACK + 20:
            raise ValueError("Not enough candles to build features")

        df = df.copy()

        # Base features
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["candle_body"] = df["close"] - df["open"]
        df["hl_range"] = df["high"] - df["low"]

        # Indicators
        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["rsi_14"] = self._compute_rsi(df["close"], 14)
        df["volatility_10"] = df["log_return"].rolling(10).std()

        df = df.dropna()

        # Last LOOKBACK candles
        window = df.iloc[-LOOKBACK:][FEATURE_NAMES]

        # Flatten (same shape as training)
        return window.values.flatten().reshape(1, -1)

    # ----------------------------
    # Public inference API
    # ----------------------------
    
    def predict(self, candles: pd.DataFrame) -> dict:
        # ----------------------------
        # Build features (for model + regimes)
        # ----------------------------
        df = candles.copy()

        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["candle_body"] = df["close"] - df["open"]
        df["hl_range"] = df["high"] - df["low"]

        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["rsi_14"] = self._compute_rsi(df["close"], 14)
        df["volatility_10"] = df["log_return"].rolling(10).std()

        df = df.dropna()

        if len(df) < LOOKBACK:
            raise ValueError("Not enough data after indicators")

        # ----------------------------
        # Model inference
        # ----------------------------
        window = df.iloc[-LOOKBACK:][FEATURE_NAMES]
        X = window.values.flatten().reshape(1, -1)

        pred_class = int(self.model.predict(X)[0])
        probs = self.model.predict_proba(X)[0]

        label_map = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}

        probs_dict = {
            "DOWN": float(probs[0]),
            "SIDEWAYS": float(probs[1]),
            "UP": float(probs[2]),
        }

        # ----------------------------
        # Confidence logic
        # ----------------------------
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        top_label, top_prob = sorted_probs[0]
        second_label, second_prob = sorted_probs[1]

        margin = top_prob - second_prob

        MIN_TOP_PROB = 0.45
        MIN_MARGIN = 0.15

        if top_prob < MIN_TOP_PROB or margin < MIN_MARGIN:
            signal = "UNCERTAIN"
        else:
            signal = top_label

        if margin >= 0.35:
            confidence_level = "HIGH"
        elif margin >= 0.20:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # ----------------------------
        # Regime-aware abstain (v1)
        # ----------------------------
        latest = df.iloc[-1]

        vol_regime = self.tag_volatility(latest["volatility_10"])
        rsi_regime = self.tag_rsi(latest["rsi_14"])
        time_regime = self.tag_time_of_day(latest["datetime"])

        if (
            vol_regime == "LOW_VOL"
            and time_regime == "MID"
            and rsi_regime == "NEUTRAL"
        ):
            signal = "UNCERTAIN"
            confidence_level = "LOW"
        
     





        # ----------------------------
        # Final response
        # ----------------------------
        return {
            "signal": signal,
            "confidence_level": confidence_level,
            "prediction": label_map[pred_class],
            "probabilities": probs_dict,
        }
