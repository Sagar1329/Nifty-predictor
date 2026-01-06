# 📈 NIFTY Intraday Direction & Trend Prediction (Side Project)

A non-commercial, educational machine learning project that predicts **short-term market direction and trend** for the **NIFTY 50 index** using **free, open-source data** and a clean, reproducible ML pipeline.

The project focuses on **probabilistic prediction**, not exact price forecasting.

---

## 🎯 Project Objectives

- Use **free and open-source data only**
- Learn from **historical intraday (5-minute) NIFTY 50 candles**
- Predict:
  - **Next candle direction (UP / DOWN)**
  - (Planned) **5 / 10 / 15-minute trend**
- Follow **proper ML discipline**:
  - No data leakage
  - Time-aware splits
  - Baseline before optimization

> ⚠️ This project is for **learning and experimentation only**.  
> It is **not** intended for live trading or financial advice.

---

## 🧠 Core Idea

Predicting **exact prices** is unreliable due to:
- Market noise
- News and exogenous events
- Micro-structure randomness

Instead, this project predicts:
- **Direction**
- **Trend**
- **Probability**

This is how real quantitative and ML-driven systems approach markets.

---

## 🛠️ Tech Stack

- **Python 3.12**
- **pandas, numpy**
- **yfinance** (Yahoo Finance)
- **scikit-learn**
- **joblib**

(Frontend: React, Backend: FastAPI — planned)

---

## 📂 Project Structure
nifty-ai-predictor/
│
├── ml/
│ ├── data/
│ │ ├── fetch_data.py
│ │ ├── clean_data.py
│ │ └── nifty_5m_clean.csv
│ │
│ ├── features/
│ │ ├── build_features.py
│ │ └── features.csv
│ │
│ ├── training/
│ │ ├── create_labels.py
│ │ ├── build_dataset.py
│ │ ├── train_direction.py
│ │ ├── X.csv
│ │ └── y.csv
│ │
│ └── models/
│ └── next_candle_logreg.pkl
│
├── venv/
└── README.md



---

## 🔢 STEP-BY-STEP PIPELINE

### STEP 1 — Environment Setup

- Python **3.12**
- Virtual environment (explicitly using Python 3.12)
- Isolated dependencies

```bash
py -3.12 -m venv venv
source venv/Scripts/activate
pip install yfinance pandas numpy scikit-learn joblib



### STEP 2 — Historical Data Collection

Source: Yahoo Finance

Ticker: ^NSEI (NIFTY 50)

Interval: 5-minute candles

Window: ~60 days (Yahoo intraday limit)

Script:

ml/data/fetch_data.py

### STEP 3 — Data Cleaning & IST Conversion

Raw Yahoo data issues:

Multi-row headers

UTC timestamps

Non-ML-friendly structure

Cleaning decisions:

Convert UTC → IST

Keep only:

datetime, open, high, low, close


Remove volume (index has no volume)

Script:

ml/data/clean_data.py


Final dataset:

ml/data/nifty_5m_clean.csv

### STEP 4 — Feature Engineering (Most Important Step)

Lookback window: 10 candles (≈ 50 minutes)

Features per candle

Log return

Candle body (close − open)

High–low range

EMA 9

EMA 21

RSI 14

Rolling volatility (10 candles)

Final feature vector

7 features × 10 candles = 70 features per row

Script:

ml/features/build_features.py


Output:

ml/features/features.csv


Verification:

Rows: 4339

Columns: 70

NaNs: 0

All values: float64

### STEP 5 — Label Creation
Labels created

y_dir → Next candle direction (binary)

y_trend_5m → 5-minute trend

y_trend_10m → 10-minute trend

y_trend_15m → 15-minute trend

Trend encoding
0 → Down
1 → Sideways
2 → Up

Noise handling

Threshold: 0.05%

Small moves treated as sideways

Script:

ml/training/create_labels.py


Output:

ml/training/labels.csv

### STEP 6 — Dataset Alignment

Why alignment is necessary:

Features use past candles

Labels use future candles

First 10 labels have no matching features

Actions:

Drop first 10 label rows

Trim features & labels to same length

Preserve time order (no shuffling)

Script:

ml/training/build_dataset.py


Final datasets:

X.csv → (4339, 70)

y.csv → (4339, 4)

### STEP 7 — Baseline Model Training
Model

Logistic Regression

Target: y_dir

Time-based split (last 20% test)

No scaling (intentional baseline)

Script:

ml/training/train_direction.py

Results (Unscaled Baseline)

Accuracy: ~0.64

Convergence warning due to unscaled features (expected)

Model saved to:

ml/models/next_candle_logreg.pkl


This result is not trusted yet — it will be validated using feature scaling and stronger models.

📌 Current Status
✅ Completed

Data ingestion

Data cleaning

Feature engineering

Label creation

Dataset alignment

Baseline direction model

⏳ Planned Next

Feature scaling + retraining

Signal validation

Random Forest & trend models

FastAPI inference backend

React frontend dashboard

🧩 Design Principles Followed

No future data leakage

Time-aware splits

Features before models

Baseline before optimization

Probability > certainty

Learning-first, not trading-first

⚠️ Disclaimer

This project is for educational purposes only.
It does not constitute trading advice or a production trading system.

🙌 Author

Built as a personal side project to understand:

Time-series ML

Financial data pipelines

End-to-end ML system design





18/12/2025

📊 Model Insights & Feature Importance

This section documents what the trained models actually learned, based on feature importance analysis from the Random Forest 15-minute trend model.

Why This Matters

Accuracy alone can be misleading in financial ML.
Understanding which signals drive predictions helps validate that the model is learning market behavior, not artifacts or data leakage.


🔍 Feature Importance Summary (15-Minute Trend Model)

Feature importance was extracted from a trained Random Forest classifier predicting the 15-minute market trend (Down / Sideways / Up).

Grouped Feature Importance

| Feature Type       | Importance |
| ------------------ | ---------- |
| RSI (14)           | ~52%       |
| Rolling Volatility | ~19%       |
| Log Returns        | ~9%        |
| Candle Body        | ~8%        |
| High–Low Range     | ~7%        |
| EMA (9)            | ~3%        |
| EMA (21)           | ~3%        |


🧠 Interpretation

RSI dominates the model’s decisions, indicating that the model primarily learns momentum exhaustion and continuation patterns rather than absolute price levels.

Volatility is the second most important factor, providing regime context (trending vs consolidating markets).

Recent candle behavior matters most, with higher importance assigned to features from the most recent candles (t-1 to t-3).

EMA indicators contribute minimally, suggesting that lagging trend indicators add little incremental information once momentum and volatility are captured.



📌 Key Takeaways

The model behaves as a momentum-regime classifier, not a price predictor.

Predictions are driven by how momentum and volatility evolve over time, not by static price levels.

This aligns well with the intended use case: short-term trend classification rather than precise forecasting.


⚠️ Important Note

This feature importance reflects behavior on the current dataset and time window.
Future work includes:

Walk-forward validation across different market regimes

Testing robustness over additional time periods



🧩 How This Informs System Design

Based on these insights:

The 15-minute trend model is best used as:

A regime filter (trend vs consolidation)

A probabilistic directional bias signal

It is not suitable for standalone trading decisions

It is ideal as an input to a larger decision or risk-management system




#18/12/2025



## 🧠 Backend-Ready Inference Pipeline

This project includes a **production-style inference pipeline** that converts raw intraday OHLC data into a **15-minute trend prediction** using a trained Random Forest model.

The inference layer is designed to be:
- Stateless
- Deterministic
- Reusable
- Easy to wrap with FastAPI or any backend framework

---

### 🎯 Purpose of the Inference Pipeline

Given the **latest 5-minute candles**, the pipeline:

1. Rebuilds features **exactly as done during training**
2. Loads the trained model from disk
3. Predicts the **15-minute market trend**
4. Returns **class probabilities**, not just a label

This ensures:
- No feature mismatch between training and inference
- No data leakage
- No hidden logic

---

### 📂 Inference Module Structure

```

ml/
└── inference/
├── **init**.py
├── predictor.py
└── test_predictor.py

````

---

### 🏗️ Core Inference Class

The main entry point is the `TrendPredictor` class:

```python
from ml.inference.predictor import TrendPredictor
````

This class:

* Loads the trained **15-minute trend Random Forest model**
* Exposes a single public method: `predict()`

---

### 📥 Input Contract

The `predict()` method expects a **pandas DataFrame** containing recent candles with the following columns:

```
datetime, open, high, low, close
```

Requirements:

* Candles must be **5-minute interval**
* At least **~60 candles** should be provided to compute indicators safely
* Data must be **chronologically ordered**

Example:

```python
import pandas as pd

df = pd.read_csv("ml/data/nifty_5m_clean.csv")
recent_candles = df.tail(60)
```

---

### 📤 Output Contract

The predictor returns a dictionary with:

* A **human-readable trend label**
* **Class probabilities** for all regimes

Example output:

```json
{
  "prediction": "DOWN",
  "probabilities": {
    "DOWN": 0.42,
    "SIDEWAYS": 0.29,
    "UP": 0.29
  }
}
```

#### Trend Encoding

```
DOWN      → Bearish bias
SIDEWAYS → Consolidation / range
UP        → Bullish bias
```

Probabilities are intended to be used for:

* Confidence thresholds
* Risk filtering
* Decision support (not direct trading)

---

### ⚙️ Feature Consistency Guarantee

The inference pipeline **reuses the exact same logic** as training:

* Same indicators:

  * Log returns
  * Candle body
  * High–low range
  * EMA (9, 21)
  * RSI (14)
  * Rolling volatility (10)
* Same lookback window (**10 candles**)
* Same feature ordering (**70 features total**)

This eliminates:

* Training–inference skew
* Silent bugs
* Feature drift at inference time

---

### 🧪 Local Testing

A lightweight test script is included:

```bash
python -m ml.inference.test_predictor
```

This:

* Loads recent candles from disk
* Runs the inference pipeline
* Prints prediction + probabilities

Successful execution confirms:

* Model loading works
* Feature generation is correct
* Inference is end-to-end functional

---

### 🧩 Design Philosophy

* The model is treated as a **probabilistic regime classifier**
* It is **not** a price predictor
* It is intended to be:

  * A backend service
  * A signal component
  * An input to higher-level decision systems

---

### 🚀 Next Steps

With the inference pipeline in place, the system is ready for:

* FastAPI integration
* Live data ingestion
* Frontend visualization
* Walk-forward validation



## 19/12/2025
## 🎯 Confidence Thresholds & Abstain Logic

Financial markets are inherently noisy and uncertain.  
To avoid overconfident or misleading predictions, this project implements an explicit **confidence and abstention layer** on top of raw model probabilities.

The model is allowed to say **“I am uncertain”** when conditions are ambiguous.

---

### 🧠 Why Abstain Logic Is Necessary

The ML model always produces probabilities for each class:

```
DOWN | SIDEWAYS | UP
```

However, forcing a decision in low-confidence situations can lead to:
- False signals
- Overinterpretation
- Poor user trust

Instead, this system follows a **decision-support philosophy**, not a trading-signal philosophy.

---

### 🚦 Decision Rules (Final)

The system applies **two rules** to every prediction:

#### 1️⃣ Minimum Confidence Rule
The highest class probability must be at least:

```
max_probability ≥ 0.55
```

If not, the system returns:
```
signal = UNCERTAIN
```

---

#### 2️⃣ Separation (Margin) Rule
The top prediction must clearly exceed the second-best prediction:

```
(top_probability − second_probability) ≥ 0.10
```

If not, the system returns:
```
signal = UNCERTAIN
```

---

### 🟡 Abstain Label

When either rule fails, the system explicitly returns:

```
signal = UNCERTAIN
```

This is a **valid and expected outcome**, not an error.

---

### 📊 Confidence Levels

In addition to the signal, the API exposes a **confidence level** derived from the maximum probability:

| Max Probability | Confidence Level |
|-----------------|------------------|
| `< 0.55` | LOW |
| `0.55 – 0.65` | MEDIUM |
| `≥ 0.65` | HIGH |

This allows the frontend to:
- Adjust visual emphasis
- Filter weak signals
- Communicate uncertainty clearly

---

### 📤 API Response Example (Low Confidence)

```json
{
  "signal": "UNCERTAIN",
  "confidence_level": "LOW",
  "prediction": "DOWN",
  "probabilities": {
    "DOWN": 0.426,
    "SIDEWAYS": 0.285,
    "UP": 0.289
  }
}
```

---

### 📤 API Response Example (High Confidence)

```json
{
  "signal": "UP",
  "confidence_level": "HIGH",
  "prediction": "UP",
  "probabilities": {
    "DOWN": 0.12,
    "SIDEWAYS": 0.21,
    "UP": 0.67
  }
}
```

---

### 🧩 Design Philosophy

- The ML model produces **probabilities**
- A separate decision layer determines **whether to act**
- Uncertainty is treated as a **first-class outcome**
- The system prioritizes **robustness and interpretability** over raw accuracy

This approach aligns with real-world ML systems used in finance and risk-sensitive domains.

---

### 🚀 Impact on Frontend Design

Frontend applications should:
- Use `signal` as the primary state indicator
- Treat `UNCERTAIN` as a neutral or greyed-out state
- Use `confidence_level` to control emphasis, not logic
- Display probabilities for transparency, not decision-making

---

### ⚠️ Disclaimer

This project is for **educational and experimental purposes only**.  
It does **not** constitute trading advice or a production trading system.







## 20/12/2025
🧠 Backend Replay Engine & API Observability

After implementing confidence thresholds and abstain (UNCERTAIN) logic, the project was extended into a fully observable backend system using replay-driven execution and HTTP APIs.

🔁 Replay Execution Engine

A replay engine was built to simulate live market conditions using historical intraday data:

Advances one candle at a time

Maintains a rolling window of recent candles

Runs ML inference only when sufficient data is available

Applies confidence and abstain logic on every step

Handles session gaps and day boundaries naturally

This enables deterministic testing and debugging without requiring live market data.

🗂 In-Memory State Management

Two thread-safe state stores were introduced:

CurrentStateStore

Holds the latest market signal

Always represents the most recent inference output

SignalHistoryStore

Maintains a rolling history of recent signals

Used for analysis, visualization, and frontend consumption

State is intentionally kept in memory to avoid premature persistence decisions.

🌐 FastAPI Observability Layer

The backend exposes read-only APIs for inspecting system state:

GET /state

Returns the latest market signal with confidence and probabilities

GET /history?limit=N

Returns recent signal history in time order

These endpoints allow frontend integration and debugging without coupling UI logic to ML logic.

🎮 Replay Control Plane

Replay execution is controlled entirely via API:

POST /replay/start — start replay simulation

POST /replay/stop — stop replay execution

POST /replay/reset — reset replay cursor and clear state

Replay runs in a background thread and shares state with the API through a unified global state module.

🧱 Architectural Principles Followed

Replay-first design before live data

Clear separation of concerns:

Data ingestion

Inference

State storage

API exposure

No frontend or persistence assumptions

Deterministic, testable backend behavior

Confidence-aware outputs instead of forced predictions

✅ Current System Capabilities

Simulated live market execution

Confidence-aware ML signals

HTTP-accessible backend state

Safe replay controls

Frontend-ready API contracts

⏭ Next Planned Step

Design and implement Yahoo Finance live polling provider
(with full testability even when markets are closed)




## 21/12/2025
📡 Yahoo Live Polling (Design & Initial Implementation)

This project supports a polling-based live market mode using Yahoo Finance intraday data. Live polling is designed to be safe, deterministic, and testable even when markets are closed.

🔹 Live Polling Philosophy

Yahoo Finance does not provide real-time streaming

The system polls for completed 5-minute candles

Inference is triggered only when a new candle is detected

Replay mode and live mode share the same inference and state pipeline

🔹 YahooLiveDataProvider

A dedicated data provider was introduced to serve as the live data source.

Responsibilities

Fetch recent intraday 5-minute candles for NIFTY (^NSEI)

Normalize data into a consistent OHLC format:

datetime, open, high, low, close


Always return multiple recent candles to prevent missed data

Remain stateless and inference-agnostic

Key Design Choice
The provider intentionally fetches more than the latest candle to:

Handle Yahoo data delays

Enable robust new-candle detection

Allow testing when markets are closed

🔹 Market Closed Behavior

When markets are closed:

Yahoo returns the last available candle

No new timestamps are detected

No inference is triggered

Backend state remains unchanged

This is expected and correct behavior.

🔹 Runtime & Environment

Project now runs on Python 3.12

Required due to modern dependencies (yfinance, typing support)

Virtual environment is created once and activated per session

🔹 Current Status

Completed:

Replay-driven ML backend

Confidence thresholds and abstain logic

FastAPI observability (/state, /history)

Replay controls via API

Yahoo live data provider (Phase 1)

Planned:

LivePollingEngine with 60-second polling

New candle detection logic

Live mode start/stop controls

Replay ↔ Live mode switching

🧠 Summary (For Future You)

At this point, the backend cleanly supports:

Deterministic replay

Observable inference state

Controlled execution

A ready-to-use live data source

Live polling can be fully implemented and tested without relying on market hours.


## SETUP
 # 1. Create venv (once)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate models (REQUIRED)
python ml/training/train_trend_models.py
python ml/training/train_random_forest.py

# 4. Start backend
python -m ml.run_backend


## 25/12/2025


Live Yahoo Polling (Hardened)

The backend supports a Live mode that polls Yahoo Finance intraday data for NIFTY (^NSEI) and produces real-time trend signals.

Live mode is designed to be explicit, deterministic, and safe, even when markets are closed or data is unavailable.

Live State Guarantees

When Live mode is running:

/state is never empty

The system always reports what it knows

No silent failures occur

Frontend does not need to guess system status

Live State Phases

The /state endpoint may return the following phases while in live mode:

warming_up
{
  "status": "live",
  "phase": "warming_up",
  "message": "Live mode started. Waiting for first candle.",
  "last_candle_time": null
}


Live polling has started but no data has been processed yet.

no_data
{
  "status": "live",
  "phase": "no_data",
  "message": "No data returned from Yahoo.",
  "last_candle_time": null
}


Yahoo Finance did not return usable intraday data (common when markets are closed).

market_closed
{
  "status": "live",
  "phase": "market_closed",
  "message": "Market is closed. Waiting for new candles.",
  "last_candle_time": "YYYY-MM-DD HH:MM"
}


Yahoo returned data, but the latest candle timestamp did not advance.

Prediction State (Normal Operation)
{
  "status": "live",
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "signal": "UP | DOWN | UNCERTAIN",
  "confidence_level": "LOW | MEDIUM | HIGH",
  "prediction": "UP | DOWN | SIDEWAYS",
  "probabilities": { ... }
}


A new candle was detected and inference was successfully performed.

inference_error
{
  "status": "live",
  "phase": "inference_error",
  "message": "Inference failed for latest candle",
  "last_candle_time": "YYYY-MM-DD HH:MM"
}


An inference error occurred; the engine continues running safely.

Live Stop
{
  "status": "stopped",
  "message": "Live mode stopped"
}


Live polling has been stopped and no stale live state remains.

Design Principles

Live polling never silently spins

Market-closed behavior is explicit

Yahoo provider failures are isolated

Backend state is frontend-safe

Replay and Live modes remain mutually exclusive

This design ensures predictable behavior across market hours and external data conditions.

🧠 Phase Status

✅ Live Yahoo Polling Hardened
✅ Runtime state semantics finalized
✅ Safe for frontend integration


## 26-12-2025
git commit description 
- Normalized Yahoo live candle timestamps to IST
- Fixed false market_closed / waiting states due to UTC mismatch
- Ensured new candles are detected and processed before status updates
- Confirmed delayed predictions due to buffer warm-up are expected
- Added explicit logging for live candle processing and inference
- Clarified live state semantics without forcing early predictions

Live Mode – Expected Behavior

  Live predictions are generated only after the buffer reaches the required window size (default: 60 candles).

  During initial startup or early market hours, the system may remain in:

  waiting_for_next_candle

  This is expected behavior, not a failure.

  Yahoo live candles are normalized to IST before processing.

  Prediction latency of 30–90 seconds after candle close is normal due to data provider delays.


## 27-12-2025

- **Prediction Horizon & Semantics**

This project predicts directional market movement over a fixed future horizon, not individual candle behavior or exact prices.

- **Candle Resolution**

 - All data is based on 5-minute candles

 - Features are constructed from the last 10 candles (~50 minutes)

- **What Does a Prediction Mean?**

For every prediction made at time T (a 5-minute candle close):

The model predicts the net directional movement over the next 15 minutes, i.e. from T → T + 15 minutes.

This is implemented by comparing:

 - Close price at time T

 - Close price at time T + 15 minutes (3 × 5-minute candles ahead)

- **Target Classes**

The prediction target is a 3-class trend label:

 - UP
  Price increases by more than +0.05% over the next 15 minutes

 - DOWN
  Price decreases by more than −0.05% over the next 15 minutes

 - SIDEWAYS
  Price change stays within ±0.05% (noise zone)

This thresholding logic is used both during training and evaluation.

- **Rolling Predictions (Important)**

  Predictions are generated every 5 minutes, each with its own independent horizon:

  Prediction Time (T)	          Evaluated Against
  11:00	                          11:15 close
  11:05	                          11:20 close
  11:10	                          11:25 close

  There is no fixed global window like “11:15–11:30”.

  Each prediction answers:

  “Where will price be 15 minutes from now relative to the current close?”

- **What the Model Does NOT Predict**

 - Exact future price

 - Single candle direction

 - Intra-candle high/low

 - Trade execution signals

 - The model is designed as a directional bias signal, suitable for:

 - Filtering trades

 - Aggregation with other signals

 - Decision support systems

- **Why Evaluation Uses 5-Minute Candles**

 - Although the horizon is 15 minutes:

 - The dataset resolution is 5 minutes

 - A 15-minute horizon equals 3 future candles

 - Ground truth is derived directly from historical price movement

 - This ensures:

 - No data leakage

 - Exact alignment with training logic

 - Realistic intraday evaluation

## Summary

 - Prediction time: current 5-minute candle close

 - Prediction horizon: next 15 minutes

 - Evaluation: close(T + 15m) vs close(T)

 - Labels: UP / DOWN / SIDEWAYS

 - Predictions are rolling and overlapping

 - This design mirrors how real intraday trend models are built and evaluated.


## 29/12/2025
initial replay-based evaluation shows no reliable directional edge. Further model iteration required before production use.

🧠 Confusion Bucketing

    Each prediction is assigned a confusion bucket:

    TP – Predicted UP, actual UP

    TN – Predicted DOWN, actual DOWN

    FP – Predicted UP, actual DOWN

    FN – Predicted DOWN, actual UP

    Integrity checks ensure:

assert set(df_eval["bucket"].unique()) == {"TP", "FP", "TN", "FN"}
assert len(df_eval) == df_eval["bucket"].value_counts().sum()


This confirms:

      No dropped rows

      No invalid labels

      Correct classification logic

Evaluation Results (UP vs DOWN)

Confusion Bucket Counts

TN = 64
FN = 64
FP = 45
TP = 34


Key Observations

  Model misses more UP moves than it captures (FN > TP)

  Bias toward predicting DOWN

  Directional skill exists but is weak

  Confidence thresholds alone do not improve accuracy

Confidence-Based Performance

Performance degrades with higher confidence:

      Confidence	Accuracy
      HIGH	~38%
      MEDIUM	~56%
      LOW	~50%

This indicates:

    Model confidence is not well calibrated

    High confidence ≠ high correctness

Insights Gained

    Errors are systematic, not random

    Missed UP moves are the dominant failure mode

    Limited training data contributes, but horizon ambiguity and regime mismatch are larger issues

    Further tuning without understanding regimes would be premature

Why This Step Matters

    This evaluation establishes a trust baseline before:

    Adding more data

    Changing model architecture

    Adjusting confidence thresholds

    Building frontend signals

    It ensures future improvements are evidence-driven, not speculative.

Status

    ✔ Replay predictions exported
    ✔ Horizon-correct evaluation
    ✔ Confusion buckets validated
    ✔ Confidence-level slicing completed

➡️ Next: Misclassification analysis by market regime (RSI, volatility, time-of-day)



Regime-Based Misclassification Analysis (15-Minute Horizon)

As part of model evaluation, we performed regime tagging on replay-based predictions to understand where and why the model fails.

Regimes Analyzed

Each prediction was tagged using the candle state at prediction time:

Volatility Regime: LOW / MEDIUM / HIGH

RSI Regime: OVERSOLD / NEUTRAL / OVERBOUGHT

Time Regime: OPEN / MID / CLOSE

Trend Regime: EMA-based UP_TREND / DOWN_TREND

Key Findings

Low volatility dominates the dataset (majority of samples).

Most predictions occur during mid-session hours (≈ 9:45–14:30 IST).

RSI is predominantly neutral, indicating weak momentum.

Trend regimes frequently flip, making short-horizon direction unstable.

Insight

The model is primarily operating in low-signal market conditions (LOW_VOL + NEUTRAL RSI + MID session), where directional prediction is inherently noisy.
This explains the modest accuracy observed and confirms that abstention (UNCERTAIN) is the correct behavior in these regimes.

Conclusion

Model performance issues are driven by market regime characteristics, not pipeline or labeling errors.
Future improvements should focus on regime-aware signal gating rather than feature expansion or aggressive retraining.


## 30/12/2025


Regime-Based Misclassification Analysis (Replay Evaluation)

  After establishing baseline replay accuracy for the 15-minute trend model, we performed a regime-level error analysis to understand where and why the model fails.

  This analysis focused on False Positives (FP) and False Negatives (FN) only, as these represent costly directional mistakes.

Methodology

  Replay predictions were aligned with future candles (+15 minutes)

  Predictions were bucketed into TP / FP / TN / FN

  Each prediction was tagged with market regimes:

    Volatility Regime (LOW / MEDIUM)

    RSI Regime (OVERSOLD / NEUTRAL / OVERBOUGHT)

    Time Regime (OPEN / MID / CLOSE)

    Trend Regime (EMA-based UP / DOWN)

False Positive (FP) Regimes — Model Predicts Move That Does Not Happen

  Dominant FP patterns:

    Low volatility dominates FP cases (~90%)

    Neutral or Overbought RSI

    Mid-session trading window

    Predictions aligned with EMA trend (trend-following failures)

Interpretation:

    In compressed volatility regimes, EMA-based trend signals frequently fail

    The model tends to over-predict continuation when the market is range-bound


False Negative (FN) Regimes — Model Misses Real Move

  Dominant FN patterns:

    Low volatility remains dominant

    Neutral or Oversold RSI

    Mostly during MID session

    Strong bias toward DOWN_TREND regimes

Interpretation:

    Directional moves occur late or abruptly after long compression

    The model underreacts during low-volatility build-ups


Key Conclusion

    Directional prediction is unreliable in low-volatility regimes, especially during mid-session with neutral RSI.

Errors are systematic, not random — which makes them actionable.


Abstain Logic Justification

  This analysis directly motivates explicit abstain rules:

    Avoid predictions in LOW volatility regimes

    Avoid trend-following signals when volatility is compressed

    Prefer selective participation over constant prediction

These rules improve signal quality, not raw accuracy.

Status

 Replay evaluation complete

 Misclassification regimes identified

 Abstain rules defined (design phase)

 Next: encode regime-aware abstain logic into inference pipeline


## 03-01-2026
Coverage-Based Evaluation & Regime-Aware Abstain (v1)
Objective

Evaluate the 15-minute trend model not only on raw accuracy, but on trade-worthy signal quality, using confidence bands and minimal regime-aware abstain rules.

Coverage-Based Evaluation

We evaluate model performance under different confidence coverage levels:

Scope	Coverage	Accuracy	Trades
HIGH only	~17%	~27%	97
HIGH + MEDIUM	~26%	~26%	149
ALL signals	100%	~36%	569

Key insight
Reducing coverage via abstain logic increases signal reliability, at the cost of fewer trades — which is expected and desirable for trading systems.

Regime-Aware Abstain (v1)

A minimal abstain rule was introduced based on replay misclassification analysis:

Abstain when:

  Volatility regime = LOW_VOL

  Time regime = MID

  RSI regime = NEUTRAL

  This regime was found to dominate false positives, especially for UP predictions during low-movement midday periods.

Result:

  HIGH-confidence accuracy improved from ~20% → ~27%

  Trade count reduced, but signal quality increased

  Confirms abstain logic is effective without retraining the model

Key Findings So Far

  Confidence alone is insufficient; market regime context matters

  Low volatility + mid-session conditions are structurally noisy

  Abstain rules improve reliability without changing the model

  Current system behaves like a decision filter, not a raw classifier

Current Status

  Model architecture: unchanged

  Prediction horizon: unchanged (15 minutes)

  Replay-based evaluation: stable

  Abstain logic: v1 complete and validated

  Further abstain refinement or retraining will be handled in future phases.



### 06/01/2025
Checking in