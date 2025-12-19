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
