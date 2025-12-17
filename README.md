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