# Step-by-Step Retraining Guide (From Scratch)
## Step 1 — Prepare Raw Candle Data

### Input

- 1-minute historical data (CSV)

- Columns required:

  - `date, open, high, low, close, volume`

### Script
```py
python ml/data/resample_1m_to_5m.py;
```

### What it does

- Converts 1-minute candles → 5-minute candles

- Filters NSE trading hours

- Produces clean OHLCV candles

### Output

```
ml/data/nifty_5m_clean.csv
```

## Step 2 — Feature Engineering

### Script

```
python ml/features/build_features.py
```

### What it does

- Computes technical indicators:

    - log returns

    - candle body

    - high-low range

    - EMA(9), EMA(21)

    - RSI(14)

    - volatility(10)

- Builds rolling lookback windows (LOOKBACK = 10)

### Output
```
ml/features/features.csv
```

### Expected shape
```
(rows ≈ number_of_candles - warmup, columns = 70)
```

## Step 3 — Label Generation

### Script

```
python ml/training/create_labels.py
```

### What it does

- Generates supervised labels:

    - y_dir → next candle direction

    - y_trend_5m

    - y_trend_10m

    - y_trend_15m

- Uses ±0.05% threshold

### Output

```
ml/training/labels.csv
```

## Step 4 — Dataset Alignment

### Script
```
python ml/training/build_dataset.py
```

### Why

- Features use past candles

- Labels use future candles

- First LOOKBACK rows must be dropped

### What it does

- Aligns features and labels

- Preserves time order (no shuffling)

### Outputs

```
ml/training/X.csv
ml/training/y.csv
```

## Step 5 — Train Direction Model (Baseline)

### Script

```
python ml/training/train_direction.py
```

### Model

- Logistic Regression

- Target: y_dir

- Time-based split (last 20% test)

### Output

```
ml/models/next_candle_logreg.pkl
```

## Step 6 — Train Scaled Direction Model

### Script
```
python ml/training/train_direction_scaled.py
```

### Enhancement

- StandardScaler + Logistic Regression

### Output
```
ml/models/next_candle_logreg_scaled.pkl
```
## Step 7 — Train Trend Models (Logistic Regression)

### Script
```
python ml/training/train_trend_models.py
```

### Targets

- y_trend_5m

- y_trend_10m

- y_trend_15m

### Outputs
```
ml/models/y_trend_5m_logreg_scaled.pkl
ml/models/y_trend_10m_logreg_scaled.pkl
ml/models/y_trend_15m_logreg_scaled.pkl
```
## Step 8 — Train Random Forest (Primary Model)

### Script
```
python ml/training/train_rf_trend_15m.py
```

### Model

- Random Forest

- Target: y_trend_15m

- Class-balanced

- Depth-controlled

### Output
```
ml/models/y_trend_15m_rf.pkl
```

**This is the model used by live inference.**

## Step 9 — Inference Validation

**Used by**

- Replay engine

- Live polling engine

### Script (implicit)
```
TrendPredictor.predict(candles)
```

### Output
```
{
  "signal": "UP | DOWN | UNCERTAIN",
  "confidence_level": "HIGH | MEDIUM | LOW",
  "prediction": "UP | DOWN | SIDEWAYS",
  "probabilities": {...}
}
```