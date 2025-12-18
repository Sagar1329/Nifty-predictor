import pandas as pd
from ml.inference.predictor import TrendPredictor

# Load recent candles (example: last ~60 rows)
df = pd.read_csv("ml/data/nifty_5m_clean.csv")

recent = df.tail(60)

predictor = TrendPredictor()
result = predictor.predict(recent)

print(result)
