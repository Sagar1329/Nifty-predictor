import pandas as pd
import json

df = pd.read_csv("ml/data/nifty_5m_clean.csv")

payload = df.tail(60).to_dict(orient="records")

print(json.dumps(payload, indent=2))
