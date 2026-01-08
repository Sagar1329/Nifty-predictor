import pandas as pd

df = pd.read_csv("ml/data/nifty_50_minute_data.csv")

print(df.head())
print(df.tail())
print(df.info())
print("Rows:", len(df))

df = df.rename(columns={"date": "datetime"})
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# Drop invalid timestamps
df = df.dropna(subset=["datetime"])

# Sort
df = df.sort_values("datetime").reset_index(drop=True)

print("Datetime range:", df["datetime"].min(), "to", df["datetime"].max())

print(df["datetime"].dt.tz)

df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute

outside_hours = df[
    (df["hour"] < 9) |
    (df["hour"] > 15) |
    ((df["hour"] == 9) & (df["minute"] < 15)) |
    ((df["hour"] == 15) & (df["minute"] > 30))
]

print("Rows outside trading hours:", len(outside_hours))


df["delta"] = df["datetime"].diff()

# Any gap > 1 minute inside sessions
gaps = df[df["delta"] > pd.Timedelta(minutes=1)]

print("Large gaps found:", len(gaps))
print(gaps.head(10))


df["delta"] = df["datetime"].diff()

# Any gap > 1 minute inside sessions
gaps = df[df["delta"] > pd.Timedelta(minutes=1)]

print("Large gaps found:", len(gaps))
print(gaps.head(10))


invalid_prices = df[
    (df["open"] <= 0) |
    (df["high"] <= 0) |
    (df["low"] <= 0) |
    (df["close"] <= 0) |
    (df["high"] < df["low"])
]

print("Invalid price rows:", len(invalid_prices))
