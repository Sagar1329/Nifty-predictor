import pandas as pd

FEATURES_FILE = "ml/features/features.csv"
LABELS_FILE = "ml/training/labels.csv"

OUTPUT_X = "ml/training/X.csv"
OUTPUT_Y = "ml/training/y.csv"

LOOKBACK = 10

def build_dataset():
    X = pd.read_csv(FEATURES_FILE)
    y = pd.read_csv(LABELS_FILE)

    # Drop first LOOKBACK labels to align with features
    y = y.iloc[LOOKBACK:].reset_index(drop=True)

    # Trim to same length
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len].reset_index(drop=True)
    y = y.iloc[:min_len].reset_index(drop=True)

    # Final sanity check
    assert len(X) == len(y), "Feature-label length mismatch"

    X.to_csv(OUTPUT_X, index=False)
    y.to_csv(OUTPUT_Y, index=False)

    print("Final dataset built:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    build_dataset()
