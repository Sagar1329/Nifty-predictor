import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

X_FILE = "ml/training/X.csv"
Y_FILE = "ml/training/y.csv"
MODEL_FILE = "ml/models/y_trend_15m_rf.pkl"

TEST_RATIO = 0.2

def train_rf():
    os.makedirs("ml/models", exist_ok=True)

    # Load data
    X = pd.read_csv(X_FILE)
    y = pd.read_csv(Y_FILE)["y_trend_15m"]

    # Time-based split
    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_FILE)
    print("\nModel saved to:", MODEL_FILE)

if __name__ == "__main__":
    train_rf()
