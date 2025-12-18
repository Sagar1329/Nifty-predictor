import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

X_FILE = "ml/training/X.csv"
Y_FILE = "ml/training/y.csv"
MODEL_DIR = "ml/models"

TEST_RATIO = 0.2

TARGETS = [
    "y_trend_5m",
    "y_trend_10m",
    "y_trend_15m"
]

def train_trend_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = pd.read_csv(X_FILE)
    y = pd.read_csv(Y_FILE)

    split_idx = int(len(X) * (1 - TEST_RATIO))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    for target in TARGETS:
        print("\n==============================")
        print(f"Training model for {target}")
        print("==============================")

        y_target = y[target]
        y_train, y_test = y_target.iloc[:split_idx], y_target.iloc[split_idx:]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                multi_class="auto"
            ))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        model_path = f"{MODEL_DIR}/{target}_logreg_scaled.pkl"
        joblib.dump(pipeline, model_path)
        print("Model saved to:", model_path)

if __name__ == "__main__":
    train_trend_models()
