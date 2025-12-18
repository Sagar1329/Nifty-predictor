import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

X_FILE = "ml/training/X.csv"
Y_FILE = "ml/training/y.csv"
MODEL_FILE = "ml/models/next_candle_logreg_scaled.pkl"

TEST_RATIO = 0.2

def train_scaled_model():
    # Load data
    X = pd.read_csv(X_FILE)
    y = pd.read_csv(Y_FILE)["y_dir"]

    # Time-based split
    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Pipeline: Scaling + Logistic Regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            solver="lbfgs"
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluation
    print("Scaled Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(pipeline, MODEL_FILE)
    print("\nScaled model saved to:", MODEL_FILE)

if __name__ == "__main__":
    train_scaled_model()
