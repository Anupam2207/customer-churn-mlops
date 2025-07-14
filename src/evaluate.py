# src/evaluate.py
import pandas as pd
import sys
import pickle
import mlflow
from sklearn.metrics import accuracy_score, f1_score

def evaluate(test_csv, model_path):
    df = pd.read_csv(test_csv)
    X_test = df.drop("Churn", axis=1)
    y_test = df["Churn"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1_score", f1)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

if __name__ == "__main__":
    test_csv = sys.argv[1]
    model_path = sys.argv[2]

    mlflow.set_tracking_uri("mlruns/")
    mlflow.set_experiment("Churn_Prediction_Experiment")

    with mlflow.start_run():
        evaluate(test_csv, model_path)
