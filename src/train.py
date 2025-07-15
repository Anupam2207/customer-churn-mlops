import pandas as pd
import sys
import mlflow
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(train_path, model_path, params):
    df = pd.read_csv(train_path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        criterion=params["criterion"],
        random_state=params["random_state"]
    )

    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    mlflow.log_params(params)
    mlflow.log_metric("train_accuracy", acc)

    # ✅ Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train_csv = sys.argv[1]
    model_path = sys.argv[2]

    params = {
        "test_size": 0.2,
        "random_state": 42,
        "max_depth": 5,
        "n_estimators": 100,
        "criterion": "gini"
    }

    mlflow.set_tracking_uri("mlruns/")
    mlflow.set_experiment("Churn_Prediction_Experiment")
    with mlflow.start_run():
        train_model(train_csv, model_path, params)
