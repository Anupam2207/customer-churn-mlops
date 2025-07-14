import mlflow

# Set tracking URI (local folder-based)
mlflow.set_tracking_uri("mlruns/")
mlflow.set_experiment("Churn_Prediction_Experiment")

with mlflow.start_run():
    mlflow.log_param("example_param", 123)
    mlflow.log_metric("example_metric", 0.95)
    print("Test run logged to MLflow.")
