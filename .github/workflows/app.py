# app/app.py

import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import logging

app = FastAPI(title="Customer Churn Predictor")

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Load the model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the input schema
class CustomerData(BaseModel):
    features: dict  # Accepts a dictionary of features

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Churn Prediction API!"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        df = pd.DataFrame([data.features])
        prediction = model.predict(df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}

Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
def log_routes():
    for route in app.routes:
        logging.warning(f"ROUTE: {route.path}")
