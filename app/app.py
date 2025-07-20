# app/app.py

import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Customer Churn Predictor")

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
