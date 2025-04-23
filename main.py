# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf

# Load your models
model = tf.keras.models.load_model("price_model.h5")
scaler = joblib.load("scaler.pkl")
waste_encoder = joblib.load("waste_encoder.pkl")
demand_encoder = joblib.load("demand_encoder.pkl")

app = FastAPI()

class InputData(BaseModel):
    waste_type: str
    demand: str
    weight: float

@app.post("/predict")
def predict_price(data: InputData):
    waste = waste_encoder.transform([data.waste_type])
    demand = demand_encoder.transform([data.demand])
    features = scaler.transform([[waste[0], demand[0], data.weight]])
    prediction = model.predict(features)
    return {"predicted_price": float(prediction[0][0])}
