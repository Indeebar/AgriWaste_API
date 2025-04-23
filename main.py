from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
import os
import uvicorn

# Optional: Disable GPU usage to avoid CUDA warnings on Render
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load trained model and pre-processing tools
model = tf.keras.models.load_model(
    "price_model.h5",
    custom_objects={"MeanSquaredError": tf.keras.losses.MeanSquaredError, "MeanAbsoluteError": tf.keras.metrics.MeanAbsoluteError}
)
scaler = joblib.load("scaler.pkl")
waste_encoder = joblib.load("waste_encoder.pkl")
demand_encoder = joblib.load("demand_encoder.pkl")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (so frontend can access the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this if you want to allow only specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data model
class InputData(BaseModel):
    waste_type: str
    demand: str
    weight: float

@app.get("/")
def root():
    return {"message": "🌾 AgriWaste Price Prediction API is up and running!"}

@app.post("/predict")
def predict_price(data: InputData):
    # Encode inputs safely
    try:
        waste_encoded = waste_encoder.transform([data.waste_type])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid agricultural waste type")

    try:
        demand_encoded = demand_encoder.transform([data.demand])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid demand level (use High, Medium, or Low)")

    # Scale the quantity
    try:
        quantity_scaled = scaler.transform([[data.weight]])[0][0]
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to scale the quantity. Ensure it is a valid number.")

    # Arrange features in correct order: [quantity_scaled, demand_encoded, waste_encoded]
    input_features = np.array([[quantity_scaled, demand_encoded, waste_encoded]])

    # Predict price per kg
    predicted_price_per_kg = model.predict(input_features)[0][0]
    total_price = predicted_price_per_kg * data.weight

    # Return predictions
    return {
        "predicted_price_per_kg": round(float(predicted_price_per_kg), 2),
        "total_price": round(float(total_price), 2),
        "unit": "INR"
    }

# Entry point for Uvicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets the PORT environment variable
    uvicorn.run("main:app", host="0.0.0.0", port=port)
