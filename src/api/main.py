from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# Charger modèle et scaler au démarrage
MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


app = FastAPI(title="Predictive Maintenance API")


# Structure des données d’entrée
class MachineData(BaseModel):
    Type: int
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float


@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}


@app.post("/predict")
def predict(data: MachineData):

    # Feature engineering identique à train.py
    temp_diff = data.Process_temperature_K - data.Air_temperature_K
    mech_power = data.Rotational_speed_rpm * data.Torque_Nm
    wear_ratio = data.Tool_wear_min / (data.Rotational_speed_rpm + 1)

    features = np.array([[
        data.Type,
        data.Air_temperature_K,
        data.Process_temperature_K,
        data.Rotational_speed_rpm,
        data.Torque_Nm,
        data.Tool_wear_min,
        temp_diff,
        mech_power,
        wear_ratio
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "failure_probability": float(probability)
    }
