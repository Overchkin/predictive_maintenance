import pandas as pd
import os
from utils.helpers import load_object

class Predictor:
    def __init__(self, model_path=None, scaler_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

        self.model = load_object(model_path)
        self.scaler = load_object(scaler_path)

        # --- Encoder les types produits au même format que le modèle ---
        self.type_mapping = {"L": 0, "M": 1, "H": 2}  # correspond au fit

    def predict(self, input_dict):
        df = pd.DataFrame([input_dict])

        # Encoder la colonne Type
        df["Type"] = df["Type"].map(self.type_mapping)

        numeric_features = ["Air temperature [K]", "Process temperature [K]",
                            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

        # scaler uniquement sur les colonnes numériques
        df_scaled = df.copy()
        df_scaled[numeric_features] = self.scaler.transform(df[numeric_features])

        # Ordre exact des colonnes pour le modèle
        final_order = ["Air temperature [K]", "Process temperature [K]",
                       "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Type"]
        df_scaled = df_scaled[final_order]

        pred = self.model.predict(df_scaled)[0]
        prob = self.model.predict_proba(df_scaled)[0][1]

        return {"prediction": int(pred), "failure_probability": float(prob)}
