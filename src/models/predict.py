import pandas as pd
import numpy as np

from src.utils.helpers import load_object
from src.features.create_features import engineer_features


MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"


class Predictor:

    def __init__(self):
        self.model = load_object(MODEL_PATH)
        self.scaler = load_object(SCALER_PATH)

    def preprocess_input(self, input_dict: dict) -> pd.DataFrame:
        """
        Convert raw input into model-ready format.
        """

        df = pd.DataFrame([input_dict])

        # Encode Type
        df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

        # Feature engineering
        df = engineer_features(df)

        return df

    def predict(self, input_dict: dict) -> dict:
        """
        Predict machine failure probability.
        """

        df_processed = self.preprocess_input(input_dict)

        X_scaled = self.scaler.transform(df_processed)

        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "failure_probability": float(probability)
        }
