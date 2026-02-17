# run_pipeline.py
from src.data.ingest import load_raw_data
from src.data.preprocess import preprocess_data
from src.models.train import train
from src.models.predict import Predictor
import pandas as pd

# 1️⃣ Load raw data
print("Loading raw data...")
df = load_raw_data("data/raw/ai4i2020.csv")

# 2️⃣ Preprocess & feature engineering
print("Preprocessing data...")
X_scaled, y, scaler, cols = preprocess_data(df)

# 3️⃣ Train model
print("Training model...")
train()

# 4️⃣ Quick prediction test
print("Testing prediction...")
predictor = Predictor()
sample_input = {
    "Type": "L",
    "Air temperature [K]": 300,
    "Process temperature [K]": 310,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40,
    "Tool wear [min]": 5,
}
print(predictor.predict(sample_input))

