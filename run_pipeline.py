import sys
import os
import pandas as pd

# Ajouter src au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.train import train
from models.predict import Predictor
from utils.helpers import save_object, load_object

print("Loading raw data...")
df = pd.read_csv("data/raw/ai4i2020.csv")

print("Preprocessing data...")
features = ["Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Type"]
target = "Machine failure"

# Encodage du type produit
df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

X = df[features]
y = df[target]

print("Training model...")
model, scaler = train(X, y)
print("Training complete. Model saved.")

print("Testing prediction...")
predictor = Predictor()
sample_input = X.iloc[0].to_dict()
result = predictor.predict(sample_input)
print(result)
