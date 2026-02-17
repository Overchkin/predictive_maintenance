import pandas as pd
from src.data.preprocess import preprocess_data


def test_preprocess_runs():
    df = pd.read_csv("data/raw/ai4i2020.csv")
    X, y, scaler, features = preprocess_data(df)

    assert X.shape[0] == len(y)
