import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.features.create_features import engineer_features


def preprocess_data(df: pd.DataFrame):
    """
    Clean and prepare dataset for modeling.
    """

    df = df.drop(columns=["UDI", "Product ID"])

    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    df = engineer_features(df)

    df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns
