import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for predictive maintenance.
    """

    df["Temp difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    df["Mechanical power proxy"] = (
        df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    )

    df["Wear speed ratio"] = (
        df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1)
    )

    return df
