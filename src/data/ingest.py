import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.
    """
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
