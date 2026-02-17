from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data.ingest import load_raw_data
from src.data.preprocess import preprocess_data
from src.utils.helpers import save_object


DATA_PATH = "data/raw/ai4i2020.csv"
MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def train():
    df = load_raw_data(DATA_PATH)

    X_scaled, y, scaler, feature_names = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    save_object(model, MODEL_PATH)
    save_object(scaler, SCALER_PATH)

    print("Training complete. Model saved.")
