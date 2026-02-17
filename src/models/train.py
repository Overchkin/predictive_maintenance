from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils.helpers import save_object
import os

def train(X_train, y_train, model_path=None, scaler_path=None):
    """
    Entraîne un Random Forest Classifier avec StandardScaler sur les colonnes numériques.
    X_train : DataFrame pandas contenant toutes les features
    y_train : Series pandas, target
    """
    numeric_features = ["Air temperature [K]", "Process temperature [K]",
                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

    # Scaler uniquement sur les colonnes numériques
    scaler = StandardScaler()
    X_scaled = X_train.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])

    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_train)

    # Chemins par défaut
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
    if scaler_path is None:
        scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

    save_object(model, model_path)
    save_object(scaler, scaler_path)

    return model, scaler
