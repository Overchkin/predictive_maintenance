import pickle
import os

def save_object(obj, filepath):
    """Sauvegarde un objet Python avec pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_object(filepath):
    """Charge un objet Python avec pickle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
    with open(filepath, "rb") as f:
        return pickle.load(f)
