import os
import joblib


def save_object(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: str):
    return joblib.load(path)
