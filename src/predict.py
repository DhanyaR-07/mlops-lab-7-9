"""Load the saved pipeline and predict new records."""
import os
import joblib
import pandas as pd

DEFAULT_MODEL_PATH = "models/model.joblib"


def load_model(path: str | None = None):
    path = path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def predict(model, records: list[dict]) -> list[float]:
    """records: list of dicts with the model's feature names."""
    df = pd.DataFrame(records)
    missing = set(model.feature_names_in_) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    df = df[list(model.feature_names_in_)]  # enforce exact column order
    return model.predict(df).tolist()


if __name__ == "__main__":
    m = load_model()
    sample = {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02,
              "Population": 322.0, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23}
    print(predict(m, [sample]))