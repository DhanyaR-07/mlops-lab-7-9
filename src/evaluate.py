"""Evaluate the saved model on held-out data and write a report."""
import json
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_config, load_data


def evaluate_model(model_path: str, test_path: str, target: str, metrics_path: str) -> dict:
    model = joblib.load(model_path)
    test = load_data(test_path)
    X, y = test.drop(columns=[target]), test[target]
    pred = model.predict(X)
    metrics = {
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "n_test_rows": int(len(y)),
    }
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    cfg = load_config()
    print(evaluate_model(cfg["paths"]["model_path"], cfg["data"]["test_path"],
                         cfg["data"]["target"], cfg["paths"]["metrics_path"]))