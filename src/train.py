"""Train the model and save the full inference pipeline."""
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_config, load_data, preprocess_data, split_data


def train_model(X_train, y_train, model_cfg: dict, random_state: int) -> Pipeline:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)  # preprocessing is fit on training data only
    return pipeline


def main():
    cfg = load_config()
    df = preprocess_data(load_data(cfg["data"]["raw_path"]))
    os.makedirs(os.path.dirname(cfg["data"]["processed_path"]), exist_ok=True)
    df.to_csv(cfg["data"]["processed_path"], index=False)

    rs = cfg["training"]["random_state"]
    X_train, X_test, y_train, y_test = split_data(
        df, cfg["data"]["target"], cfg["training"]["test_size"], rs)

    # Save the held-out test set for evaluate.py
    test_df = X_test.copy()
    test_df[cfg["data"]["target"]] = y_test
    test_df.to_csv(cfg["data"]["test_path"], index=False)

    pipeline = train_model(X_train, y_train, cfg["model"], rs)

    os.makedirs(os.path.dirname(cfg["paths"]["model_path"]), exist_ok=True)
    joblib.dump(pipeline, cfg["paths"]["model_path"])
    print(f"Model saved to {cfg['paths']['model_path']}")

    pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"Test RMSE={rmse:.4f}  R2={r2:.4f}")

    # Optional MLflow logging
    try:
        import mlflow
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        with mlflow.start_run():
            mlflow.log_params({**cfg["model"], **cfg["training"]})
            mlflow.log_metrics({"rmse": rmse, "r2": r2})
            mlflow.log_artifact(cfg["paths"]["model_path"])
    except ImportError:
        print("MLflow not installed; skipping tracking.")


if __name__ == "__main__":
    main()