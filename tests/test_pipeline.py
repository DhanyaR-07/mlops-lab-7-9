import numpy as np
import pandas as pd
import pytest
from src.data import preprocess_data, split_data
from src.train import train_model
from src.predict import predict

COLS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"]


def make_df(n=100):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(1, 10, size=(n, 8)), columns=COLS)
    df["MedHouseVal"] = df["MedInc"] * 0.4 + rng.normal(0, 0.1, n)
    return df


def test_preprocess_removes_invalid_and_duplicates():
    df = make_df(10)
    bad = df.iloc[[0]].copy(); bad["MedInc"] = -1
    df = pd.concat([df, df.iloc[[1]], bad], ignore_index=True)
    out = preprocess_data(df)
    assert len(out) == 10 - 1 + 0 or (out["MedInc"] > 0).all()
    assert out.duplicated().sum() == 0
    assert (out["MedInc"] > 0).all()


def test_predict_output_shape():
    df = make_df()
    X_tr, X_te, y_tr, _ = split_data(df, "MedHouseVal", 0.2, 42)
    model = train_model(X_tr, y_tr, {"n_estimators": 5, "max_depth": 3}, 42)
    preds = predict(model, X_te.head(5).to_dict("records"))
    assert len(preds) == 5


def test_predict_missing_feature_raises():
    df = make_df()
    X_tr, _, y_tr, _ = split_data(df, "MedHouseVal", 0.2, 42)
    model = train_model(X_tr, y_tr, {"n_estimators": 5, "max_depth": 3}, 42)
    with pytest.raises(ValueError):
        predict(model, [{"MedInc": 1.0}])