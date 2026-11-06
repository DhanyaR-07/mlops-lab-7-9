"""Data loading, cleaning and splitting."""
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_raw_dataset(path: str) -> None:
    """Download the public California housing data and save it as CSV."""
    from sklearn.datasets import fetch_california_housing
    df = fetch_california_housing(as_frame=True).frame
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Raw dataset saved to {path} ({len(df)} rows)")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}. Run: python -m src.data")
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, missing and invalid rows."""
    df = df.drop_duplicates().dropna()
    df = df[(df["MedInc"] > 0) & (df["HouseAge"] >= 0) & (df["AveRooms"] > 0)
            & (df["Population"] > 0) & (df["AveOccup"] > 0)]
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, target: str, test_size: float, random_state: int):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    cfg = load_config()
    make_raw_dataset(cfg["data"]["raw_path"])