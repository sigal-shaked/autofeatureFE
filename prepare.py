"""
Fixed data loading and pipeline executor — do NOT modify this file.

Loads the raw California Housing dataset, applies the pipeline defined in
pipeline.json via operations.py, and returns train/val arrays for train.py.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from operations import apply_pipeline

# ─── Fixed constants — do NOT change ─────────────────────────────────────────
RANDOM_SEED = 42
VAL_SIZE = 0.2
PIPELINE_FILE = Path(__file__).parent / "pipeline.json"
# ─────────────────────────────────────────────────────────────────────────────


def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data, apply pipeline.json, return (X_train, X_val, y_train, y_val)."""
    # Load raw features
    raw = fetch_california_housing()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = raw.target.astype(np.float32)

    # Fixed train / val split — indices never change
    idx = np.arange(len(df))
    idx_train, idx_val = train_test_split(
        idx, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    y_train = y[idx_train]
    y_val = y[idx_val]

    # Apply the feature engineering pipeline
    with open(PIPELINE_FILE) as f:
        config = json.load(f)
    df_train, df_val = apply_pipeline(config["steps"], df_train, df_val)

    return (
        df_train.values.astype(np.float32),
        df_val.values.astype(np.float32),
        y_train,
        y_val,
    )


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = prepare_data()
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Target: min={y_train.min():.3f}  max={y_train.max():.3f}  mean={y_train.mean():.3f}")
