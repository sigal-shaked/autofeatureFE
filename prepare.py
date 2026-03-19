"""
Fixed data loading and pipeline executor — do NOT modify this file.

Reads task.json to select the dataset, applies the pipeline defined in
pipeline.json via operations.py, and returns train/val arrays for train.py.

Supported datasets
------------------
  california_housing  — regression,      8 features
  breast_cancer       — classification,  30 features
  wine                — classification,  13 features (3 classes)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from operations import apply_pipeline

# ─── Fixed constants — do NOT change ─────────────────────────────────────────
RANDOM_SEED   = 42
VAL_SIZE      = 0.2
TASK_FILE     = Path(__file__).parent / "task.json"
PIPELINE_FILE = Path(__file__).parent / "pipeline.json"
# ─────────────────────────────────────────────────────────────────────────────


def _load_dataset(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (df_features, y) for the requested dataset."""
    if dataset == "california_housing":
        from sklearn.datasets import fetch_california_housing
        raw = fetch_california_housing()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        y  = raw.target.astype(np.float32)

    elif dataset == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        raw = load_breast_cancer()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        y  = raw.target.astype(np.int32)

    elif dataset == "wine":
        from sklearn.datasets import load_wine
        raw = load_wine()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        y  = raw.target.astype(np.int32)

    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            "Supported: california_housing, breast_cancer, wine"
        )

    return df, y


def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data, apply pipeline.json, return (X_train, X_val, y_train, y_val)."""
    task_cfg = json.loads(TASK_FILE.read_text())
    dataset  = task_cfg["dataset"]

    df, y = _load_dataset(dataset)

    # Fixed train / val split — never changes
    idx = np.arange(len(df))
    idx_train, idx_val = train_test_split(
        idx, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val   = df.iloc[idx_val].reset_index(drop=True)
    y_train  = y[idx_train]
    y_val    = y[idx_val]

    # Apply feature engineering pipeline
    pipeline_cfg = json.loads(PIPELINE_FILE.read_text())
    df_train, df_val = apply_pipeline(pipeline_cfg["steps"], df_train, df_val)

    return (
        df_train.values.astype(np.float32),
        df_val.values.astype(np.float32),
        y_train,
        y_val,
    )


def get_feature_names() -> list[str]:
    """Return original feature names for the configured dataset (used by agent)."""
    task_cfg = json.loads(TASK_FILE.read_text())
    df, _    = _load_dataset(task_cfg["dataset"])
    return df.columns.tolist()


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = prepare_data()
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"y dtype : {y_train.dtype}  range [{y_train.min():.3f}, {y_train.max():.3f}]")
