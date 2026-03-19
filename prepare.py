"""
Feature engineering and preprocessing pipeline.

This file is modified by the agent to improve feature quality.
The agent may freely edit everything below the dashed separator.

Returns:
    X_train, X_val: numpy arrays of shape (n_samples, n_features)
    y_train, y_val: numpy arrays of shape (n_samples,)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Fixed constants — do NOT change ─────────────────────────────────────────
RANDOM_SEED = 42
VAL_SIZE = 0.2
# ─────────────────────────────────────────────────────────────────────────────

# EXPERIMENT: baseline — raw features with standard scaling


def prepare_data():
    """Load, preprocess, and engineer features.

    Returns (X_train, X_val, y_train, y_val) as float32 numpy arrays.
    """
    # Load dataset
    # Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    raw = fetch_california_housing()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = raw.target.astype(np.float32)

    # ── Feature engineering ───────────────────────────────────────────────────
    # (baseline: no extra features — agent will improve this)

    X = df.copy()

    # ── Feature selection ─────────────────────────────────────────────────────
    # (baseline: keep all features)

    # ── Train / val split (do NOT change seed or size) ───────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    # ── Scaling ───────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = prepare_data()
    print(f"Train : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Target: min={y_train.min():.3f}  max={y_train.max():.3f}  mean={y_train.mean():.3f}")
