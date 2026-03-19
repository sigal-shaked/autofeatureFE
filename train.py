"""
Fixed training script — do NOT modify this file.

Trains an XGBoost model on the features produced by prepare.py and
prints the validation RMSE so the agent can track progress.

Output lines consumed by the agent:
    val_rmse: <float>
    n_features: <int>
"""

import sys
import time
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

# ─── Fixed model hyperparameters — do NOT change ─────────────────────────────
MODEL_PARAMS = dict(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="rmse",
    verbosity=0,
)
# ─────────────────────────────────────────────────────────────────────────────


def main():
    t0 = time.perf_counter()

    # Load features from prepare.py
    from prepare import prepare_data

    X_train, X_val, y_train, y_val = prepare_data()

    n_train, n_features = X_train.shape
    print(f"n_train   : {n_train}")
    print(f"n_features: {n_features}")

    # Train
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    elapsed = time.perf_counter() - t0
    print(f"elapsed_s : {elapsed:.1f}")
    print(f"val_rmse  : {rmse:.6f}")


if __name__ == "__main__":
    main()
