"""
Fixed training script — do NOT modify this file.

Reads task.json to select the model type and evaluation metric.
Prints three lines consumed by agent.py:

    val_score  : <float>   — raw metric value (rmse | auc | logloss)
    metric_name: <str>     — metric identity so agent knows direction
    n_features : <int>
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

# ─── Fixed model hyperparameters — do NOT change ─────────────────────────────
_BASE_PARAMS = dict(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    verbosity=0,
)
# ─────────────────────────────────────────────────────────────────────────────

TASK_FILE = Path(__file__).parent / "task.json"


def main() -> None:
    t0 = time.perf_counter()

    task_cfg    = json.loads(TASK_FILE.read_text())
    task        = task_cfg["task"]        # "regression" | "classification"
    metric_name = task_cfg["metric"]      # "rmse" | "auc" | "logloss"

    from prepare import prepare_data
    X_train, X_val, y_train, y_val = prepare_data()

    n_train, n_features = X_train.shape
    print(f"n_train   : {n_train}")
    print(f"n_features: {n_features}")

    if task == "regression":
        _run_regression(X_train, X_val, y_train, y_val, metric_name)

    elif task == "classification":
        n_classes = len(np.unique(y_train))
        _run_classification(X_train, X_val, y_train, y_val, metric_name, n_classes)

    else:
        sys.exit(f"ERROR: Unknown task {task!r}. Use 'regression' or 'classification'.")

    print(f"elapsed_s : {time.perf_counter() - t0:.1f}")


def _run_regression(X_train, X_val, y_train, y_val, metric_name: str) -> None:
    from xgboost import XGBRegressor
    from sklearn.metrics import root_mean_squared_error

    if metric_name != "rmse":
        print(f"WARNING: metric {metric_name!r} not supported for regression; using rmse.")
        metric_name = "rmse"

    model = XGBRegressor(
        **_BASE_PARAMS,
        eval_metric="rmse",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    score  = root_mean_squared_error(y_val, y_pred)

    print(f"val_score  : {score:.6f}")
    print(f"metric_name: rmse")


def _run_classification(X_train, X_val, y_train, y_val, metric_name: str, n_classes: int) -> None:
    from xgboost import XGBClassifier

    if metric_name not in ("auc", "logloss"):
        print(f"WARNING: metric {metric_name!r} not supported for classification; using auc.")
        metric_name = "auc"

    binary = n_classes == 2
    xgb_eval = "auc" if (metric_name == "auc" and binary) else "mlogloss"

    model = XGBClassifier(
        **_BASE_PARAMS,
        objective="binary:logistic" if binary else "multi:softprob",
        num_class=None if binary else n_classes,
        eval_metric=xgb_eval,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    if metric_name == "auc":
        from sklearn.metrics import roc_auc_score
        y_proba = model.predict_proba(X_val)
        if binary:
            score = roc_auc_score(y_val, y_proba[:, 1])
        else:
            score = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")

    else:  # logloss
        from sklearn.metrics import log_loss
        y_proba = model.predict_proba(X_val)
        score   = log_loss(y_val, y_proba)

    print(f"val_score  : {score:.6f}")
    print(f"metric_name: {metric_name}")


if __name__ == "__main__":
    main()
