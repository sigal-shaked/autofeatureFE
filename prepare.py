"""
Fixed data loading and pipeline executor — do NOT modify this file.

Flow
----
  1. Load raw data       (_load_raw)
  2. Infer field types   (field_types.infer_field_types)
  3. Train/val split     (fixed seed + ratio)
  4. Auto-encode         (field_types.auto_encode)  — drop IDs, encode categoricals
  5. Apply pipeline      (operations.apply_pipeline) — agent-controlled feature ops
  6. Return float32 arrays

Supported datasets
------------------
  california_housing  — regression,      8 features
  breast_cancer       — classification,  30 features
  wine                — classification,  13 features (3 classes)
  csv                 — any tabular CSV; set csv_path + target_column in task.json
  openml              — any OpenML dataset; set openml_id or openml_suite in task.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from operations import apply_pipeline
from field_types import infer_field_types, auto_encode

# ─── Fixed constants — do NOT change ─────────────────────────────────────────
RANDOM_SEED   = 42
VAL_SIZE      = 0.2
TASK_FILE     = Path(__file__).parent / "task.json"
PIPELINE_FILE = Path(__file__).parent / "pipeline.json"
# ─────────────────────────────────────────────────────────────────────────────

# Suite name → OpenML suite ID
_OPENML_SUITE_IDS = {
    "OpenML-CC18": 99,  "cc18": 99,
    "OpenML-CTR23": 353, "ctr23": 353,
}


def _field_types_cache_path(task_cfg: dict) -> Path:
    key = task_cfg.get("openml_id") or task_cfg.get("openml_suite") or task_cfg["dataset"]
    idx = task_cfg.get("openml_suite_index", "")
    suffix = f"_{idx}" if idx != "" else ""
    return Path(__file__).parent / f"field_types_{key}{suffix}.json"


def _load_raw(task_cfg: dict) -> tuple[pd.DataFrame, np.ndarray, dict | None]:
    """
    Return (df_features, y, openml_features_or_None).

    openml_features is a dict {col_name: OpenMLDataFeature} when loaded from
    OpenML; None for sklearn / CSV datasets.
    """
    dataset = task_cfg["dataset"]

    if dataset == "california_housing":
        from sklearn.datasets import fetch_california_housing
        raw = fetch_california_housing()
        df  = pd.DataFrame(raw.data, columns=raw.feature_names)
        y   = raw.target.astype(np.float32)
        return df, y, None

    elif dataset == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        raw = load_breast_cancer()
        df  = pd.DataFrame(raw.data, columns=raw.feature_names)
        y   = raw.target.astype(np.int32)
        return df, y, None

    elif dataset == "wine":
        from sklearn.datasets import load_wine
        raw = load_wine()
        df  = pd.DataFrame(raw.data, columns=raw.feature_names)
        y   = raw.target.astype(np.int32)
        return df, y, None

    elif dataset == "csv":
        csv_path   = Path(__file__).parent / task_cfg["csv_path"]
        target_col = task_cfg["target_column"]
        df_full    = pd.read_csv(csv_path)
        y          = df_full[target_col].values
        df         = df_full.drop(columns=[target_col])
        if task_cfg["task"] == "classification":
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y.astype(str)).astype(np.int32)
        else:
            y = y.astype(np.float32)
        return df, y, None

    elif dataset == "openml":
        return _load_openml(task_cfg)

    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            "Supported: california_housing, breast_cancer, wine, csv, openml"
        )


def _load_openml(task_cfg: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load a dataset from OpenML by ID or benchmark suite."""
    import openml

    if "openml_id" in task_cfg:
        ds     = openml.datasets.get_dataset(
            task_cfg["openml_id"],
            download_data=True,
            download_qualities=False,
            download_features_meta_data=True,
        )
        target = task_cfg.get("target_column") or ds.default_target_attribute

    elif "openml_suite" in task_cfg:
        suite_key = task_cfg["openml_suite"]
        suite_id  = _OPENML_SUITE_IDS.get(suite_key, suite_key)
        suite     = openml.study.get_suite(suite_id)
        idx       = task_cfg.get("openml_suite_index", 0)
        task      = openml.tasks.get_task(suite.tasks[idx])
        ds        = openml.datasets.get_dataset(
            task.dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=True,
        )
        target = task.target_name

    else:
        raise ValueError(
            "OpenML dataset requires 'openml_id' or 'openml_suite' in task.json. "
            "Example: {\"dataset\": \"openml\", \"openml_id\": 31}"
        )

    X, y_series, _, _ = ds.get_data(target=target, dataset_format="dataframe")

    # OpenML feature metadata keyed by column name
    openml_features = {f.name: f for f in ds.features.values()}

    # Encode target
    if task_cfg["task"] == "classification":
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(
            y_series.fillna("__missing__").astype(str)
        ).astype(np.int32)
    else:
        y = pd.to_numeric(y_series, errors="coerce").fillna(0).values.astype(np.float32)

    return X, y, openml_features


def _get_field_types(
    task_cfg: dict,
    df: pd.DataFrame,
    openml_features: dict | None,
) -> dict[str, str]:
    """Infer + cache field types.  Cached file acts as human-editable override."""
    cache = _field_types_cache_path(task_cfg)
    types = infer_field_types(df, openml_features=openml_features, override_file=cache)
    # Write cache if it doesn't exist yet (first run)
    if not cache.exists():
        cache.write_text(json.dumps(types, indent=2) + "\n")
    return types


# ─── Public API ───────────────────────────────────────────────────────────────

def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data, auto-encode, apply pipeline.json → (X_train, X_val, y_train, y_val)."""
    task_cfg = json.loads(TASK_FILE.read_text())

    df, y, openml_features = _load_raw(task_cfg)
    field_types            = _get_field_types(task_cfg, df, openml_features)

    # Fixed train/val split — never changes
    idx = np.arange(len(df))
    idx_train, idx_val = train_test_split(idx, test_size=VAL_SIZE, random_state=RANDOM_SEED)
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val   = df.iloc[idx_val].reset_index(drop=True)
    y_train  = y[idx_train]
    y_val    = y[idx_val]

    # Auto-encode (before pipeline): drop IDs, expand datetimes, ordinal-encode categoricals
    df_train, df_val, _ = auto_encode(df_train, df_val, field_types)

    # Agent-controlled feature engineering pipeline
    pipeline_cfg = json.loads(PIPELINE_FILE.read_text())
    df_train, df_val = apply_pipeline(pipeline_cfg["steps"], df_train, df_val)

    return (
        df_train.values.astype(np.float32),
        df_val.values.astype(np.float32),
        y_train,
        y_val,
    )


def get_feature_names() -> list[str]:
    """Return feature names after auto-encoding (used by agent to build prompts)."""
    task_cfg = json.loads(TASK_FILE.read_text())
    df, _, openml_features = _load_raw(task_cfg)
    field_types            = _get_field_types(task_cfg, df, openml_features)
    # Apply auto-encode on a tiny copy just to get the final column names
    dummy = df.head(2).reset_index(drop=True)
    encoded, _, _ = auto_encode(dummy, dummy.copy(), field_types)
    return encoded.columns.tolist()


def get_field_types() -> dict[str, str]:
    """Return {col: type} for the configured dataset (cached after first call)."""
    task_cfg = json.loads(TASK_FILE.read_text())
    cache    = _field_types_cache_path(task_cfg)
    if cache.exists():
        return json.loads(cache.read_text())
    df, _, openml_features = _load_raw(task_cfg)
    return _get_field_types(task_cfg, df, openml_features)


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = prepare_data()
    print(f"Train   : {X_train.shape}")
    print(f"Val     : {X_val.shape}")
    print(f"y dtype : {y_train.dtype}  range [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"Features: {get_feature_names()}")
