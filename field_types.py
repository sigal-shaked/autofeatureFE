"""
Field type inference and automatic encoding for tabular datasets.

Types
-----
  numeric     — continuous float/int; apply transforms/scaling as usual
  categorical — nominal (string or low-cardinality int); ordinal-encoded before pipeline
  ordinal     — ordered categorical; treated same as categorical for encoding
  id          — unique row identifier; dropped automatically
  datetime    — date/time column; decomposed into numeric components

Resolution priority
-------------------
  1. Per-dataset override file  field_types_<id>.json  (human-editable, highest priority)
  2. OpenML metadata            (if dataset loaded from OpenML)
  3. Statistical heuristics     (dtype, cardinality, name patterns)

LLM refinement
--------------
  Called once during data-profile generation (agent.py).  The LLM receives the
  heuristic types plus sample values and can correct mistakes.  The corrected
  types are written back to the override file so they persist.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Type constants ─────────────────────────────────────────────────────────────
NUMERIC     = "numeric"
CATEGORICAL = "categorical"
ORDINAL     = "ordinal"
ID          = "id"
DATETIME    = "datetime"

VALID_TYPES = {NUMERIC, CATEGORICAL, ORDINAL, ID, DATETIME}

_ID_KEYWORDS       = {"id", "key", "index", "uuid", "guid", "code", "no", "num", "number"}
_DATETIME_KEYWORDS = {"date", "time", "timestamp", "year", "month", "day", "hour", "week"}

# ── Heuristic inference ────────────────────────────────────────────────────────

def _heuristic(col: str, series: pd.Series, n_rows: int) -> str:
    col_l = col.lower().replace("-", "_").replace(" ", "_")
    col_words = set(col_l.split("_"))

    # Datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME

    # Object (string) dtype
    if series.dtype == object:
        # Try to detect datetime-ish strings
        if col_words & _DATETIME_KEYWORDS:
            sample = series.dropna().head(10)
            try:
                pd.to_datetime(sample)
                return DATETIME
            except (ValueError, TypeError):
                pass
        return CATEGORICAL

    # Boolean
    if series.dtype == bool:
        return CATEGORICAL

    n_unique = series.nunique(dropna=True)

    # Integer that uniquely identifies every row and has ID-like name → drop
    if pd.api.types.is_integer_dtype(series) and n_unique == n_rows and col_words & _ID_KEYWORDS:
        return ID

    # Integer or float with very low cardinality → likely categorical
    if n_unique <= 10 and not pd.api.types.is_float_dtype(series):
        return CATEGORICAL

    return NUMERIC


def infer_field_types(
    df: pd.DataFrame,
    openml_features: dict | None = None,   # {col_name: OpenMLDataFeature}
    override_file: Path | None = None,
) -> dict[str, str]:
    """
    Return {col: type_str} for every column in df.

    Parameters
    ----------
    df              : Raw feature DataFrame (before any encoding).
    openml_features : Optional dict of OpenML feature objects keyed by column name.
    override_file   : JSON file with type overrides (human-editable, highest priority).
    """
    n_rows = len(df)
    result: dict[str, str] = {}

    for col in df.columns:
        # 1. OpenML metadata
        if openml_features and col in openml_features:
            feat = openml_features[col]
            dtype = getattr(feat, "data_type", None)
            if dtype == "nominal":
                result[col] = CATEGORICAL
            elif dtype == "numeric":
                # Still apply heuristics to catch hidden IDs / low-cardinality ints
                result[col] = _heuristic(col, df[col], n_rows)
            elif dtype in ("string", "text"):
                result[col] = CATEGORICAL
            elif dtype == "date":
                result[col] = DATETIME
            else:
                result[col] = _heuristic(col, df[col], n_rows)
        else:
            # 2. Heuristics
            result[col] = _heuristic(col, df[col], n_rows)

    # 3. Override file (highest priority — applied last)
    if override_file and override_file.exists():
        overrides = json.loads(override_file.read_text())
        for col, t in overrides.items():
            if col in result and t in VALID_TYPES:
                result[col] = t

    return result


# ── Auto-encoding ──────────────────────────────────────────────────────────────

def auto_encode(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    field_types: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Apply automatic encoding based on field types.  Fits on df_train, transforms both.

    - id       → dropped
    - datetime → year / month / day / weekday / hour components added, original dropped
    - categorical / ordinal → ordinal-encoded (fit on train; unseen val values → -1)
    - numeric  → unchanged

    Returns (df_train_encoded, df_val_encoded, report) where report summarises
    what was done (shown in data profile).
    """
    df_train = df_train.copy()
    df_val   = df_val.copy()
    report: dict[str, list[str]] = {
        "dropped_id":          [],
        "encoded_categorical": [],
        "expanded_datetime":   [],
    }

    for col, ftype in field_types.items():
        if col not in df_train.columns:
            continue

        if ftype == ID:
            df_train.drop(columns=[col], inplace=True)
            df_val.drop(columns=[col], inplace=True)
            report["dropped_id"].append(col)

        elif ftype == DATETIME:
            for df in (df_train, df_val):
                parsed = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"]    = parsed.dt.year.fillna(0).astype(float)
                df[f"{col}_month"]   = parsed.dt.month.fillna(0).astype(float)
                df[f"{col}_day"]     = parsed.dt.day.fillna(0).astype(float)
                df[f"{col}_weekday"] = parsed.dt.weekday.fillna(0).astype(float)
                df[f"{col}_hour"]    = parsed.dt.hour.fillna(0).astype(float)
            df_train.drop(columns=[col], inplace=True)
            df_val.drop(columns=[col], inplace=True)
            report["expanded_datetime"].append(col)

        elif ftype in (CATEGORICAL, ORDINAL):
            # Build vocab from train; unseen values in val get -1
            vocab = {
                v: i
                for i, v in enumerate(df_train[col].astype(str).unique())
            }
            df_train[col] = df_train[col].astype(str).map(vocab).astype(float)
            df_val[col]   = df_val[col].astype(str).map(vocab).fillna(-1).astype(float)
            report["encoded_categorical"].append(col)

        # numeric: no change

    return df_train, df_val, report


# ── LLM refinement prompt ──────────────────────────────────────────────────────

LLM_REFINE_SYSTEM = """\
You are a data scientist reviewing automatically inferred column types for a tabular dataset.

Types:
  numeric     — continuous number (age, price, distance, ...)
  categorical — nominal label with no intrinsic order (country, product_type, ...)
  ordinal     — ordered category (low/medium/high, rating 1-5, ...)
  id          — unique row identifier that should be DROPPED (customer_id, row_no, ...)
  datetime    — date or timestamp (should be decomposed into components)

Be especially careful about:
- Integer columns that are actually IDs (very high cardinality, "id"/"key"/"no" in name)
- Low-cardinality integers that are actually categorical codes
- Columns that look numeric but represent ordered categories (e.g. 1=low, 2=med, 3=high)

Return ONLY a JSON object {col_name: new_type} for columns where you want to CHANGE the
inferred type.  Return {} if everything looks correct.  No explanation, just JSON.
"""


def build_llm_refine_prompt(df: pd.DataFrame, field_types: dict[str, str]) -> str:
    """Build the user message for LLM type-refinement."""
    lines = ["column | inferred_type | dtype | n_unique | n_missing | sample_values"]
    lines.append("-" * 80)
    for col, ftype in field_types.items():
        if col not in df.columns:
            continue
        s         = df[col]
        n_unique  = s.nunique(dropna=True)
        n_missing = s.isna().sum()
        sample    = s.dropna().head(5).tolist()
        lines.append(f"{col:<30} | {ftype:<12} | {str(s.dtype):<8} | {n_unique:>8} | {n_missing:>9} | {sample}")
    return "\n".join(lines)


def apply_llm_refinement(
    field_types: dict[str, str],
    llm_response: str,
    override_file: Path,
) -> dict[str, str]:
    """
    Parse the LLM JSON response and merge corrections into field_types.
    Saves updated types to override_file so they persist.
    """
    import re
    m = re.search(r"\{[^}]*\}", llm_response, re.DOTALL)
    if not m:
        return field_types

    try:
        corrections = json.loads(m.group())
    except json.JSONDecodeError:
        return field_types

    updated = dict(field_types)
    for col, t in corrections.items():
        if col in updated and t in VALID_TYPES:
            updated[col] = t

    # Persist only the corrections (override file stays minimal and human-readable)
    existing = json.loads(override_file.read_text()) if override_file.exists() else {}
    existing.update({c: t for c, t in corrections.items() if t in VALID_TYPES})
    override_file.write_text(json.dumps(existing, indent=2) + "\n")

    return updated
