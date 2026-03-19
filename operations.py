"""
Fixed operation library — do NOT modify this file.

Each operation takes (step_params, df_train, df_val), fits on df_train,
and transforms both DataFrames.  Operations are applied in pipeline order,
so later steps can reference features created by earlier steps.
"""

import builtins as _builtins
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

# ── Dispatch ──────────────────────────────────────────────────────────────────

ALLOWED_OPS = {
    # unary transforms (in-place)
    "log1p", "sqrt", "square", "cube", "reciprocal", "abs",
    # stateful unary (fit on train)
    "clip", "rank", "quantile_normal", "bin",
    # binary → new feature
    "ratio", "product", "diff", "sum_pair", "log_ratio",
    # multi-feature → new features
    "polynomial", "interaction",
    # geographic
    "kmeans_cluster", "kmeans_distance", "distance_to_point",
    # selection
    "drop", "select",
    # scaling (fit on train)
    "scale",
    # freestyle (opt-in, guarded by agent.py validation)
    "freestyle",
}


def apply_step(
    step: dict,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch a single pipeline step."""
    op = step["op"]
    if op not in ALLOWED_OPS:
        raise ValueError(
            f"Unknown operation {op!r}. Allowed: {sorted(ALLOWED_OPS)}"
        )
    fn = globals()[f"_op_{op}"]
    return fn(step, df_train.copy(), df_val.copy())


def apply_pipeline(
    steps: list[dict],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all pipeline steps in order."""
    for i, step in enumerate(steps):
        try:
            df_train, df_val = apply_step(step, df_train, df_val)
        except Exception as e:
            raise RuntimeError(f"Step {i} ({step.get('op')!r}) failed: {e}") from e
    return df_train, df_val


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return a / (np.where(b >= 0, b + eps, b - eps))


# ── Unary transforms (stateless) ──────────────────────────────────────────────


def _op_log1p(step, tr, va):
    """log(1 + x). Clips negatives to 0 first.
    params: features (list[str])
    """
    for f in step["features"]:
        tr[f] = np.log1p(np.maximum(tr[f].values, 0.0))
        va[f] = np.log1p(np.maximum(va[f].values, 0.0))
    return tr, va


def _op_sqrt(step, tr, va):
    """sqrt(|x|). Safe for negative values.
    params: features (list[str])
    """
    for f in step["features"]:
        tr[f] = np.sqrt(np.abs(tr[f].values))
        va[f] = np.sqrt(np.abs(va[f].values))
    return tr, va


def _op_square(step, tr, va):
    """x².
    params: features (list[str])
    """
    for f in step["features"]:
        tr[f] = tr[f].values ** 2
        va[f] = va[f].values ** 2
    return tr, va


def _op_cube(step, tr, va):
    """x³.
    params: features (list[str])
    """
    for f in step["features"]:
        tr[f] = tr[f].values ** 3
        va[f] = va[f].values ** 3
    return tr, va


def _op_reciprocal(step, tr, va):
    """1 / (x + epsilon).  Good for 1/AveOccup etc.
    params: features (list[str]), epsilon (float, default 1e-3)
    """
    eps = step.get("epsilon", 1e-3)
    for f in step["features"]:
        tr[f] = 1.0 / (tr[f].values + eps)
        va[f] = 1.0 / (va[f].values + eps)
    return tr, va


def _op_abs(step, tr, va):
    """Absolute value.
    params: features (list[str])
    """
    for f in step["features"]:
        tr[f] = np.abs(tr[f].values)
        va[f] = np.abs(va[f].values)
    return tr, va


# ── Unary transforms (stateful — fit on train) ────────────────────────────────


def _op_clip(step, tr, va):
    """Clip to [p_low, p_high] percentiles (fit on train).
    params: features (list[str]), low_pct (float, default 1), high_pct (float, default 99)
    """
    low_pct = step.get("low_pct", 1)
    high_pct = step.get("high_pct", 99)
    for f in step["features"]:
        lo = float(np.percentile(tr[f], low_pct))
        hi = float(np.percentile(tr[f], high_pct))
        tr[f] = tr[f].values.clip(lo, hi)
        va[f] = va[f].values.clip(lo, hi)
    return tr, va


def _op_rank(step, tr, va):
    """Map values to [0, 1] rank based on train distribution.
    params: features (list[str])
    """
    for f in step["features"]:
        sorted_vals = np.sort(tr[f].values)
        n = len(sorted_vals)
        tr[f] = np.searchsorted(sorted_vals, tr[f].values) / n
        va[f] = np.searchsorted(sorted_vals, va[f].values) / n
    return tr, va


def _op_quantile_normal(step, tr, va):
    """Quantile-transform a feature to a standard normal distribution (fit on train).
    params: features (list[str])
    """
    for f in step["features"]:
        qt = QuantileTransformer(output_distribution="normal", random_state=42)
        tr[f] = qt.fit_transform(tr[[f]]).ravel()
        va[f] = qt.transform(va[[f]]).ravel()
    return tr, va


def _op_bin(step, tr, va):
    """Bin into equal-frequency buckets (fit bin edges on train).
    params: features (list[str]), n_bins (int, default 10)
    """
    n_bins = step.get("n_bins", 10)
    for f in step["features"]:
        _, edges = pd.qcut(tr[f], q=n_bins, retbins=True, duplicates="drop")
        tr[f] = pd.cut(tr[f], bins=edges, labels=False, include_lowest=True).astype(float)
        va[f] = pd.cut(va[f], bins=edges, labels=False, include_lowest=True).fillna(0).astype(float)
    return tr, va


# ── Binary ops (create new features) ─────────────────────────────────────────


def _op_ratio(step, tr, va):
    """numerator / denominator → new column.
    params: numerator (str), denominator (str), name (str), epsilon (float, default 1e-6)
    """
    eps = step.get("epsilon", 1e-6)
    name = step["name"]
    tr[name] = _safe_div(tr[step["numerator"]].values, tr[step["denominator"]].values, eps)
    va[name] = _safe_div(va[step["numerator"]].values, va[step["denominator"]].values, eps)
    return tr, va


def _op_product(step, tr, va):
    """a * b → new column.
    params: a (str), b (str), name (str)
    """
    tr[step["name"]] = tr[step["a"]].values * tr[step["b"]].values
    va[step["name"]] = va[step["a"]].values * va[step["b"]].values
    return tr, va


def _op_diff(step, tr, va):
    """a - b → new column.
    params: a (str), b (str), name (str)
    """
    tr[step["name"]] = tr[step["a"]].values - tr[step["b"]].values
    va[step["name"]] = va[step["a"]].values - va[step["b"]].values
    return tr, va


def _op_sum_pair(step, tr, va):
    """a + b → new column.
    params: a (str), b (str), name (str)
    """
    tr[step["name"]] = tr[step["a"]].values + tr[step["b"]].values
    va[step["name"]] = va[step["a"]].values + va[step["b"]].values
    return tr, va


def _op_log_ratio(step, tr, va):
    """log(numerator / denominator) → new column.
    params: numerator (str), denominator (str), name (str), epsilon (float, default 1e-6)
    """
    eps = step.get("epsilon", 1e-6)
    num = tr[step["numerator"]].values + eps
    den = tr[step["denominator"]].values + eps
    tr[step["name"]] = np.log(np.abs(num / den))
    num = va[step["numerator"]].values + eps
    den = va[step["denominator"]].values + eps
    va[step["name"]] = np.log(np.abs(num / den))
    return tr, va


# ── Multi-feature ops ─────────────────────────────────────────────────────────


def _op_polynomial(step, tr, va):
    """Polynomial + interaction features on a feature subset (fit on train).
    Adds new columns; does NOT remove originals.
    params: features (list[str]), degree (int, default 2), interaction_only (bool, default false)
    """
    features = step["features"]
    degree = step.get("degree", 2)
    interaction_only = step.get("interaction_only", False)
    pf = PolynomialFeatures(
        degree=degree, interaction_only=interaction_only, include_bias=False
    )
    X_tr = pf.fit_transform(tr[features].values)
    X_va = pf.transform(va[features].values)
    names = pf.get_feature_names_out(features)
    for i, col in enumerate(names):
        if col not in tr.columns:
            tr[col] = X_tr[:, i]
            va[col] = X_va[:, i]
    return tr, va


def _op_interaction(step, tr, va):
    """All pairwise products of a feature subset → new columns.
    params: features (list[str])
    """
    features = step["features"]
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            col = f"{a}_x_{b}"
            tr[col] = tr[a].values * tr[b].values
            va[col] = va[a].values * va[b].values
    return tr, va


# ── Geographic ops ────────────────────────────────────────────────────────────


def _op_kmeans_cluster(step, tr, va):
    """K-Means cluster assignment → new integer column (fit on train).
    params: features (list[str]), n_clusters (int, default 10), name (str, default "geo_cluster")
    """
    features = step["features"]
    n_clusters = step.get("n_clusters", 10)
    name = step.get("name", "geo_cluster")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    tr[name] = km.fit_predict(tr[features].values).astype(float)
    va[name] = km.predict(va[features].values).astype(float)
    return tr, va


def _op_kmeans_distance(step, tr, va):
    """Distance to each K-Means centroid → n_clusters new columns (fit on train).
    params: features (list[str]), n_clusters (int, default 10), prefix (str, default "kdist")
    """
    features = step["features"]
    n_clusters = step.get("n_clusters", 10)
    prefix = step.get("prefix", "kdist")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(tr[features].values)
    for k, (tr_col, va_col) in enumerate(
        zip(km.transform(tr[features].values).T, km.transform(va[features].values).T)
    ):
        col = f"{prefix}_{k}"
        tr[col] = tr_col
        va[col] = va_col
    return tr, va


def _op_distance_to_point(step, tr, va):
    """Euclidean distance to a fixed (lat, lon) reference point → new column.
    params: lat (str), lon (str), target_lat (float), target_lon (float), name (str)
    Example reference points for California:
      San Francisco: target_lat=37.77, target_lon=-122.42
      Los Angeles:   target_lat=34.05, target_lon=-118.24
      San Diego:     target_lat=32.72, target_lon=-117.15
    """
    lat_col = step["lat"]
    lon_col = step["lon"]
    tlat = step["target_lat"]
    tlon = step["target_lon"]
    name = step["name"]
    tr[name] = np.sqrt((tr[lat_col].values - tlat) ** 2 + (tr[lon_col].values - tlon) ** 2)
    va[name] = np.sqrt((va[lat_col].values - tlat) ** 2 + (va[lon_col].values - tlon) ** 2)
    return tr, va


# ── Selection ─────────────────────────────────────────────────────────────────


def _op_drop(step, tr, va):
    """Drop specified columns.
    params: features (list[str])
    """
    cols = [f for f in step["features"] if f in tr.columns]
    return tr.drop(columns=cols), va.drop(columns=cols)


def _op_select(step, tr, va):
    """Keep only specified columns (in given order).
    params: features (list[str])
    """
    cols = [f for f in step["features"] if f in tr.columns]
    return tr[cols], va[cols]


# ── Scaling ───────────────────────────────────────────────────────────────────


def _op_scale(step, tr, va):
    """Scale all remaining features (fit on train).
    params: method (str): "standard" | "robust" | "minmax" | "quantile"
    """
    method = step.get("method", "standard")
    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "quantile":
        scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    else:
        raise ValueError(f"Unknown scale method {method!r}. Use: standard, robust, minmax, quantile")
    cols = tr.columns.tolist()
    tr_arr = scaler.fit_transform(tr.values)
    va_arr = scaler.transform(va.values)
    return pd.DataFrame(tr_arr, columns=cols), pd.DataFrame(va_arr, columns=cols)


# ── Freestyle (opt-in, disabled by default) ───────────────────────────────────
#
# Guardrails applied in layers:
#   1. Pattern blocklist  — catches obvious attacks at the source string level
#   2. Restricted globals — exec sees only np, pd, and a safe builtin whitelist
#   3. Post-exec checks   — row count, dtype, NaN/inf validation
#
# The same code snippet runs independently on df_train and df_val,
# so it cannot observe which split it is operating on.

_FREESTYLE_BLOCKED_PATTERNS = [
    # imports
    "import ",          # catches "import os", "from os import ..."
    "__import__",       # dynamic import
    # code execution
    "eval(",
    "exec(",
    "compile(",
    # file / network / process
    "open(",
    "socket",
    "urllib",
    "requests",
    "subprocess",
    "shutil",
    # introspection / escape hatches
    "globals(",
    "locals(",
    "vars(",
    "dir(",
    "getattr(",
    "setattr(",
    "delattr(",
    "hasattr(",
    # dunder attribute access (e.g. obj.__class__.__subclasses__())
    ".__",
    "['__",
    '["__',
    # process control
    "exit(",
    "quit(",
    "breakpoint(",
    "input(",
]

# Whitelist of builtins available inside freestyle code.
_FREESTYLE_SAFE_BUILTINS: dict = {
    name: getattr(_builtins, name)
    for name in [
        "abs", "bool", "dict", "enumerate", "filter", "float",
        "int", "isinstance", "len", "list", "map", "max", "min",
        "range", "round", "set", "sorted", "str", "sum", "tuple",
        "zip", "print",                     # print is fine for debugging
        "True", "False", "None",
        "ValueError", "TypeError",          # allow raising these
    ]
    if hasattr(_builtins, name)
}

_FREESTYLE_MAX_LINES = 15
_FREESTYLE_MAX_CHARS = 600


def check_freestyle_safety(code: str, name: str = "freestyle") -> None:
    """Raise ValueError if the code contains any blocked pattern.
    Called both from agent.py (before the pipeline is committed) and
    from _op_freestyle (defense-in-depth at execution time).
    """
    if not code or not code.strip():
        raise ValueError(f"freestyle '{name}': 'code' is empty")

    n_lines = len(code.splitlines())
    if n_lines > _FREESTYLE_MAX_LINES:
        raise ValueError(
            f"freestyle '{name}': code is {n_lines} lines; "
            f"max allowed is {_FREESTYLE_MAX_LINES}"
        )
    if len(code) > _FREESTYLE_MAX_CHARS:
        raise ValueError(
            f"freestyle '{name}': code is {len(code)} chars; "
            f"max allowed is {_FREESTYLE_MAX_CHARS}"
        )

    for pattern in _FREESTYLE_BLOCKED_PATTERNS:
        if pattern in code:
            raise ValueError(
                f"freestyle '{name}': blocked pattern {pattern!r} found in code"
            )


def _op_freestyle(step, tr, va):
    """Execute a short Python snippet to create or transform features.

    The snippet runs independently on df_train and df_val (same code, no
    shared state), so it cannot observe which split it is processing.

    params:
      code (str) — Python snippet; `df` is the current DataFrame,
                   `np` and `pd` are available.  Modify `df` in-place
                   or reassign columns.  Do NOT filter rows.
      name (str) — human-readable label for this step (used in logs)

    Example:
      {"op": "freestyle",
       "code": "df['family_size'] = df['sibsp'] + df['parch'] + 1",
       "name": "family size"}
    """
    code = step.get("code", "")
    name = step.get("name", "freestyle")

    # Layer 1: pattern check (defense-in-depth — agent also checks before commit)
    check_freestyle_safety(code, name)

    def _exec_on(df: pd.DataFrame) -> pd.DataFrame:
        # Layer 2: restricted execution environment
        globs = {"__builtins__": _FREESTYLE_SAFE_BUILTINS, "np": np, "pd": pd}
        locs  = {"df": df.copy()}
        exec(code, globs, locs)  # noqa: S102
        return locs["df"]

    tr_out = _exec_on(tr)
    va_out = _exec_on(va)

    # Layer 3: post-execution checks
    if len(tr_out) != len(tr):
        raise ValueError(
            f"freestyle '{name}': row count changed "
            f"(train {len(tr)} → {len(tr_out)}). Code must not filter rows."
        )
    if len(va_out) != len(va):
        raise ValueError(
            f"freestyle '{name}': row count changed "
            f"(val {len(va)} → {len(va_out)}). Code must not filter rows."
        )

    bad_cols = [c for c in tr_out.columns if tr_out[c].dtype == object]
    if bad_cols:
        raise ValueError(
            f"freestyle '{name}': all columns must be numeric; "
            f"got object dtype in: {bad_cols}"
        )

    # Replace inf/NaN silently (common in ratio-like expressions)
    tr_out = tr_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    va_out = va_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return tr_out, va_out
