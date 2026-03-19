#!/usr/bin/env python3
"""
AutoFeature Agent

Autonomous agent that iteratively improves pipeline.json using an LLM API.

Search strategy
---------------
Maintains a top-k pool of the best pipelines seen so far (ranked by score).
Each iteration selects a base from the pool via round-robin, asks the LLM to
improve it, evaluates, and updates the pool if the result qualifies.

This avoids tunnelling into local optima — different pool members represent
different regions of the feature engineering space.

Score
-----
  score = val_rmse + COMPLEXITY_ALPHA * log(n_features)

A small complexity penalty discourages unnecessary feature bloat.
Set COMPLEXITY_ALPHA=0 (env var) to rank purely by val_rmse.

The LLM may only compose operations from the fixed library in operations.py.
No arbitrary code is generated or executed.

Usage:
    ANTHROPIC_API_KEY=sk-...  uv run agent.py
    OPENAI_API_KEY=sk-...     LLM_PROVIDER=openai uv run agent.py
    LLM_MODEL=claude-opus-4-6 ANTHROPIC_API_KEY=sk-... uv run agent.py
    TOPK=3 COMPLEXITY_ALPHA=0 ANTHROPIC_API_KEY=sk-... uv run agent.py
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

TASK_FILE = Path("task.json")

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL    = "gpt-4o"
MAX_TOKENS       = 4096
MAX_LLM_RETRIES  = 3

# History shown to the LLM (all configurable via env vars):
#   INCLUDE_HISTORY=false  → omit history section entirely
#   HISTORY_SIZE=N         → max number of experiments to show (0 = unlimited)
#   HISTORY_FILTER=kept    → show only experiments that entered the pool
#                  all     → show every experiment (default)
INCLUDE_HISTORY  = os.environ.get("INCLUDE_HISTORY", "true").lower() != "false"
HISTORY_SIZE     = int(os.environ.get("HISTORY_SIZE", "20"))   # 0 = unlimited
HISTORY_FILTER   = os.environ.get("HISTORY_FILTER", "all")     # "all" | "kept"
TRAIN_TIMEOUT    = 120   # seconds

# Data profile — one-time LLM analysis of the raw dataset, optionally injected
# into every iteration prompt.
#   DATA_PROFILE=false      → disable entirely
#   DATA_PROFILE=true       → generate on first run, cache in data_profile_<dataset>.md
#   REGEN_PROFILE=true      → force regeneration even if cache exists
INCLUDE_DATA_PROFILE = os.environ.get("DATA_PROFILE", "true").lower() != "false"
REGEN_PROFILE        = os.environ.get("REGEN_PROFILE", "false").lower() == "true"

TOPK             = int(float(os.environ.get("TOPK", "5")))
COMPLEXITY_ALPHA = float(os.environ.get("COMPLEXITY_ALPHA", "0.005"))

# Freestyle op — disabled by default.  When enabled the LLM may propose a
# short Python snippet as an operation.  The code is safety-checked before
# it is committed or executed.  See operations.py for the full guardrail list.
#   ALLOW_FREESTYLE=true   → enable
ALLOW_FREESTYLE = os.environ.get("ALLOW_FREESTYLE", "false").lower() == "true"

RESULTS_FILE = Path("results.tsv")
PIPELINE_FILE = Path("pipeline.json")
POOL_FILE     = Path("topk_pool.json")   # not tracked by git

# ─── Operation reference (shown verbatim to the LLM) ─────────────────────────

_FREESTYLE_REFERENCE = """
── Freestyle (use ONLY when no built-in op can express the idea) ─────────────
{"op": "freestyle",
 "code": "df['family_size'] = df['sibsp'] + df['parch'] + 1",
 "name": "family size"}

Rules — strictly enforced; violations abort the run:
• `df` is the current pandas DataFrame; modify it in-place or assign columns.
• `np` (numpy) and `pd` (pandas) are the only available names. No imports.
• The SAME code runs independently on train and val — do NOT use row indices,
  row counts, or any value that differs between the two splits.
• Do NOT filter or reorder rows (row count must stay the same).
• All resulting columns must be numeric (float/int). No object/string columns.
• Max 15 lines / 600 characters.
• Blocked: import, eval, exec, open, globals, getattr, .__ and similar.
• Prefer built-in ops when possible — freestyle counts toward the complexity penalty.
"""


def _build_operations_reference(feature_names: list[str]) -> str:
    names_str = ", ".join(feature_names)
    return f"""\
AVAILABLE OPERATIONS
====================
All operations reference features by column name.  Binary / multi-feature ops
create NEW columns; originals are kept unless you add a "drop" step.
Steps are applied left-to-right; later steps can reference columns created earlier.

Original features: {names_str}

── Unary transforms (modify a feature in-place) ─────────────────────────────
{"op": "log1p",           "features": ["MedInc"]}
{"op": "sqrt",            "features": ["Population"]}
{"op": "square",          "features": ["MedInc"]}
{"op": "cube",            "features": ["MedInc"]}
{"op": "reciprocal",      "features": ["AveOccup"], "epsilon": 1e-3}
{"op": "abs",             "features": ["Latitude"]}
{"op": "clip",            "features": ["AveRooms"], "low_pct": 1, "high_pct": 99}
{"op": "rank",            "features": ["MedInc"]}             // → [0,1] based on train CDF
{"op": "quantile_normal", "features": ["Population"]}         // → N(0,1) via quantile transform
{"op": "bin",             "features": ["HouseAge"], "n_bins": 10}   // equal-frequency bins

── Binary ops (create a NEW named column) ───────────────────────────────────
{"op": "ratio",     "numerator": "AveRooms", "denominator": "AveBedrms", "name": "rooms_per_bedrm", "epsilon": 1e-6}
{"op": "product",   "a": "MedInc", "b": "AveRooms",   "name": "income_x_rooms"}
{"op": "diff",      "a": "AveRooms", "b": "AveBedrms", "name": "extra_rooms"}
{"op": "sum_pair",  "a": "AveRooms", "b": "AveBedrms", "name": "total_rooms"}
{"op": "log_ratio", "numerator": "AveRooms", "denominator": "AveBedrms", "name": "log_room_ratio", "epsilon": 1e-6}

── Multi-feature ops (create several new columns) ───────────────────────────
{"op": "polynomial", "features": ["MedInc", "AveOccup"], "degree": 2, "interaction_only": false}
{"op": "interaction", "features": ["MedInc", "Latitude", "Longitude"]}  // pairwise products

── Geographic ops ────────────────────────────────────────────────────────────
{"op": "kmeans_cluster",  "features": ["Latitude", "Longitude"], "n_clusters": 10, "name": "geo_cluster"}
{"op": "kmeans_distance", "features": ["Latitude", "Longitude"], "n_clusters": 8,  "prefix": "kdist"}
{"op": "distance_to_point", "lat": "Latitude", "lon": "Longitude",
    "target_lat": 37.77, "target_lon": -122.42, "name": "dist_sf"}
// CA reference points: LA (34.05,-118.24), SD (32.72,-117.15),
//   San Jose (37.34,-121.89), Sacramento (38.58,-121.49), Fresno (36.74,-119.79)

── Selection ─────────────────────────────────────────────────────────────────
{"op": "drop",   "features": ["AveBedrms"]}
{"op": "select", "features": ["MedInc", "Latitude", "Longitude"]}  // keep only these

── Scaling (always the last step) ───────────────────────────────────────────
{"op": "scale", "method": "standard"}   // StandardScaler
{"op": "scale", "method": "robust"}     // RobustScaler (percentile-based)
{"op": "scale", "method": "minmax"}     // MinMaxScaler → [0,1]
{"op": "scale", "method": "quantile"}   // QuantileTransformer → N(0,1)
""" + (_FREESTYLE_REFERENCE if ALLOW_FREESTYLE else "")


def _build_system_prompt(task_cfg: dict, feature_names: list[str]) -> str:
    task        = task_cfg["task"]    # regression | classification
    metric_name = task_cfg["metric"]  # rmse | auc | logloss
    dataset     = task_cfg["dataset"]

    ops_ref = _build_operations_reference(feature_names)

    n0 = len(feature_names)
    n1 = max(n0 * 4, 30)

    if metric_name == "rmse":
        metric_desc  = "val_rmse (lower is better)"
        primary_desc = "primary = val_rmse"
        base_primary = 0.430
    elif metric_name == "auc":
        metric_desc  = "val_auc (higher is better)"
        primary_desc = "primary = 1 − val_auc   ← so lower primary is always better"
        base_primary = 1 - 0.970
    else:  # logloss
        metric_desc  = "val_logloss (lower is better)"
        primary_desc = "primary = val_logloss"
        base_primary = 0.15

    score0 = base_primary + COMPLEXITY_ALPHA * math.log(n0)
    thresh = base_primary - COMPLEXITY_ALPHA * (math.log(n1) - math.log(n0))

    if task == "regression":
        strategy = """\
- Ratio features between related columns are often the strongest signal.
- Log-transform skewed features before using them in ratios.
- Geographic features (clusters, distance to city centres) capture spatial effects.
- Polynomial degree-2 on a SMALL set of strong predictors adds useful interactions.
- Clip outliers before log/ratio ops to avoid extreme values."""
    else:
        strategy = """\
- Ratio features between related columns often separate classes well.
- Log-transform skewed or heavy-tailed features to reduce their dynamic range.
- Polynomial degree-2 on a small set of discriminative features adds interactions.
- Dropping redundant or noisy features can help the classifier generalise.
- "robust" or "quantile" scaling is useful when features have very different scales."""

    return f"""\
You are an expert data scientist specialising in feature engineering for tabular {task}.

TASK
====
Dataset  : {dataset}
ML task  : {task}
Metric   : {metric_desc}

Improve the feature engineering pipeline to minimise the SCORE:

  {primary_desc}
  score = primary + {COMPLEXITY_ALPHA} × ln(n_features)

The complexity term penalises feature bloat.  With {n0} features scoring {score0:.4f},
a {n1}-feature pipeline needs primary < {thresh:.4f} to beat it.
The ML model (XGBoost) and ALL its hyperparameters are FIXED.

{ops_ref}

PIPELINE FORMAT
===============
Return the complete new pipeline as a JSON object inside a ```json ... ``` block:

```json
{{
  "description": "one-line description of the key change",
  "steps": [
    {{"op": "...", ...}},
    {{"op": "scale", "method": "standard"}}
  ]
}}
```

RULES
=====
1. Return the COMPLETE pipeline (all steps), not just the new/changed ones.
2. Always end with a "scale" step.
3. Use only operations listed above — unknown ops are rejected.
4. Columns created by earlier steps can be referenced by later steps.
5. Use the exact "name" / "prefix" you assigned when referencing new columns.

STRATEGY TIPS
=============
{strategy}
- Dropping weak features improves generalisation AND lowers the complexity penalty.
- Prefer simple pipelines: a small metric gain with many extra features may not improve score.

Think step by step, then output only the JSON.
"""

# ─── LLM client ───────────────────────────────────────────────────────────────


def detect_provider() -> tuple[str, str]:
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider == "openai" or (
        not provider
        and os.environ.get("OPENAI_API_KEY")
        and not os.environ.get("ANTHROPIC_API_KEY")
    ):
        return "openai", os.environ.get("LLM_MODEL", DEFAULT_OPENAI_MODEL)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        if os.environ.get("OPENAI_API_KEY"):
            return "openai", os.environ.get("LLM_MODEL", DEFAULT_OPENAI_MODEL)
        sys.exit("ERROR: Set ANTHROPIC_API_KEY or OPENAI_API_KEY before running.")
    return "anthropic", os.environ.get("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)


def call_llm(messages: list[dict], provider: str, model: str, system_prompt: str) -> str:
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            system=system_prompt, messages=messages,
        )
        return resp.content[0].text
    elif provider == "openai":
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model=model, max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        return resp.choices[0].message.content
    raise ValueError(f"Unknown provider: {provider}")


# ─── Pipeline validation ──────────────────────────────────────────────────────

_ALLOWED_OPS = {
    "log1p", "sqrt", "square", "cube", "reciprocal", "abs",
    "clip", "rank", "quantile_normal", "bin",
    "ratio", "product", "diff", "sum_pair", "log_ratio",
    "polynomial", "interaction",
    "kmeans_cluster", "kmeans_distance", "distance_to_point",
    "drop", "select", "scale",
}


def extract_pipeline(response: str) -> dict | None:
    m = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
    if not m:
        m = re.search(r"```\s*(\{.*?\})\s*```", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            return None
    m = re.search(r"\{[\s\S]*\"steps\"[\s\S]*\}", response)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def validate_pipeline(config: dict) -> tuple[bool, str]:
    if not isinstance(config, dict):
        return False, "Top-level must be a JSON object"
    if "steps" not in config:
        return False, "Missing 'steps' key"
    steps = config["steps"]
    if not isinstance(steps, list) or not steps:
        return False, "'steps' must be a non-empty list"

    allowed = _ALLOWED_OPS | ({"freestyle"} if ALLOW_FREESTYLE else set())

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return False, f"Step {i} is not an object"
        op = step.get("op")
        if op not in allowed:
            if op == "freestyle":
                return False, (
                    f"Step {i}: 'freestyle' op requires ALLOW_FREESTYLE=true "
                    "(set env var or pass allow_freestyle=True to run())"
                )
            return False, f"Step {i}: unknown op {op!r}"
        if op == "freestyle":
            if "code" not in step:
                return False, f"Step {i}: freestyle step missing 'code' field"
            if "name" not in step:
                return False, f"Step {i}: freestyle step missing 'name' field"
            # Safety check before the code ever touches disk or gets committed
            from operations import check_freestyle_safety
            try:
                check_freestyle_safety(step["code"], step["name"])
            except ValueError as e:
                return False, f"Step {i}: {e}"

    if steps[-1].get("op") != "scale":
        return False, "Last step must be a 'scale' operation"
    if "description" not in config:
        return False, "Missing 'description' key"
    return True, ""


# ─── Scoring ──────────────────────────────────────────────────────────────────


def compute_score(primary: float, n_features: int) -> float:
    """score = primary_metric + COMPLEXITY_ALPHA * ln(n_features)  (always minimised)"""
    return primary + COMPLEXITY_ALPHA * math.log(max(n_features, 1))


# ─── Top-k pool ───────────────────────────────────────────────────────────────


@dataclass
class PoolEntry:
    rank: int
    experiment: int
    val_rmse: float
    n_features: int
    score: float
    description: str
    steps: list[dict]


def load_pool() -> list[PoolEntry]:
    if not POOL_FILE.exists():
        return []
    try:
        raw = json.loads(POOL_FILE.read_text())
        return [PoolEntry(**e) for e in raw]
    except Exception:
        return []


def save_pool(pool: list[PoolEntry]) -> None:
    POOL_FILE.write_text(json.dumps([asdict(e) for e in pool], indent=2) + "\n")


def update_pool(pool: list[PoolEntry], entry: PoolEntry, k: int) -> list[PoolEntry]:
    """Add entry to pool, keep best k by score, re-rank."""
    pool = pool + [entry]
    pool.sort(key=lambda e: e.score)
    pool = pool[:k]
    for i, e in enumerate(pool):
        e.rank = i + 1
    return pool


def format_pool(pool: list[PoolEntry]) -> str:
    rows = [f"rank | score  | val_rmse | n_feat | description"]
    rows.append("---- | ------ | -------- | ------ | -----------")
    for e in pool:
        rows.append(
            f"{e.rank:4d} | {e.score:.4f} | {e.val_rmse:.6f} | {e.n_features:6d} | {e.description}"
        )
    return "\n".join(rows)


# ─── Git helpers ──────────────────────────────────────────────────────────────


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def git_setup_branch() -> str:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = f"autofeature/{tag}"
    git("checkout", "-b", branch)
    return branch


def git_commit(message: str) -> bool:
    git("add", str(PIPELINE_FILE))
    return git("commit", "-m", message, check=False).returncode == 0


def git_revert_last() -> None:
    git("reset", "--hard", "HEAD~1")


# ─── Training ─────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    val_score: float | None    # raw metric value (rmse, auc, or logloss)
    metric_name: str | None    # "rmse" | "auc" | "logloss"
    n_features: int | None
    crashed: bool
    output: str

    def primary(self) -> float | None:
        """Normalised metric where lower is always better."""
        if self.val_score is None:
            return None
        if self.metric_name == "auc":
            return 1.0 - self.val_score
        return self.val_score  # rmse and logloss are already lower-is-better


def run_train() -> RunResult:
    try:
        proc = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True, text=True, timeout=TRAIN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return RunResult(None, None, None, True, "TIMEOUT")

    out = proc.stdout + proc.stderr
    crashed = proc.returncode != 0
    val_score = n_features = None
    metric_name = None

    m = re.search(r"val_score\s*:\s*([0-9.]+)", out)
    if m:
        val_score = float(m.group(1))
    m = re.search(r"metric_name\s*:\s*(\w+)", out)
    if m:
        metric_name = m.group(1)
    m = re.search(r"n_features\s*:\s*([0-9]+)", out)
    if m:
        n_features = int(m.group(1))
    if val_score is None:
        crashed = True

    return RunResult(val_score, metric_name, n_features, crashed, out)


# ─── Results log ──────────────────────────────────────────────────────────────


@dataclass
class Record:
    timestamp: str
    experiment: int
    description: str
    val_score: float | None    # raw metric (rmse / auc / logloss)
    metric_name: str | None
    n_features: int | None
    score: float | None        # composite score (primary + complexity penalty)
    pool_rank: int | None      # rank in pool after this experiment (None if not in pool)
    kept: bool
    crashed: bool


def init_results() -> None:
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "timestamp\texperiment\tdescription\tval_score\tmetric\tn_features\tscore\tpool_rank\tkept\tcrashed\n"
        )


def load_results() -> list[Record]:
    if not RESULTS_FILE.exists():
        return []
    records = []
    for line in RESULTS_FILE.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        try:
            records.append(Record(
                timestamp=parts[0],
                experiment=int(parts[1]),
                description=parts[2],
                val_score=float(parts[3]) if parts[3] not in ("CRASH", "") else None,
                metric_name=parts[4] if parts[4] else None,
                n_features=int(parts[5]) if parts[5] else None,
                score=float(parts[6]) if parts[6] not in ("CRASH", "") else None,
                pool_rank=int(parts[7]) if parts[7] else None,
                kept=parts[8] == "yes",
                crashed=parts[9] == "yes",
            ))
        except (ValueError, IndexError):
            pass
    return records


def append_result(r: Record) -> None:
    score_s = f"{r.val_score:.6f}" if r.val_score is not None else "CRASH"
    comp_s  = f"{r.score:.6f}"     if r.score     is not None else "CRASH"
    rank_s  = str(r.pool_rank)     if r.pool_rank is not None else ""
    with RESULTS_FILE.open("a") as f:
        f.write(
            f"{r.timestamp}\t{r.experiment}\t{r.description}\t"
            f"{score_s}\t{r.metric_name or ''}\t{r.n_features or ''}\t"
            f"{comp_s}\t{rank_s}\t"
            f"{'yes' if r.kept else 'no'}\t{'yes' if r.crashed else 'no'}\n"
        )


# ─── Prompt construction ──────────────────────────────────────────────────────


def format_history(records: list[Record], metric_name: str) -> str:
    if not records:
        return "(no experiments yet)"

    # Apply filter
    visible = [r for r in records if r.kept] if HISTORY_FILTER == "kept" else records

    # Apply size cap (0 = unlimited)
    if HISTORY_SIZE > 0:
        visible = visible[-HISTORY_SIZE:]

    if not visible:
        return "(no matching experiments yet)"

    col = metric_name or "metric"
    filter_note = " (pool entries only)" if HISTORY_FILTER == "kept" else ""
    rows = [f"#exp | score  | {col:9s} | n_feat | pool | description{filter_note}"]
    rows.append(f"---- | ------ | {'-'*9} | ------ | ---- | -----------")
    for r in visible:
        raw_s   = f"{r.val_score:.6f}" if r.val_score is not None else "CRASH   "
        score_s = f"{r.score:.4f}"     if r.score     is not None else "CRASH "
        feat_s  = str(r.n_features)    if r.n_features else "?"
        rank_s  = f"#{r.pool_rank}"    if r.pool_rank  else "  -"
        rows.append(f"{r.experiment:4d} | {score_s} | {raw_s} | {feat_s:6s} | {rank_s:4s} | {r.description}")
    return "\n".join(rows)


# ─── Data profiler ───────────────────────────────────────────────────────────

_PROFILE_SYSTEM = """\
You are a data scientist specialising in feature engineering for tabular ML.

Given statistical summaries of a dataset, produce CONCISE, actionable insights
for a feature engineering agent.  Cover:
1. Which features are skewed and would benefit from log / sqrt / clip transforms.
2. Which feature pairs are likely to form useful ratios, differences, or products.
3. Which features correlate most strongly with the target.
4. Any high inter-feature correlations that suggest redundancy or interaction potential.
5. Any other dataset-specific quirks that should guide feature construction.

Be specific — use the actual feature names.  Keep the response under 500 words.
Use bullet points.  Do NOT suggest model changes; focus only on the features.
"""


def _compute_raw_stats(task_cfg: dict, feature_names: list[str]) -> str:
    """Return a statistical summary of the raw training data as a formatted string."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from prepare import _load_dataset

    df, y = _load_dataset(task_cfg["dataset"])
    idx = np.arange(len(df))
    idx_train, _ = train_test_split(idx, test_size=0.2, random_state=42)
    df_tr = df.iloc[idx_train].reset_index(drop=True)
    y_tr  = y[idx_train]

    lines = [
        f"DATASET  : {task_cfg['dataset']}",
        f"TASK     : {task_cfg['task']}",
        f"METRIC   : {task_cfg['metric']}",
        f"SAMPLES  : {len(df)} total  ({len(df_tr)} train  {len(df)-len(df_tr)} val)",
        f"FEATURES : {len(feature_names)}",
        "",
        f"{'Feature':<22} {'mean':>9} {'std':>9} {'skew':>6} {'min':>9} {'p25':>9} {'p50':>9} {'p75':>9} {'max':>9}",
        "-" * 100,
    ]
    for col in feature_names:
        v = df_tr[col].values
        skew = float(df_tr[col].skew())
        p25, p50, p75 = float(df_tr[col].quantile(0.25)), float(df_tr[col].median()), float(df_tr[col].quantile(0.75))
        lines.append(
            f"{col:<22} {v.mean():>9.3f} {v.std():>9.3f} {skew:>6.2f}"
            f" {v.min():>9.3f} {p25:>9.3f} {p50:>9.3f} {p75:>9.3f} {v.max():>9.3f}"
        )

    lines += ["", "CORRELATION WITH TARGET (Pearson r):"]
    corrs = sorted(
        [(col, float(df_tr[col].corr(import_series(y_tr, df_tr)))) for col in feature_names],
        key=lambda x: abs(x[1]), reverse=True,
    )
    for col, r in corrs:
        lines.append(f"  {col:<22}: {r:+.4f}")

    lines += ["", "INTER-FEATURE CORRELATIONS (|r| ≥ 0.4):"]
    corr_mat = df_tr.corr()
    pairs = [
        (a, b, float(corr_mat.loc[a, b]))
        for i, a in enumerate(feature_names)
        for b in feature_names[i + 1:]
        if abs(corr_mat.loc[a, b]) >= 0.4
    ]
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if pairs:
        for a, b, r in pairs:
            lines.append(f"  {a} ↔ {b}: {r:+.4f}")
    else:
        lines.append("  (none above threshold)")

    if task_cfg["task"] == "regression":
        lines += ["", f"TARGET: mean={y_tr.mean():.4f}  std={y_tr.std():.4f}  skew={float(import_series(y_tr, df_tr).skew()):.2f}  min={y_tr.min():.4f}  max={y_tr.max():.4f}"]
    else:
        import numpy as np
        classes, counts = np.unique(y_tr, return_counts=True)
        lines += ["", "TARGET CLASS DISTRIBUTION:"]
        for cls, cnt in zip(classes, counts):
            lines.append(f"  class {cls}: {cnt} ({100*cnt/len(y_tr):.1f}%)")

    return "\n".join(lines)


def import_series(arr, ref_df):
    """Wrap a numpy array as a pandas Series aligned with ref_df's index."""
    import pandas as pd
    return pd.Series(arr, index=ref_df.index)


def load_or_generate_profile(
    task_cfg: dict,
    feature_names: list[str],
    provider: str,
    model: str,
) -> str | None:
    """Return the data profile string, generating and caching it if needed."""
    if not INCLUDE_DATA_PROFILE:
        return None

    profile_path = Path(f"data_profile_{task_cfg['dataset']}.md")

    if profile_path.exists() and not REGEN_PROFILE:
        print(f"Data profile     : loaded from {profile_path}")
        return profile_path.read_text()

    print("Generating data profile (one-time LLM call)...")
    stats = _compute_raw_stats(task_cfg, feature_names)

    messages = [{"role": "user", "content": f"Analyse this dataset for feature engineering:\n\n{stats}"}]
    insights = call_llm(messages, provider, model, _PROFILE_SYSTEM)

    # Cache: store both stats and insights so the file is human-readable
    profile_path.write_text(
        f"# Data Profile: {task_cfg['dataset']}\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"## Raw Statistics\n```\n{stats}\n```\n\n"
        f"## LLM Insights\n{insights}\n"
    )
    print(f"Data profile     : saved to {profile_path}")
    return insights   # only the insights go into the prompt


# ─── Prompt construction ──────────────────────────────────────────────────────

def build_messages(
    pool: list[PoolEntry],
    base: PoolEntry,
    records: list[Record],
    task_cfg: dict,
    data_profile: str | None = None,
) -> list[dict]:
    metric_name = task_cfg["metric"]
    base_json   = json.dumps({"description": base.description, "steps": base.steps}, indent=2)

    profile_section = (
        f"\nDATA INSIGHTS (one-time analysis of the raw dataset):\n{data_profile}\n"
        if data_profile else ""
    )
    history_section = (
        f"\nExperiment history (most recent last):\n{format_history(records, metric_name)}\n"
        if INCLUDE_HISTORY else ""
    )

    content = f"""\
Top-{TOPK} pipeline pool (rank 1 = best score):
{format_pool(pool)}

Base pipeline for this iteration: rank {base.rank} — "{base.description}"
```json
{base_json}
```
{profile_section}{history_section}
Suggest ONE focused improvement to the BASE PIPELINE ABOVE and return the complete updated pipeline.
"""
    return [{"role": "user", "content": content}]


# ─── Main loop ────────────────────────────────────────────────────────────────


def main(n_iterations: int | None = None) -> None:
    provider, model = detect_provider()

    # Load task config and build prompt once (it's constant for the session)
    task_cfg = json.loads(TASK_FILE.read_text())
    from prepare import get_feature_names
    feature_names = get_feature_names()
    system_prompt = _build_system_prompt(task_cfg, feature_names)
    metric_name   = task_cfg["metric"]

    print(f"Provider         : {provider}")
    print(f"Model            : {model}")
    print(f"Task             : {task_cfg['task']} / {task_cfg['dataset']}")
    print(f"Metric           : {metric_name}")
    print(f"Top-k            : {TOPK}")
    print(f"Complexity alpha : {COMPLEXITY_ALPHA}")
    history_desc = (
        "off" if not INCLUDE_HISTORY
        else f"{HISTORY_FILTER}, last {'∞' if HISTORY_SIZE == 0 else HISTORY_SIZE}"
    )
    print(f"History          : {history_desc}")
    print(f"Data profile     : {'enabled' if INCLUDE_DATA_PROFILE else 'disabled'}")
    print(f"Freestyle ops    : {'ENABLED ⚠️  (LLM code will be exec-d)' if ALLOW_FREESTYLE else 'disabled'}")
    if ALLOW_FREESTYLE:
        print("  ⚠️  WARNING: freestyle mode executes LLM-generated Python code.")
        print("     Safety checks are applied but cannot guarantee full sandboxing.")
        print("     Only enable this on trusted, non-production machines.")
    print()

    branch       = git_setup_branch()
    print(f"Branch           : {branch}")
    data_profile = load_or_generate_profile(task_cfg, feature_names, provider, model)

    init_results()
    records  = load_results()
    pool     = load_pool()
    exp_n    = len(records) + 1
    pool_idx = 0   # round-robin cursor

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("─" * 60)
    print("Measuring baseline...")
    baseline = run_train()
    if baseline.crashed:
        print("Baseline CRASHED:")
        print(baseline.output[-800:])
        sys.exit("Fix pipeline.json / task.json before running the agent.")

    baseline_steps   = json.loads(PIPELINE_FILE.read_text())["steps"]
    baseline_primary = baseline.primary()
    baseline_score   = compute_score(baseline_primary, baseline.n_features)
    print(
        f"  {metric_name}={baseline.val_score:.6f}"
        f"  n_features={baseline.n_features}"
        f"  score={baseline_score:.4f}"
    )

    baseline_entry = PoolEntry(
        rank=1, experiment=exp_n,
        val_rmse=baseline.val_score, n_features=baseline.n_features,
        score=baseline_score, description="baseline", steps=baseline_steps,
    )
    pool = update_pool(pool, baseline_entry, TOPK)
    save_pool(pool)

    append_result(Record(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        experiment=exp_n, description="baseline",
        val_score=baseline.val_score, metric_name=metric_name,
        n_features=baseline.n_features, score=baseline_score,
        pool_rank=baseline_entry.rank, kept=True, crashed=False,
    ))
    records = load_results()
    exp_n += 1

    # ── Agent loop ────────────────────────────────────────────────────────────
    iteration = 0
    while n_iterations is None or iteration < n_iterations:
        iteration += 1
        base = pool[pool_idx % len(pool)]
        pool_idx += 1

        print()
        print(f"{'─'*60}")
        print(f"Experiment #{exp_n}  |  best score={pool[0].score:.4f}  |  base=rank{base.rank} ({base.description})")

        # Stage base pipeline so the LLM sees it as the starting point
        base_config = {"description": base.description, "steps": base.steps}
        PIPELINE_FILE.write_text(json.dumps(base_config, indent=2) + "\n")

        messages = build_messages(pool, base, records, task_cfg, data_profile)

        # ── LLM call ──────────────────────────────────────────────────────────
        new_config = None
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            print(f"  Calling {provider}/{model} (attempt {attempt})...")
            try:
                response = call_llm(messages, provider, model, system_prompt)
            except Exception as e:
                print(f"  API error: {e}")
                time.sleep(5)
                continue

            new_config = extract_pipeline(response)
            if new_config is None:
                print("  No JSON found; retrying...")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Return the pipeline inside a ```json ... ``` block."})
                continue

            ok, err = validate_pipeline(new_config)
            if not ok:
                print(f"  Invalid pipeline: {err}")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Validation failed: {err}. Fix and return the complete pipeline."})
                new_config = None
                continue

            break

        if new_config is None:
            print(f"  No valid pipeline after {MAX_LLM_RETRIES} attempts; skipping.")
            exp_n += 1
            continue

        description = new_config.get("description", "(no description)")
        print(f"  Change : {description}")
        print(f"  Steps  : {len(new_config['steps'])}")

        # ── Commit & evaluate ─────────────────────────────────────────────────
        new_json = json.dumps(new_config, indent=2)
        PIPELINE_FILE.write_text(new_json + "\n")
        committed = git_commit(f"experiment #{exp_n}: {description}")
        if not committed:
            print("  Nothing changed vs HEAD; skipping.")
            exp_n += 1
            continue

        print("  Running train.py...")
        result = run_train()

        if result.crashed:
            print(f"  CRASHED:\n{result.output[-500:]}")
            git_revert_last()
            append_result(Record(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                experiment=exp_n, description=description,
                val_score=None, metric_name=metric_name,
                n_features=None, score=None,
                pool_rank=None, kept=False, crashed=True,
            ))
            records = load_results()
            exp_n += 1
            continue

        # ── Score & pool update ───────────────────────────────────────────────
        primary     = result.primary()
        score       = compute_score(primary, result.n_features)
        worst_score = pool[-1].score if len(pool) >= TOPK else float("inf")
        enters_pool = score < worst_score or len(pool) < TOPK
        delta       = score - pool[0].score
        sign        = "✓" if enters_pool else "✗"

        print(
            f"  {sign} {metric_name}={result.val_score:.6f}"
            f"  n_features={result.n_features}"
            f"  score={score:.4f}  Δ{delta:+.4f}"
            + ("  → enters pool" if enters_pool else "")
        )

        if enters_pool:
            new_entry = PoolEntry(
                rank=0, experiment=exp_n,
                val_rmse=result.val_score, n_features=result.n_features,
                score=score, description=description,
                steps=new_config["steps"],
            )
            pool = update_pool(pool, new_entry, TOPK)
            save_pool(pool)
            pool_rank = new_entry.rank
            print(f"  Pool: {' | '.join(f'#{e.rank} {e.score:.4f}' for e in pool)}")
        else:
            git_revert_last()
            pool_rank = None

        append_result(Record(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            experiment=exp_n, description=description,
            val_score=result.val_score, metric_name=metric_name,
            n_features=result.n_features, score=score,
            pool_rank=pool_rank, kept=enters_pool, crashed=False,
        ))
        records = load_results()
        exp_n += 1


def run(
    n_iterations: int | None = None,
    *,
    topk: int | None = None,
    complexity_alpha: float | None = None,
    include_history: bool | None = None,
    history_size: int | None = None,
    history_filter: str | None = None,
    include_data_profile: bool | None = None,
    regen_profile: bool | None = None,
    train_timeout: int | None = None,
    allow_freestyle: bool | None = None,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    working_dir: str | None = None,
) -> None:
    """Run the agent loop from Python code.

    Example
    -------
    >>> from agent import run
    >>> run(n_iterations=10, anthropic_api_key="sk-ant-...", topk=5)

    Parameters
    ----------
    n_iterations    : Number of agent iterations to run (None = infinite).
    topk            : Size of the top-k pipeline pool (default 5).
    complexity_alpha: Weight of the feature-count penalty in the score (default 0.005).
    include_history : Whether to include experiment history in each prompt.
    history_size    : Max experiments to show in history (0 = all).
    history_filter  : "all" (default) or "kept" (only pool entries).
    include_data_profile: Whether to generate/use the one-time data profile.
    regen_profile   : Force regeneration of the data profile even if cached.
    train_timeout   : Seconds before a training run is killed (default 120).
    allow_freestyle : Allow the LLM to propose short Python snippets as ops.
                      Disabled by default. Code is safety-checked before exec.
                      Only enable on trusted machines — see operations.py for guardrails.
    anthropic_api_key: Anthropic API key (overrides ANTHROPIC_API_KEY env var).
    openai_api_key  : OpenAI API key (overrides OPENAI_API_KEY env var).
    llm_provider    : "anthropic" or "openai" (auto-detected if not set).
    llm_model       : LLM model name override.
    working_dir     : Path to the repo directory (defaults to current directory).
    """
    global TOPK, COMPLEXITY_ALPHA, INCLUDE_HISTORY, HISTORY_SIZE, HISTORY_FILTER
    global INCLUDE_DATA_PROFILE, REGEN_PROFILE, TRAIN_TIMEOUT, ALLOW_FREESTYLE
    global TASK_FILE, PIPELINE_FILE, POOL_FILE, RESULTS_FILE

    if working_dir is not None:
        import os as _os
        _os.chdir(working_dir)
        TASK_FILE     = Path("task.json")
        PIPELINE_FILE = Path("pipeline.json")
        POOL_FILE     = Path("topk_pool.json")
        RESULTS_FILE  = Path("results.tsv")

    if topk             is not None: TOPK              = topk
    if complexity_alpha is not None: COMPLEXITY_ALPHA  = complexity_alpha
    if include_history  is not None: INCLUDE_HISTORY   = include_history
    if history_size     is not None: HISTORY_SIZE      = history_size
    if history_filter   is not None: HISTORY_FILTER    = history_filter
    if include_data_profile is not None: INCLUDE_DATA_PROFILE = include_data_profile
    if regen_profile    is not None: REGEN_PROFILE     = regen_profile
    if train_timeout    is not None: TRAIN_TIMEOUT     = train_timeout
    if allow_freestyle  is not None: ALLOW_FREESTYLE   = allow_freestyle

    if anthropic_api_key: os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if openai_api_key:    os.environ["OPENAI_API_KEY"]    = openai_api_key
    if llm_provider:      os.environ["LLM_PROVIDER"]      = llm_provider
    if llm_model:         os.environ["LLM_MODEL"]         = llm_model

    main(n_iterations)


if __name__ == "__main__":
    main()
