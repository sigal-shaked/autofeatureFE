# AutoFeatureFE

Autonomous feature-engineering optimizer.  An LLM agent iteratively proposes
feature pipelines; a fixed XGBoost model evaluates them.  Only the features
change — the model and its hyperparameters never do.

---

## Architecture

```
task.json          ← human configures task / dataset / metric (never touched by agent)
pipeline.json      ← agent edits this (JSON, not code)
    ↓
prepare.py         ← FIXED: loads data, applies pipeline, returns X/y arrays
operations.py      ← FIXED: closed library of ~40 allowed feature ops
train.py           ← FIXED: trains XGBoost, prints val_score / metric_name / n_features
    ↓
results.tsv        ← append-only experiment log (all runs, human-readable)
topk_pool.json     ← top-k best pipelines (JSON, survives restarts)
data_profile_*.md  ← cached one-time LLM analysis of the raw dataset
```

The agent never generates or executes arbitrary Python.  It only writes JSON
to `pipeline.json`.  `prepare.py` and `operations.py` are the only code that
runs on data, and they are fixed.

---

## Files

| File | Role | Editable? |
|---|---|---|
| `task.json` | Task config (dataset, metric, task type) | Human only |
| `pipeline.json` | Current feature pipeline | Agent writes |
| `prepare.py` | Data loading + pipeline execution | Fixed |
| `operations.py` | Feature op library | Fixed |
| `train.py` | Model training + evaluation | Fixed |
| `agent.py` | LLM agent loop | Fixed |
| `results.tsv` | Experiment log | Append-only |
| `topk_pool.json` | Top-k pipeline pool | Agent writes |

---

## Configuration (task.json)

```json
{
  "task":    "regression",            // or "classification"
  "dataset": "california_housing",   // see supported datasets below
  "metric":  "rmse"                  // rmse | auc | logloss
}
```

**Supported datasets (built-in):**
- `california_housing` — regression, 8 features
- `breast_cancer`      — binary classification, 30 features
- `wine`               — multiclass classification, 13 features

**Custom CSV dataset:**
```json
{
  "task":          "classification",
  "dataset":       "csv",
  "csv_path":      "mydata.csv",
  "target_column": "label",
  "metric":        "auc"
}
```

---

## Pipeline format (pipeline.json)

```json
{
  "description": "log fare + family size ratio + geo clusters",
  "steps": [
    {"op": "log1p",         "features": ["fare"]},
    {"op": "ratio",         "numerator": "fare", "denominator": "family_size", "name": "fare_per_person"},
    {"op": "kmeans_cluster","features": ["lat", "lon"], "n_clusters": 8, "name": "geo_cluster"},
    {"op": "scale",         "method": "standard"}
  ]
}
```

Rules enforced by `validate_pipeline()`:
- All ops must be from the allowed set in `operations.py`
- Last step must always be `"scale"`
- Must include `"description"`

---

## Agent loop

1. **Baseline** — evaluate the current `pipeline.json`, add to pool
2. **Round-robin** — pick a base pipeline from the top-k pool
3. **LLM call** — ask LLM to improve the base pipeline; up to 3 retries on invalid JSON
4. **Validate** — check all ops are allowed, last step is scale, no blocked patterns (freestyle)
5. **Evaluate** — write candidate to `pipeline.json`, run `train.py`
6. **Score** — `score = primary_metric + COMPLEXITY_ALPHA × ln(n_features)`
   - regression: `primary = val_rmse`
   - classification AUC: `primary = 1 − val_auc`
   - classification logloss: `primary = val_logloss`
7. **Pool update** — if score enters top-k: keep; else restore base pipeline
8. **Log** — append row to `results.tsv`

---

## Allowed operations

### Unary transforms (in-place)
`log1p`, `sqrt`, `square`, `cube`, `reciprocal`, `abs`

### Stateful unary (fit on train, apply to val)
`clip`, `rank`, `quantile_normal`, `bin`

### Binary → new column
`ratio`, `product`, `diff`, `sum_pair`, `log_ratio`

### Multi-feature → new columns
`polynomial`, `interaction`

### Geographic
`kmeans_cluster`, `kmeans_distance`, `distance_to_point`

### Selection
`drop`, `select`

### Scaling (always last)
`scale` — methods: `standard`, `robust`, `minmax`, `quantile`

### Freestyle (opt-in, disabled by default)
Short Python snippets (`df['x'] = ...`).  Guarded by a pattern blocklist,
restricted builtins, and post-exec row/dtype checks.
Enable with `ALLOW_FREESTYLE=true` or `allow_freestyle=True` in `run()`.

---

## Running

**Terminal:**
```bash
# Anthropic
ANTHROPIC_API_KEY=sk-ant-... uv run agent.py

# OpenAI
OPENAI_API_KEY=sk-... LLM_PROVIDER=openai uv run agent.py

# Key options
TOPK=5                  # pool size (default 5)
COMPLEXITY_ALPHA=0.005  # feature-count penalty (default 0.005)
INCLUDE_HISTORY=true    # include past experiments in prompt (default true)
HISTORY_SIZE=20         # max history rows shown to LLM (0 = all)
HISTORY_FILTER=all      # "all" or "kept" (pool entries only)
DATA_PROFILE=true       # one-time LLM dataset analysis (default true)
ALLOW_FREESTYLE=false   # allow LLM Python snippets (default false)
```

**Python API:**
```python
from agent import run

run(
    n_iterations         = 20,
    anthropic_api_key    = "sk-ant-...",
    topk                 = 5,
    complexity_alpha     = 0.005,
    include_history      = True,
    history_size         = 20,
    include_data_profile = True,
    allow_freestyle      = False,
    working_dir          = "/path/to/repo",   # if running from elsewhere
)
```

---

## Experiment log (results.tsv)

One row per experiment.  Columns:

| Column | Description |
|---|---|
| timestamp | ISO-8601 |
| experiment | Sequential integer |
| description | LLM-written description of the change |
| val_score | Raw metric value (rmse / auc / logloss) |
| metric | Metric name |
| n_features | Feature count after pipeline |
| score | Composite score (primary + complexity penalty) |
| pool_rank | Rank in top-k pool (blank if not in pool) |
| kept | yes / no |
| crashed | yes / no |

View with: `column -t -s $'\t' results.tsv`
