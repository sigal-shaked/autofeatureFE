# AutoFeatureFE

An autonomous agent that iteratively discovers better feature engineering pipelines for tabular ML — using an LLM (Claude / GPT-4) as the reasoning engine and XGBoost as the fixed evaluator.

**No GPU required.** The model and its hyperparameters never change. Only the feature pipeline improves.

---

## How it works

```
┌─────────────────────────────────────────────────────────┐
│                      Agent loop                         │
│                                                         │
│  1. Pick a base pipeline from the top-k pool            │
│  2. Ask LLM to propose one improvement                  │
│  3. Validate JSON (all ops must be from closed set)     │
│  4. Auto-encode raw data → apply pipeline → train XGB   │
│  5. Score = primary_metric + α × ln(n_features)         │
│  6. If score enters top-k pool → keep; else restore     │
│  7. Log to results.tsv → repeat                         │
└─────────────────────────────────────────────────────────┘
```

The LLM never writes arbitrary code. It selects and composes operations from a **fixed library of ~40 feature ops** expressed as JSON. This makes every experiment fully auditable, reproducible, and safe to run.

---

## Features

- **Closed operation set** — the LLM picks from log transforms, ratios, polynomial interactions, k-means clusters, binning, scaling, and more. Compositions of arbitrary depth are allowed.
- **Auto preprocessing** — ID columns dropped, datetimes decomposed, categoricals ordinal-encoded automatically before any pipeline runs. Field types inferred from OpenML metadata + heuristics, then optionally refined by LLM.
- **Top-k pool** — maintains the k best pipelines found so far, exploring from all of them via round-robin to avoid local optima.
- **Composite scoring** — `score = primary_metric + α × ln(n_features)` penalises feature bloat.
- **Regression & classification** — RMSE, AUC, or log-loss; XGBoost regressor or classifier selected automatically.
- **Any tabular dataset** — sklearn built-ins, CSV files, or any [OpenML](https://openml.org) dataset by ID.  Ships with support for the full [OpenML-CC18](https://www.openml.org/s/99) (72 classification tasks) and [OpenML-CTR23](https://www.openml.org/s/353) (35 regression tasks) benchmark suites.
- **Persistent history** — `results.tsv` and `topk_pool.json` survive restarts; the agent resumes automatically from the pool.
- **Python API** — call `run()` from a notebook or script with no terminal required.
- **Optional freestyle ops** — off by default; when enabled, the LLM can propose short Python snippets guarded by a blocklist, restricted builtins, and post-exec validation.

---

## Quick start

```bash
git clone https://github.com/sigal-shaked/autofeatureFE
cd autofeatureFE
uv sync

# Anthropic Claude (default)
ANTHROPIC_API_KEY=sk-ant-... uv run agent.py

# OpenAI GPT-4
OPENAI_API_KEY=sk-... LLM_PROVIDER=openai uv run agent.py
```

Default task: California Housing regression. Change `task.json` to switch datasets.

---

## Dataset configuration (`task.json`)

```jsonc
// sklearn built-in (regression)
{ "task": "regression", "dataset": "california_housing", "metric": "rmse" }

// sklearn built-in (classification)
{ "task": "classification", "dataset": "breast_cancer", "metric": "auc" }

// any CSV file
{ "task": "classification", "dataset": "csv",
  "csv_path": "mydata.csv", "target_column": "label", "metric": "auc" }

// any OpenML dataset by ID
{ "task": "classification", "dataset": "openml", "openml_id": 31, "metric": "auc" }

// OpenML benchmark suite by index
{ "task": "regression", "dataset": "openml",
  "openml_suite": "OpenML-CTR23", "openml_suite_index": 0, "metric": "rmse" }
```

---

## Key options

| Env var | Default | Description |
|---|---|---|
| `MAX_ITER` | `0` (∞) | Stop after N iterations |
| `TOPK` | `5` | Pool size |
| `COMPLEXITY_ALPHA` | `0.005` | Feature-count penalty weight |
| `INCLUDE_HISTORY` | `true` | Show past experiments in each prompt |
| `HISTORY_SIZE` | `20` | Max history rows shown (`0` = all) |
| `DATA_PROFILE` | `true` | One-time LLM analysis of the raw dataset |
| `ALLOW_FREESTYLE` | `false` | Allow LLM Python snippets as ops |

---

## Python API

```python
from agent import run

run(
    n_iterations         = 20,
    anthropic_api_key    = "sk-ant-...",
    topk                 = 5,
    complexity_alpha     = 0.005,
    include_data_profile = True,
    working_dir          = "/path/to/repo",
)
```

---

## Allowed operations

| Category | Operations |
|---|---|
| Unary transforms | `log1p` `sqrt` `square` `cube` `reciprocal` `abs` |
| Stateful unary | `clip` `rank` `quantile_normal` `bin` |
| Preprocessing | `fillna` `drop_missing_cols` |
| Binary → new column | `ratio` `product` `diff` `sum_pair` `log_ratio` |
| Multi-feature | `polynomial` `interaction` |
| Geographic | `kmeans_cluster` `kmeans_distance` `distance_to_point` |
| Selection | `drop` `select` |
| Scaling (always last) | `scale` — `standard` `robust` `minmax` `quantile` |
| Freestyle (opt-in) | `freestyle` — short Python snippet, safety-checked |

Example pipeline:
```json
{
  "description": "log income + geo clusters + ratio features",
  "steps": [
    { "op": "log1p",          "features": ["MedInc", "Population"] },
    { "op": "ratio",          "numerator": "AveRooms", "denominator": "AveBedrms", "name": "rooms_per_bedrm" },
    { "op": "kmeans_cluster", "features": ["Latitude", "Longitude"], "n_clusters": 10, "name": "geo_cluster" },
    { "op": "scale",          "method": "robust" }
  ]
}
```

---

## Files

| File | Role |
|---|---|
| `agent.py` | Agent loop, LLM calls, pool management, scoring |
| `operations.py` | Fixed feature op library (do not modify) |
| `prepare.py` | Data loading, field-type auto-encoding, pipeline execution (do not modify) |
| `field_types.py` | Field type inference + auto-encoding (do not modify) |
| `train.py` | XGBoost training + evaluation (do not modify) |
| `task.json` | **Human-configured** — dataset, task, metric |
| `pipeline.json` | **Agent-written** — current feature pipeline |
| `results.tsv` | Append-only experiment log |
| `topk_pool.json` | Top-k best pipelines (persists across restarts) |
| `field_types_<id>.json` | Cached + human-editable field type overrides |
| `data_profile_<id>.md` | Cached LLM dataset analysis |

---

## Notebooks

| Notebook | Description |
|---|---|
| [`demo.ipynb`](demo.ipynb) | Single-dataset walkthrough — Titanic by default, any dataset by changing `TASK_CFG` |
| [`benchmark_openml.ipynb`](benchmark_openml.ipynb) | Batch benchmark across 15 CC18 + 15 CTR23 datasets with result aggregation and plots |

---

## Experiment log

Every run appends a row to `results.tsv`:

```
timestamp  experiment  description  val_score  metric_name  n_features  score  pool_rank  kept  crashed
```

View it with:
```bash
column -t -s $'\t' results.tsv
```

---

## Inspired by

[`autoresearch`](https://github.com/karpathy/autoresearch) by Andrej Karpathy — an agent-loop approach to model architecture search for tabular ML.  This repo shifts the optimization target from the model to the feature pipeline, enabling CPU-only runs on any tabular dataset.
