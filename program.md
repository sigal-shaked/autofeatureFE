# AutoFeature — Autonomous Feature Engineering Optimizer

## What this is

An autonomous agent loop that iteratively improves a **feature engineering
pipeline** for a tabular regression task using an LLM API.

The agent may only compose operations from a **fixed, closed library** —
no arbitrary code is generated or executed.  Every experiment is fully
auditable as a JSON file.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `pipeline.json` | **Editable** (by agent) | Feature engineering pipeline spec |
| `operations.py` | **Fixed** | Library of all allowed operations |
| `prepare.py` | **Fixed** | Loads pipeline.json, applies ops, returns arrays |
| `train.py` | **Fixed** | XGBoost training + val_rmse evaluation |
| `agent.py` | Entry point | LLM-powered agent loop |
| `results.tsv` | Auto-generated | Experiment log |

## How it works

```
SETUP:
  1. Create git branch  autofeature/<timestamp>
  2. Measure baseline val_rmse on unmodified pipeline.json
  3. Record baseline in results.tsv

LOOP (runs until Ctrl-C):
  1. Read current pipeline.json + recent experiment history
  2. Call LLM → receive improved pipeline.json
  3. Validate: all ops in allowed set, last step is "scale", valid JSON
  4. git commit pipeline.json
  5. Run train.py → val_rmse
  6. If improved → keep commit
     If not       → git reset --hard HEAD~1
  7. Record result in results.tsv
  8. Go to 1
```

## Dataset & task

- **Dataset**: California Housing (sklearn built-in)
- **Task**: Regression — predict median house value ($100Ks)
- **Metric**: `val_rmse` (lower is better)
- **Split**: fixed 80/20 train/val, seed=42 (the agent cannot change this)

Raw features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`,
`AveOccup`, `Latitude`, `Longitude`.

## Allowed operations (closed set)

| Category | Operations |
|----------|-----------|
| Unary transforms | `log1p`, `sqrt`, `square`, `cube`, `reciprocal`, `abs` |
| Stateful unary | `clip`, `rank`, `quantile_normal`, `bin` |
| Binary → new feature | `ratio`, `product`, `diff`, `sum_pair`, `log_ratio` |
| Multi-feature | `polynomial`, `interaction` |
| Geographic | `kmeans_cluster`, `kmeans_distance`, `distance_to_point` |
| Selection | `drop`, `select` |
| Scaling | `scale` (standard / robust / minmax / quantile) |

Operations can be composed freely in any order.  Features created by one
step can be referenced by any later step.

## Example pipeline.json

```json
{
  "description": "geo clusters + income log + rooms ratio",
  "steps": [
    {"op": "log1p", "features": ["MedInc", "Population"]},
    {"op": "ratio", "numerator": "AveRooms", "denominator": "AveBedrms", "name": "rooms_per_bedrm"},
    {"op": "ratio", "numerator": "Population", "denominator": "AveOccup", "name": "households"},
    {"op": "kmeans_cluster", "features": ["Latitude", "Longitude"], "n_clusters": 15, "name": "geo_cluster"},
    {"op": "distance_to_point", "lat": "Latitude", "lon": "Longitude",
        "target_lat": 37.77, "target_lon": -122.42, "name": "dist_sf"},
    {"op": "scale", "method": "robust"}
  ]
}
```

## Running the agent

```bash
uv sync

# With Anthropic Claude (default):
ANTHROPIC_API_KEY=sk-... uv run agent.py

# With OpenAI GPT-4:
OPENAI_API_KEY=sk-... LLM_PROVIDER=openai uv run agent.py

# Override model:
ANTHROPIC_API_KEY=sk-... LLM_MODEL=claude-opus-4-6 uv run agent.py
```

## Running just the training

```bash
uv run train.py   # prints val_rmse and n_features
```
