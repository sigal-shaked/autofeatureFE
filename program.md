# AutoFeature — Autonomous Feature Engineering Optimizer

## What this is

An autonomous agent loop that iteratively improves the feature engineering
pipeline (`prepare.py`) for a tabular regression task using an LLM API.

Unlike its sister project (autoresearchFE) which optimizes a neural network
architecture, this project targets the **data** side of the ML pipeline:

- The model (XGBoost) and all its hyperparameters are **fixed**.
- The agent's only lever is `prepare.py` — feature engineering, preprocessing,
  feature selection.
- No GPU required.
- The internal reasoning engine is an external LLM (Claude / GPT-4).

## Files

| File | Status | Purpose |
|------|--------|---------|
| `prepare.py` | **Editable** (by agent) | Feature engineering & preprocessing pipeline |
| `train.py` | **Fixed** (do not edit) | XGBoost training + val_rmse evaluation |
| `agent.py` | Entry point | LLM-powered agent loop |
| `results.tsv` | Auto-generated | Experiment log |

## How it works

```
SETUP:
  1. Create git branch  autofeature/<timestamp>
  2. Measure baseline val_rmse on unmodified prepare.py
  3. Record baseline in results.tsv

LOOP (runs until you Ctrl-C):
  1. Read current prepare.py + recent experiment history
  2. Call LLM with context → receive improved prepare.py
  3. Validate the code (syntax + API contract check)
  4. git commit the change
  5. Run train.py  →  val_rmse
  6. If improved  → keep commit, record result
     If not improved → git reset, record result
  7. Go to 1
```

## Dataset & task

- **Dataset**: California Housing (sklearn built-in)
- **Task**: Regression — predict median house value ($100Ks)
- **Metric**: `val_rmse` (lower is better)
- **Split**: fixed 80/20 train/val, seed=42

Raw features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`,
`AveOccup`, `Latitude`, `Longitude`.

## Running the agent

```bash
# Install dependencies
uv sync

# With Anthropic Claude (default):
ANTHROPIC_API_KEY=sk-... uv run agent.py

# With OpenAI GPT-4:
OPENAI_API_KEY=sk-... LLM_PROVIDER=openai uv run agent.py

# Override model:
ANTHROPIC_API_KEY=sk-... LLM_MODEL=claude-opus-4-6 uv run agent.py
```

The agent runs **indefinitely** until you stop it (Ctrl-C).
Check `results.tsv` for the experiment log at any time.

## Running just the training

```bash
uv run train.py
```

Prints `val_rmse: X.XXXXXX` and `n_features: N`.

## What the agent can do to prepare.py

- Add new engineered features (ratios, logs, polynomial terms, interactions)
- Geographic features (distance to city centre, latitude × longitude)
- Cluster-based features (k-means on lat/lon → location bucket)
- Change scaling / normalization
- Select / drop features
- Any transformation using numpy, pandas, or scikit-learn

## Contract: prepare_data()

The agent must keep the function signature intact:

```python
def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, X_val, y_train, y_val) as float32 numpy arrays."""
```

RANDOM_SEED = 42 and VAL_SIZE = 0.2 must not change.
