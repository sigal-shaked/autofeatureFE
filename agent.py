#!/usr/bin/env python3
"""
AutoFeature Agent

Autonomous agent that iteratively improves pipeline.json using an LLM API.
Each iteration: read current pipeline → ask LLM for improvement →
validate → run train.py → keep if val_rmse improved, revert otherwise.

The LLM may only compose operations from the fixed library in operations.py.
No arbitrary code is generated or executed.

Usage:
    ANTHROPIC_API_KEY=sk-... uv run agent.py
    OPENAI_API_KEY=sk-...   LLM_PROVIDER=openai uv run agent.py
    LLM_MODEL=claude-opus-4-6 ANTHROPIC_API_KEY=sk-... uv run agent.py
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o"
MAX_TOKENS = 4096
HISTORY_SIZE = 12
MAX_LLM_RETRIES = 3
RESULTS_FILE = Path("results.tsv")
PIPELINE_FILE = Path("pipeline.json")
TRAIN_TIMEOUT = 120  # seconds

# ─── Operation schema (shown to LLM) ─────────────────────────────────────────

OPERATIONS_REFERENCE = """\
AVAILABLE OPERATIONS
====================
All operations reference features by column name.  Binary ops and multi-feature
ops create NEW columns and do not remove the originals unless you add a "drop".
Steps are applied left-to-right; later steps can use columns created earlier.

Original features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

── Unary transforms (modify a feature in-place) ─────────────────────────────
{"op": "log1p",          "features": ["MedInc"]}
{"op": "sqrt",           "features": ["Population"]}
{"op": "square",         "features": ["MedInc"]}
{"op": "cube",           "features": ["MedInc"]}
{"op": "reciprocal",     "features": ["AveOccup"], "epsilon": 1e-3}
{"op": "abs",            "features": ["Latitude"]}
{"op": "clip",           "features": ["AveRooms"], "low_pct": 1, "high_pct": 99}
{"op": "rank",           "features": ["MedInc"]}            // → [0,1] based on train CDF
{"op": "quantile_normal","features": ["Population"]}        // → N(0,1) via quantile transform
{"op": "bin",            "features": ["HouseAge"], "n_bins": 10}   // equal-frequency bins

── Binary ops (create a NEW named column) ───────────────────────────────────
{"op": "ratio",     "numerator": "AveRooms", "denominator": "AveBedrms", "name": "rooms_per_bedrm", "epsilon": 1e-6}
{"op": "product",   "a": "MedInc", "b": "AveRooms", "name": "income_x_rooms"}
{"op": "diff",      "a": "AveRooms", "b": "AveBedrms", "name": "extra_rooms"}
{"op": "sum_pair",  "a": "AveRooms", "b": "AveBedrms", "name": "total_rooms"}
{"op": "log_ratio", "numerator": "AveRooms", "denominator": "AveBedrms", "name": "log_room_ratio", "epsilon": 1e-6}

── Multi-feature ops (create several new columns) ───────────────────────────
{"op": "polynomial", "features": ["MedInc", "AveOccup"], "degree": 2, "interaction_only": false}
{"op": "interaction", "features": ["MedInc", "Latitude", "Longitude"]}  // all pairwise products

── Geographic ops ────────────────────────────────────────────────────────────
{"op": "kmeans_cluster",  "features": ["Latitude", "Longitude"], "n_clusters": 10, "name": "geo_cluster"}
{"op": "kmeans_distance", "features": ["Latitude", "Longitude"], "n_clusters": 8,  "prefix": "kdist"}
{"op": "distance_to_point", "lat": "Latitude", "lon": "Longitude",
    "target_lat": 37.77, "target_lon": -122.42, "name": "dist_sf"}
// More CA reference points: LA (34.05,-118.24), San Diego (32.72,-117.15),
//   San Jose (37.34,-121.89), Sacramento (38.58,-121.49), Fresno (36.74,-119.79)

── Selection ─────────────────────────────────────────────────────────────────
{"op": "drop",   "features": ["AveBedrms"]}
{"op": "select", "features": ["MedInc", "Latitude", "Longitude"]}  // keep only these

── Scaling (always include as the last step) ────────────────────────────────
{"op": "scale", "method": "standard"}   // StandardScaler
{"op": "scale", "method": "robust"}     // RobustScaler (percentile-based, outlier-robust)
{"op": "scale", "method": "minmax"}     // MinMaxScaler → [0,1]
{"op": "scale", "method": "quantile"}   // QuantileTransformer → N(0,1)
"""

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are an expert data scientist specializing in feature engineering for tabular regression.

TASK
====
Improve the feature engineering pipeline (pipeline.json) to minimize val_rmse on
the California Housing dataset.  The ML model (XGBoost) and ALL its hyperparameters
are FIXED.  Your only lever is the pipeline of feature operations.

{OPERATIONS_REFERENCE}

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
1. Return the COMPLETE pipeline (all steps), not just the new ones.
2. Always end with a "scale" step.
3. Use only operations from the list above — no others will be accepted.
4. New columns created by binary/multi-feature ops can be used by later steps.
5. When referencing a feature in a later step, use the exact name given by "name" or "prefix".

STRATEGY TIPS
=============
- Ratio features (AveRooms/AveBedrms, Population/AveOccup) are often powerful.
- Log-transform skewed features (MedInc, Population) before using them in ratios.
- Geographic clusters on (Latitude, Longitude) capture location effects well.
- Distance to multiple city centres can encode "urban proximity" signal.
- Polynomial degree-2 on a small set of strong predictors (MedInc, AveOccup) adds interactions.
- Clipping outliers before log/ratio ops prevents extreme values.
- Dropping weak or redundant features can improve generalisation.
- Combining operations: clip → log1p → polynomial is a valid and often effective chain.
- "robust" scaling is more stable when outliers remain after clipping.

Think step by step about what to try, then output only the JSON.
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


def call_llm(messages: list[dict], provider: str, model: str) -> str:
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return resp.content[0].text
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        return resp.choices[0].message.content
    raise ValueError(f"Unknown provider: {provider}")


# ─── Pipeline parsing & validation ───────────────────────────────────────────

# Import allowed ops list from operations module without executing it
_ALLOWED_OPS = {
    "log1p", "sqrt", "square", "cube", "reciprocal", "abs",
    "clip", "rank", "quantile_normal", "bin",
    "ratio", "product", "diff", "sum_pair", "log_ratio",
    "polynomial", "interaction",
    "kmeans_cluster", "kmeans_distance", "distance_to_point",
    "drop", "select", "scale",
}


def extract_pipeline(response: str) -> dict | None:
    """Extract the JSON pipeline from the LLM response."""
    match = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
    if not match:
        match = re.search(r"```\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    # fallback: try to find bare JSON object
    match = re.search(r"\{[\s\S]*\"steps\"[\s\S]*\}", response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def validate_pipeline(config: dict) -> tuple[bool, str]:
    """Return (ok, error_message)."""
    if not isinstance(config, dict):
        return False, "Top-level must be a JSON object"
    if "steps" not in config:
        return False, "Missing 'steps' key"
    steps = config["steps"]
    if not isinstance(steps, list) or len(steps) == 0:
        return False, "'steps' must be a non-empty list"
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return False, f"Step {i} is not an object"
        op = step.get("op")
        if op not in _ALLOWED_OPS:
            return False, f"Step {i}: unknown op {op!r}. Allowed: {sorted(_ALLOWED_OPS)}"
    if steps[-1].get("op") != "scale":
        return False, "Last step must be a 'scale' operation"
    if "description" not in config:
        return False, "Missing 'description' key"
    return True, ""


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
    val_rmse: float | None
    n_features: int | None
    crashed: bool
    output: str


def run_train() -> RunResult:
    try:
        proc = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return RunResult(None, None, True, "TIMEOUT")

    out = proc.stdout + proc.stderr
    crashed = proc.returncode != 0

    val_rmse = None
    n_features = None
    m = re.search(r"val_rmse\s*:\s*([0-9.]+)", out)
    if m:
        val_rmse = float(m.group(1))
    m = re.search(r"n_features\s*:\s*([0-9]+)", out)
    if m:
        n_features = int(m.group(1))
    if val_rmse is None:
        crashed = True

    return RunResult(val_rmse, n_features, crashed, out)


# ─── Results tracking ─────────────────────────────────────────────────────────


@dataclass
class Record:
    timestamp: str
    experiment: int
    description: str
    val_rmse: float | None
    n_features: int | None
    kept: bool
    crashed: bool


def init_results() -> None:
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "timestamp\texperiment\tdescription\tval_rmse\tn_features\tkept\tcrashed\n"
        )


def load_results() -> list[Record]:
    if not RESULTS_FILE.exists():
        return []
    records = []
    lines = RESULTS_FILE.read_text().splitlines()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        try:
            records.append(Record(
                timestamp=parts[0],
                experiment=int(parts[1]),
                description=parts[2],
                val_rmse=float(parts[3]) if parts[3] != "CRASH" else None,
                n_features=int(parts[4]) if parts[4] else None,
                kept=parts[5] == "yes",
                crashed=parts[6] == "yes",
            ))
        except (ValueError, IndexError):
            pass
    return records


def append_result(r: Record) -> None:
    with RESULTS_FILE.open("a") as f:
        f.write(
            f"{r.timestamp}\t{r.experiment}\t{r.description}\t"
            f"{r.val_rmse:.6f if r.val_rmse is not None else 'CRASH'}\t"
            f"{r.n_features or ''}\t"
            f"{'yes' if r.kept else 'no'}\t"
            f"{'yes' if r.crashed else 'no'}\n"
        )


def best_rmse(records: list[Record]) -> float:
    kept = [r.val_rmse for r in records if r.kept and r.val_rmse is not None]
    return min(kept) if kept else float("inf")


# ─── Prompt construction ──────────────────────────────────────────────────────


def format_history(records: list[Record]) -> str:
    if not records:
        return "(no experiments yet)"
    recent = records[-HISTORY_SIZE:]
    rows = ["#exp | val_rmse  | n_feat | kept | description"]
    rows.append("---- | --------- | ------ | ---- | -----------")
    for r in recent:
        rmse = f"{r.val_rmse:.6f}" if r.val_rmse else "CRASH   "
        feat = str(r.n_features) if r.n_features else "?"
        kept = "yes" if r.kept else "no "
        rows.append(f"{r.experiment:4d} | {rmse} | {feat:6s} | {kept}  | {r.description}")
    return "\n".join(rows)


def build_messages(pipeline_json: str, records: list[Record]) -> list[dict]:
    current_best = best_rmse(records)
    best_str = f"{current_best:.6f}" if current_best != float("inf") else "not yet measured"
    content = f"""\
Current pipeline.json:
```json
{pipeline_json}
```

Experiment history (most recent last):
{format_history(records)}

Current best val_rmse: {best_str}

Suggest ONE focused improvement and return the complete updated pipeline.json.
"""
    return [{"role": "user", "content": content}]


# ─── Main loop ────────────────────────────────────────────────────────────────


def main() -> None:
    provider, model = detect_provider()
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print()

    branch = git_setup_branch()
    print(f"Branch   : {branch}")

    init_results()
    records = load_results()
    exp_n = len(records) + 1

    # Baseline
    print("─" * 60)
    print("Measuring baseline...")
    baseline = run_train()
    if baseline.crashed:
        print("Baseline CRASHED:")
        print(baseline.output[-800:])
        sys.exit("Fix pipeline.json manually before running the agent.")
    print(f"Baseline  val_rmse={baseline.val_rmse:.6f}  n_features={baseline.n_features}")

    append_result(Record(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        experiment=exp_n,
        description="baseline",
        val_rmse=baseline.val_rmse,
        n_features=baseline.n_features,
        kept=True,
        crashed=False,
    ))
    records = load_results()
    exp_n += 1

    # Agent loop
    while True:
        print()
        print(f"{'─'*60}")
        print(f"Experiment #{exp_n}  |  best so far: {best_rmse(records):.6f}")

        pipeline_json = PIPELINE_FILE.read_text()
        messages = build_messages(pipeline_json, records)

        # Ask LLM
        new_config = None
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            print(f"Calling {provider}/{model} (attempt {attempt})...")
            try:
                response = call_llm(messages, provider, model)
            except Exception as e:
                print(f"  API error: {e}")
                time.sleep(5)
                continue

            new_config = extract_pipeline(response)
            if new_config is None:
                print("  No JSON found in response.")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Please return the pipeline inside a ```json ... ``` block."})
                continue

            ok, err = validate_pipeline(new_config)
            if not ok:
                print(f"  Invalid pipeline: {err}")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Pipeline validation failed: {err}. Please fix and return the complete pipeline."})
                new_config = None
                continue

            break

        if new_config is None:
            print(f"  Could not get valid pipeline after {MAX_LLM_RETRIES} attempts; skipping.")
            exp_n += 1
            continue

        description = new_config.get("description", "(no description)")
        print(f"  Change : {description}")
        print(f"  Steps  : {len(new_config['steps'])}")

        # Write and commit
        new_json = json.dumps(new_config, indent=2)
        if new_json.strip() == pipeline_json.strip():
            print("  Pipeline unchanged; skipping.")
            exp_n += 1
            continue

        PIPELINE_FILE.write_text(new_json + "\n")
        committed = git_commit(f"experiment #{exp_n}: {description}")
        if not committed:
            print("  Git commit failed (nothing changed?); skipping.")
            PIPELINE_FILE.write_text(pipeline_json)
            exp_n += 1
            continue

        # Evaluate
        print("  Running train.py...")
        result = run_train()

        if result.crashed:
            print(f"  CRASHED:\n{result.output[-600:]}")
            git_revert_last()
            rec = Record(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                experiment=exp_n,
                description=description,
                val_rmse=None,
                n_features=None,
                kept=False,
                crashed=True,
            )
        else:
            current_best = best_rmse(records)
            improved = result.val_rmse < current_best
            delta = result.val_rmse - current_best
            sign = "✓" if improved else "✗"
            print(f"  {sign} val_rmse={result.val_rmse:.6f}  Δ{delta:+.6f}  n_features={result.n_features}")
            if not improved:
                git_revert_last()
            rec = Record(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                experiment=exp_n,
                description=description,
                val_rmse=result.val_rmse,
                n_features=result.n_features,
                kept=improved,
                crashed=False,
            )

        append_result(rec)
        records = load_results()
        exp_n += 1


if __name__ == "__main__":
    main()
