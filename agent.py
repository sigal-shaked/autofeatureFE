#!/usr/bin/env python3
"""
AutoFeature Agent

Autonomous agent that iteratively improves prepare.py using an LLM API.
Each iteration: read current prepare.py → ask LLM for improvement →
write new code → run train.py → keep if val_rmse improved, revert otherwise.

Usage:
    # Anthropic Claude (recommended):
    ANTHROPIC_API_KEY=sk-... uv run agent.py

    # OpenAI GPT-4:
    OPENAI_API_KEY=sk-... LLM_PROVIDER=openai uv run agent.py

    # Override model:
    ANTHROPIC_API_KEY=sk-... LLM_MODEL=claude-opus-4-6 uv run agent.py
"""

import ast
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o"
MAX_TOKENS = 8192
HISTORY_SIZE = 10        # recent experiments included in each LLM prompt
MAX_LLM_RETRIES = 3      # retries if LLM output is not valid Python
RESULTS_FILE = Path("results.tsv")
PREPARE_FILE = Path("prepare.py")
TRAIN_TIMEOUT = 300      # seconds before declaring a run failed

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert data scientist specializing in feature engineering for tabular regression.

TASK
====
Improve the prepare_data() function in prepare.py to minimize val_rmse on the
California Housing dataset.  The model (XGBoost) and ALL its hyperparameters are
FIXED — your only levers are feature engineering, preprocessing, and feature selection.

DATASET FEATURES
================
MedInc      — median income in block group
HouseAge    — median house age in block group
AveRooms    — average number of rooms per household
AveBedrms   — average number of bedrooms per household
Population  — block group population
AveOccup    — average number of household members
Latitude    — block group latitude
Longitude   — block group longitude
Target      — median house value in $100,000s (regression)

RULES
=====
1. Return the COMPLETE new prepare.py file inside a fenced code block: ```python ... ```
2. Keep RANDOM_SEED = 42 and VAL_SIZE = 0.2 unchanged.
3. prepare_data() must return (X_train, X_val, y_train, y_val) as float32 numpy arrays.
4. You may use: numpy, pandas, scikit-learn (any submodule).  No other dependencies.
5. Do NOT use external data.  Do NOT change the train/val split.

STRATEGY TIPS
=============
- Ratio features (AveRooms/AveBedrms, Population/AveOccup) are often powerful.
- Log-transforming skewed features (MedInc, Population) can help tree models.
- Geographic signals: distance to coast / city centres, lat×lon interaction.
- Cluster-based features: k-means on (Lat, Lon) to create location buckets.
- Polynomial interactions on strong predictors (MedInc, AveOccup).
- Target encoding or binning may not help (no leakage allowed on val split).
- Feature selection: dropping noisy or redundant features sometimes helps.
- XGBoost handles missing values, so you don't need to impute aggressively.

FORMAT
======
Start the file with exactly this comment (fill in the blank):
  # EXPERIMENT: <one-line description of the key change>

Then output the rest of the file.  Think step by step before writing.
"""

# ─── LLM client ───────────────────────────────────────────────────────────────


def detect_provider() -> tuple[str, str]:
    """Return (provider, model) based on environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if provider == "openai" or (not provider and os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY")):
        model = os.environ.get("LLM_MODEL", DEFAULT_OPENAI_MODEL)
        return "openai", model

    # Default: Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        if os.environ.get("OPENAI_API_KEY"):
            model = os.environ.get("LLM_MODEL", DEFAULT_OPENAI_MODEL)
            return "openai", model
        sys.exit("ERROR: Set ANTHROPIC_API_KEY or OPENAI_API_KEY before running the agent.")

    model = os.environ.get("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
    return "anthropic", model


def call_llm(messages: list[dict], provider: str, model: str) -> str:
    """Call the LLM and return the response text."""
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=full_messages,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ─── Code extraction & validation ─────────────────────────────────────────────


def extract_code(response: str) -> str | None:
    """Extract the Python code block from the LLM response."""
    # Match ```python ... ``` or ``` ... ```
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if not match:
        match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_experiment_desc(code: str) -> str:
    """Pull the # EXPERIMENT: line from the code."""
    for line in code.splitlines():
        if line.strip().startswith("# EXPERIMENT:"):
            return line.strip()[len("# EXPERIMENT:"):].strip()
    return "(no description)"


def validate_python(code: str) -> bool:
    """Return True if code parses as valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def has_prepare_data(code: str) -> bool:
    """Return True if code defines prepare_data()."""
    return "def prepare_data(" in code


# ─── Git helpers ──────────────────────────────────────────────────────────────


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def git_setup_branch() -> str:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = f"autofeature/{tag}"
    git("checkout", "-b", branch)
    print(f"Branch: {branch}")
    return branch


def git_commit(message: str) -> bool:
    git("add", str(PREPARE_FILE))
    result = git("commit", "-m", message, check=False)
    return result.returncode == 0


def git_revert_last() -> None:
    git("reset", "--hard", "HEAD~1")


# ─── Training ─────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    val_rmse: float | None
    n_features: int | None
    stdout: str
    stderr: str
    crashed: bool


def run_train() -> RunResult:
    """Run train.py and parse its output."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return RunResult(None, None, "", "TIMEOUT", crashed=True)

    stdout = proc.stdout
    stderr = proc.stderr
    crashed = proc.returncode != 0

    val_rmse = None
    n_features = None

    m = re.search(r"val_rmse\s*:\s*([0-9.]+)", stdout)
    if m:
        val_rmse = float(m.group(1))

    m = re.search(r"n_features\s*:\s*([0-9]+)", stdout)
    if m:
        n_features = int(m.group(1))

    if val_rmse is None:
        crashed = True

    return RunResult(val_rmse, n_features, stdout, stderr, crashed)


# ─── Results tracking ─────────────────────────────────────────────────────────


@dataclass
class ExperimentRecord:
    timestamp: str
    experiment: int
    description: str
    val_rmse: float | None
    n_features: int | None
    kept: bool
    crashed: bool


def load_results() -> list[ExperimentRecord]:
    records = []
    if not RESULTS_FILE.exists():
        return records
    with RESULTS_FILE.open() as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        parts = line.strip().split("\t")
        if len(parts) < 7:
            continue
        try:
            records.append(ExperimentRecord(
                timestamp=parts[0],
                experiment=int(parts[1]),
                description=parts[2],
                val_rmse=float(parts[3]) if parts[3] != "CRASH" else None,
                n_features=int(parts[4]) if parts[4] != "" else None,
                kept=parts[5] == "yes",
                crashed=parts[6] == "yes",
            ))
        except (ValueError, IndexError):
            pass
    return records


def init_results_file() -> None:
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "timestamp\texperiment\tdescription\tval_rmse\tn_features\tkept\tcrashed\n"
        )


def append_result(record: ExperimentRecord) -> None:
    with RESULTS_FILE.open("a") as f:
        f.write(
            f"{record.timestamp}\t{record.experiment}\t{record.description}\t"
            f"{record.val_rmse if record.val_rmse is not None else 'CRASH'}\t"
            f"{record.n_features or ''}\t"
            f"{'yes' if record.kept else 'no'}\t"
            f"{'yes' if record.crashed else 'no'}\n"
        )


def format_history(records: list[ExperimentRecord]) -> str:
    """Format recent experiment history for the LLM prompt."""
    if not records:
        return "(no experiments yet — this is the first run)"

    recent = records[-HISTORY_SIZE:]
    lines = ["experiment | val_rmse  | n_feat | kept | description"]
    lines.append("---------- | --------- | ------ | ---- | -----------")
    for r in recent:
        rmse_str = f"{r.val_rmse:.6f}" if r.val_rmse else "CRASH   "
        feat_str = str(r.n_features) if r.n_features else "?"
        kept_str = "yes" if r.kept else "no "
        lines.append(f"{r.experiment:10d} | {rmse_str} | {feat_str:6s} | {kept_str}  | {r.description}")
    return "\n".join(lines)


def best_rmse(records: list[ExperimentRecord]) -> float:
    kept = [r.val_rmse for r in records if r.kept and r.val_rmse is not None]
    return min(kept) if kept else float("inf")


# ─── LLM prompt construction ──────────────────────────────────────────────────


def build_prompt(prepare_code: str, records: list[ExperimentRecord]) -> list[dict]:
    """Build the message list for the LLM call."""
    current_best = best_rmse(records)
    best_str = f"{current_best:.6f}" if current_best != float("inf") else "not yet measured"

    user_content = f"""\
Current prepare.py:
```python
{prepare_code}
```

Recent experiment history (most recent last):
{format_history(records)}

Current best val_rmse: {best_str}

Suggest ONE focused improvement to the feature engineering pipeline and return the complete updated prepare.py.
"""
    return [{"role": "user", "content": user_content}]


# ─── Main agent loop ──────────────────────────────────────────────────────────


def main() -> None:
    provider, model = detect_provider()
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"Prepare  : {PREPARE_FILE.resolve()}")
    print()

    # Setup git branch
    try:
        branch = git_setup_branch()
    except subprocess.CalledProcessError as e:
        sys.exit(f"Git error: {e.stderr}")

    # Measure baseline
    init_results_file()
    records = load_results()
    experiment_n = len(records) + 1

    print("─" * 60)
    print("Measuring baseline...")
    baseline_result = run_train()
    if baseline_result.crashed:
        print("Baseline run CRASHED:")
        print(baseline_result.stderr[-1000:])
        sys.exit("Fix prepare.py manually before running the agent.")

    print(f"Baseline val_rmse : {baseline_result.val_rmse:.6f}")
    print(f"Baseline n_features: {baseline_result.n_features}")

    baseline_record = ExperimentRecord(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        experiment=experiment_n,
        description="baseline (unmodified prepare.py)",
        val_rmse=baseline_result.val_rmse,
        n_features=baseline_result.n_features,
        kept=True,
        crashed=False,
    )
    append_result(baseline_record)
    records.append(baseline_record)
    experiment_n += 1

    # Agent loop
    while True:
        print()
        print(f"{'─'*60}")
        print(f"Experiment #{experiment_n}  |  best so far: {best_rmse(records):.6f}")

        prepare_code = PREPARE_FILE.read_text()
        messages = build_prompt(prepare_code, records)

        # Ask LLM for improved prepare.py
        new_code = None
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            print(f"Calling {provider}/{model} (attempt {attempt})...")
            try:
                response = call_llm(messages, provider, model)
            except Exception as e:
                print(f"  LLM error: {e}")
                time.sleep(5)
                continue

            new_code = extract_code(response)
            if new_code is None:
                print("  No code block found in response; retrying...")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Please provide the complete prepare.py inside a ```python ... ``` code block."})
                continue

            if not validate_python(new_code):
                print("  SyntaxError in generated code; retrying...")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "The code has a syntax error. Please fix it and return the complete prepare.py."})
                new_code = None
                continue

            if not has_prepare_data(new_code):
                print("  prepare_data() not found; retrying...")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "The file must define a prepare_data() function. Please include it."})
                new_code = None
                continue

            break  # valid code obtained

        if new_code is None:
            print(f"  Failed to get valid code after {MAX_LLM_RETRIES} attempts; skipping.")
            continue

        description = extract_experiment_desc(new_code)
        print(f"  Change: {description}")

        # Write and commit
        PREPARE_FILE.write_text(new_code)
        committed = git_commit(f"experiment #{experiment_n}: {description}")
        if not committed:
            print("  Nothing changed (LLM returned same code); skipping.")
            PREPARE_FILE.write_text(prepare_code)  # restore
            continue

        # Evaluate
        print("  Running train.py...")
        result = run_train()

        if result.crashed:
            print(f"  CRASHED:\n{result.stderr[-500:]}")
            git_revert_last()
            record = ExperimentRecord(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                experiment=experiment_n,
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

            if improved:
                print(f"  ✓ IMPROVED  val_rmse={result.val_rmse:.6f}  (Δ{delta:+.6f})  features={result.n_features}")
            else:
                print(f"  ✗ no improvement  val_rmse={result.val_rmse:.6f}  (Δ{delta:+.6f})  features={result.n_features}")
                git_revert_last()

            record = ExperimentRecord(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                experiment=experiment_n,
                description=description,
                val_rmse=result.val_rmse,
                n_features=result.n_features,
                kept=improved,
                crashed=False,
            )

        append_result(record)
        records.append(record)
        experiment_n += 1


if __name__ == "__main__":
    main()
