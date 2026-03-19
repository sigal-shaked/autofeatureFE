# autofeatureFE

Autonomous feature engineering optimizer — an agent loop that uses an LLM API
(Claude / GPT-4) to iteratively improve `prepare.py` and minimize validation
RMSE on a tabular regression task.

**No GPU required.** The ML model (XGBoost) is fixed; only the data pipeline
is optimized.

## Quick start

```bash
cd autofeatureFE
uv sync

# Run with Anthropic Claude:
ANTHROPIC_API_KEY=sk-... uv run agent.py

# Run with OpenAI GPT-4:
OPENAI_API_KEY=sk-... LLM_PROVIDER=openai uv run agent.py
```

See [program.md](program.md) for full documentation.
