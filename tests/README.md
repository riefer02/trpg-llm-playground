# Local Smoke Tests

This directory contains scripts to verify the pipeline logic locally on your Mac, without needing a GPU or full dataset.

## How to Run

We use `uv` to manage dependencies and run the test in an isolated environment (avoiding the heavy GPU dependencies in `requirements.txt`).

```bash
uv run --with PyYAML --with tqdm --with openai python tests/smoke_test.py
```

## What it Tests

1.  **Ingestion Mock**: Creates a fake `raw_extracted.json` to simulate PDF text extraction.
2.  **Synthetic Data Generation**: Runs `src/data/generate_synthetic.py` using a **Mock LLM**.
    *   *Note*: The `src/utils/llm_client.py` has been updated to return valid dummy JSON when no `OPENAI_API_KEY` is present.
3.  **Config Validation**: Checks `config/rpg_finetune.yaml` to ensure the structure matches what the training script expects.

## Output

Artifacts are generated in `tests/artifacts/`:
*   `mock_raw.json`: The fake ingested text.
*   `mock_synthetic.jsonl`: The resulting training data samples.

