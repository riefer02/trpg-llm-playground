# Colab Notebooks

Use the notebook that matches the stage you want to run. All notebooks expect paths to be configured in `config/*.yaml` and will clone this repo into `/content/trpg-llm-playground`.

## Notebooks

- `run_pipeline.ipynb`: Full pipeline (ingest -> synth -> train -> eval)
- `run_synthetic_only.ipynb`: Ingest + synthetic data generation only
- `run_train_after_synth.ipynb`: Train + eval only (assumes dataset already exists)

## Quick Notes

- Set `OPENAI_API_KEY` in Colab Secrets before running synthetic generation.
- Synthetic-only installs `requirements_synth.txt` to avoid heavy training deps.
- Train/Eval uses `config/rpg_finetune.yaml` and expects dataset paths to exist on Drive.
- For a fast end-to-end test, enable `debug.enabled` in `config/synthetic_generic.yaml` and set small `max_pages`/`max_samples`.
- To keep every synthetic run, use `output.path` with `{run_id}`; training can load multiple files via a glob pattern.
- Synthetic generation streams output to Drive and supports resume checkpoints; see `resume` in `config/synthetic_generic.yaml`.
