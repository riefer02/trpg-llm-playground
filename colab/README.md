# Colab Notebooks

Use the notebook that matches the stage you want to run. All notebooks expect paths to be configured in `config/*.yaml` and will clone this repo into `/content/trpg-llm-playground`.

## Notebooks

- `run_pipeline.ipynb`: Full pipeline (ingest → synth → train → push to HF Hub)
- `run_synthetic_only.ipynb`: Ingest + synthetic data generation only
- `run_train_after_synth.ipynb`: Train + push to HF Hub (assumes dataset already exists)

## Required Secrets

Add these to Colab Secrets (key icon in left sidebar):

| Secret | Required For | Where to Get |
|--------|--------------|--------------|
| `OPENAI_API_KEY` | Synthetic generation | [OpenAI API Keys](https://platform.openai.com/api-keys) |
| `HF_TOKEN` | Push model to Hub | [HF Tokens](https://huggingface.co/settings/tokens) |

## Pipeline Stages

### 1. Synthetic Generation
- Uses OpenAI API to generate Q&A pairs from your PDF
- Streams to Drive with resume support (survives disconnects)
- Quality pipeline: verification, dedup, negatives, multi-turn

### 2. Training
- Fine-tunes Qwen2.5-7B (or other) with LoRA via Unsloth
- A100-optimized config (~10-15 min for 500 samples)
- Saves checkpoints to Drive

### 3. Push to Hugging Face Hub
- Merges LoRA weights into base model
- Uploads to your HF account for deployment
- Configure `HF_USERNAME` and `MODEL_NAME` in the notebook

### 4. (Optional) GGUF Export
- Converts to GGUF format for Ollama/local inference
- Q4_K_M quantization: ~4.5GB download
- Instructions for Ollama setup included in output

## Quick Notes

- For a fast end-to-end test, enable `debug.enabled` in config and set small `max_pages`/`max_samples`.
- Training uses `config/rpg_finetune.yaml`—update `dataset_tag` to match your synthetic data.
- To keep every synthetic run, use `output.path` with `{run_id}`; training can load multiple files via glob.
- 7B models train on free T4 tier; A100 makes it ~4x faster.
