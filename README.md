# TTRPG LLM Playground

This repository contains an end-to-end pipeline for fine-tuning Large Language Models (LLMs) on **Tabletop RPG (TTRPG)** systems. It is designed for a **hybrid workflow** prioritizing developer experience: manage configuration and code in a proper IDE, verify locally with smoke tests, then execute training on scalable cloud GPUs (like Google Colab).

## 🚀 Workflow Overview

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riefer02/trpg-llm-playground/blob/main/colab/run_pipeline.ipynb)

1.  **Configure Locally**: Edit hyperparameters and paths in `config/*.yaml`.
2.  **Verify Locally**: Run smoke tests (`tests/`) to ensure your logic and configs are sound before deploying.
3.  **Push Changes**: Commit and push your changes to GitHub.
4.  **Run Remotely**: Open `colab/run_pipeline.ipynb` in Google Colab, which clones this repo and executes the pipeline.
5.  **Save Artifacts**: LoRA adapters, synthetic datasets, and reproducibility recipes are saved automatically to your Google Drive.

## 📂 Project Structure

```text
llm-playground/
├── config/                 # Configuration files (YAML)
│   ├── rpg_finetune.yaml   # Model & training hyperparameters
│   └── synthetic_generic.yaml # Synthetic data generation settings
├── src/
│   ├── data/               # Data processing
│   │   ├── ingest_pdf.py       # Extracts text from RPG PDFs
│   │   └── generate_synthetic.py # Generates Q/A pairs via LLM (GPT-4o/5.1)
│   ├── training/           # Model training
│   │   ├── finetune_lora.py    # Unsloth/LoRA training script
│   │   └── evaluate.py         # Inference & testing script
│   └── utils/              # Shared utilities
├── colab/                  # Notebooks for remote execution
├── tests/                  # Local smoke tests
└── requirements.txt        # Dependencies
```

## 🛠️ Deployment Instructions

### Phase 1: Local Setup & Verification

**Smoke Testing (Recommended)**
Before pushing to Colab, verify your pipeline logic locally on your Mac/PC using `uv`. This runs a mock pipeline (without GPU) to catch config errors or logic bugs.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the smoke test suite
uv run --with PyYAML --with tqdm --with openai python tests/smoke_test.py
```

### Phase 2: Configuration & Google Drive

**Dynamic Project Management**
The configs use variables (`project_name`, `dataset_tag`) to organize your experiments in Google Drive automatically.

**`config/synthetic_generic.yaml`**:
```yaml
project_name: "lancer"          # e.g., "dnd5e", "cyberpunk"
dataset_tag: "v1_ctx4096"       # Version + Context Length
ingest:
  pdf_path: "/content/drive/MyDrive/Books/Lancer Core Book.pdf"
```

**`config/rpg_finetune.yaml`**:
```yaml
project_name: "lancer"
dataset_tag: "v1_ctx4096"
training:
  report_to: "wandb" # Optional: Track experiments with Weights & Biases
```

**Drive Structure (Created Automatically)**:
```text
MyDrive/
  llm_experiments/
    datasets/
      lancer_v1_ctx4096_synthetic.jsonl  # Generated Training Data
    outputs/
      lancer_v1_ctx4096_lora/           # Saved Model & Adapters
      training_config_captured.yaml     # The exact "Recipe" used
```

### Phase 3: Execution on Google Colab

1.  Navigate to `colab/run_pipeline.ipynb` in this repository.
2.  Click the "Open in Colab" button.
3.  **Important**: Update the `REPO_URL` variable if you forked this repo.
4.  Run the notebook cells in order:
    - **Ingest**: Extracts text from your PDF.
    - **Generate**: Creates high-quality Q/A pairs using OpenAI (requires API Key).
    - **Train**: Fine-tunes a Qwen/Llama model using Unsloth (up to 2x faster).
    - **Evaluate**: Tests the model against tricky questions.

## 🤖 Tech Stack & Models

### Recommended Models (Unsloth Optimized)

-   **High Performance (A100/H100)**: `unsloth/Qwen2.5-32B-Instruct-bnb-4bit`
    -   Best for complex reasoning and deep rule understanding.
    -   Requires ~20GB+ VRAM.
-   **Balanced / Free Tier (T4)**: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
    -   Good reasoning, fits on free Colab tier.
    -   Alternative: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`.

### Best Practices & "Pro" Tips

1.  **Recipe Capture**: The training script automatically saves `training_config_captured.yaml` alongside your model. This guarantees reproducibility—you'll always know exactly what hyperparameters produced that specific model.
2.  **Dataset Tagging**: Use the `dataset_tag` (e.g., `v1_ctx4096`) to manage different versions of your data (short vs. long context, different prompt styles) without overwriting previous work.
3.  **Golden Validation**: Create a `val.jsonl` file with ~50 tricky "unit test" questions that you **never** train on. The `evaluate.py` script will automatically pick this up to give you an objective benchmark of model performance.
4.  **Experiment Tracking**: Set `report_to: "wandb"` in `rpg_finetune.yaml` to log professional-grade loss curves and system metrics to the cloud.

## Dependencies

-   **Unsloth**: Faster, memory-efficient Llama 3 training.
-   **Hugging Face**: Transformers, Datasets, PEFT.
-   **PyMuPDF**: Robust PDF text extraction.
-   **OpenAI**: Synthetic data generation.
