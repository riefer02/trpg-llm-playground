# TTRPG LLM Playground

This repository contains an end-to-end pipeline for fine-tuning Large Language Models (LLMs) on **Tabletop RPG (TTRPG)** systems. It is designed for a **hybrid workflow** prioritizing developer experience: manage configuration and code in a proper IDE, then execute training on scalable cloud GPUs (like Google Colab).

## 🚀 Workflow Overview

1.  **Configure Locally**: Edit hyperparameters and paths in `config/*.yaml`.
2.  **Push Changes**: Commit and push your changes to GitHub.
3.  **Run Remotely**: Open `colab/run_pipeline.ipynb` in Google Colab, which clones this repo and executes the pipeline.
4.  **Save Artifacts**: LoRA adapters and synthetic datasets are saved to your Google Drive.

## 📂 Project Structure

```text
llm-playground/
├── config/                 # Configuration files (YAML)
│   ├── rpg_finetune.yaml   # Model & training hyperparameters
│   └── synthetic_generic.yaml # Synthetic data generation settings
├── src/
│   ├── data/               # Data processing
│   │   ├── ingest_pdf.py       # Extracts text from RPG PDFs
│   │   └── generate_synthetic.py # Generates Q/A pairs via LLM
│   ├── training/           # Model training
│   │   ├── finetune_lora.py    # Unsloth/LoRA training script
│   │   └── evaluate.py         # Inference & testing script
│   └── utils/              # Shared utilities
├── colab/                  # Notebooks for remote execution
└── requirements.txt        # Dependencies
```

## 🛠️ Deployment Instructions

### Phase 1: Local Setup & Configuration

1.  **Install Dependencies** (for local linting/testing):
    ```bash
    pip install -r requirements.txt
    ```
2.  **Prepare Configs**:
    - Edit `config/rpg_finetune.yaml` to set your desired model parameters (e.g., learning rate, steps).
    - Ensure `dataset.train_path` points to where your data will be in the Colab environment (usually generated dynamically or pulled from Drive).

### Phase 2: Google Drive Setup

1.  Create a folder in your Google Drive: `MyDrive/llm_experiments`.
2.  Upload your **TTRPG Source Books** (PDFs) to `MyDrive/Books/`.
3.  (Optional) If you have existing datasets, upload them to `MyDrive/llm_experiments/datasets/`.

### Phase 3: Execution on Google Colab

1.  Navigate to `colab/run_pipeline.ipynb` in this repository.
2.  Click the "Open in Colab" button (if viewing on GitHub) or upload the notebook to Colab manually.
3.  **Important**: In the notebook, update the `REPO_URL` variable to point to your fork/repository.
4.  Run the notebook cells in order:
    - It will mount your Google Drive.
    - It will clone this repository into the Colab runtime.
    - It will install dependencies (Unsloth, etc.).
    - It will run the training script `src/training/finetune_lora.py` using your config.
5.  **Artifacts**: The trained LoRA adapters will be saved to your specified `output_dir` (default: `outputs/lancer_lora` inside the Colab, which you should configure to copy to Drive or download).

### Phase 4: Inference / Usage

To use your trained model:

1.  Run the `src/training/evaluate.py` script (either in Colab or locally with a GPU).
2.  Ensure it points to the directory containing your saved LoRA adapters.

## 🤖 Tech Stack

- **Unsloth**: Faster, memory-efficient Llama 3 training.
- **Hugging Face**: Transformers, Datasets, PEFT.
- **Google Colab**: Free T4 GPU execution environment.
