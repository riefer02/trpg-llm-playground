# TTRPG LLM Playground

This repository contains an end-to-end pipeline for fine-tuning Large Language Models (LLMs) on **Tabletop RPG (TTRPG)** systems. It is designed for a **hybrid workflow** prioritizing developer experience: manage configuration and code in a proper IDE, verify locally with smoke tests, then execute training on scalable cloud GPUs (like Google Colab).

## 🚀 Workflow Overview

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riefer02/trpg-llm-playground/blob/main/colab/run_pipeline.ipynb)
[![Open In Colab (Synthetic Only)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riefer02/trpg-llm-playground/blob/main/colab/run_synthetic_only.ipynb)
[![Open In Colab (Train/Eval Only)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riefer02/trpg-llm-playground/blob/main/colab/run_train_after_synth.ipynb)

1.  **Configure Locally**: Edit hyperparameters and paths in `config/*.yaml`.
2.  **Verify Locally**: Run smoke tests (`tests/`) to ensure your logic and configs are sound before deploying.
3.  **Push Changes**: Commit and push your changes to GitHub.
4.  **Run Remotely**: Open `colab/run_pipeline.ipynb` in Google Colab, which clones this repo and executes the pipeline.
5.  **Save Artifacts**: LoRA adapters, synthetic datasets, and reproducibility recipes are saved automatically to your Google Drive.

## 📂 Project Structure

```text
llm-playground/
├── config/                 # Configuration files (YAML)
│   ├── rpg_finetune.yaml       # Model & training hyperparameters
│   ├── synthetic_generic.yaml  # Synthetic data generation settings
│   └── templates/              # Pre-built configs for different RPGs
│       ├── dnd5e.yaml          # D&D 5th Edition
│       ├── lancer.yaml         # Lancer RPG
│       └── blades.yaml         # Blades in the Dark
├── src/
│   ├── data/               # Data processing & synthetic generation
│   │   ├── ingest_pdf.py       # Extracts text from RPG PDFs
│   │   ├── generate_synthetic.py # Main generation pipeline
│   │   ├── synth_prompts.py    # Configurable prompt templates
│   │   ├── synth_multiturn.py  # Multi-turn conversation generation
│   │   ├── synth_difficulty.py # Difficulty stratification
│   │   ├── synth_negatives.py  # "Not found" example generation
│   │   ├── synth_verify.py     # Answer verification & quality scoring
│   │   ├── synth_dedup.py      # Semantic deduplication
│   │   └── synth_report.py     # Quality dashboard reports
│   ├── training/           # Model training & evaluation
│   │   ├── finetune_lora.py    # Unsloth/LoRA training script
│   │   ├── evaluate.py         # Inference & testing script
│   │   └── evaluate_rpg.py     # RPG-specific benchmark framework
│   ├── rag/                # RAG ingestion & indexing
│   └── utils/              # Shared utilities
├── colab/                  # Notebooks for remote execution
├── docs/                   # Documentation
│   ├── CONFIG.md           # Configuration reference
│   └── SYNTH_ROADMAP.md    # Feature roadmap & implementation status
├── tests/                  # Local smoke tests
├── requirements.txt        # Full pipeline dependencies
└── requirements_synth.txt  # Synthetic-only dependencies
```

## 📘 Configuration Quick Ref

See `docs/CONFIG.md` for a concise reference of the synthetic + training config blocks.

## 🎮 Multi-RPG Template System

Pre-built configuration templates make it easy to train models for different game systems:

```bash
# Copy a template and customize
cp config/templates/dnd5e.yaml config/my_dnd_campaign.yaml
# Edit paths and settings, then run
python -m src.data.generate_synthetic --config config/my_dnd_campaign.yaml
```

Available templates:
| Template | System | Focus |
|----------|--------|-------|
| `dnd5e.yaml` | D&D 5th Edition | Spells, builds, classic fantasy |
| `lancer.yaml` | Lancer RPG | Tactical mech combat |
| `blades.yaml` | Blades in the Dark | Heist-focused narrative play |
| `_base.yaml` | Template | Starting point for new systems |

## 🔬 Quality Enhancement Pipeline

The synthetic generation pipeline includes multiple quality enhancement passes:

### Automatic Quality Features
- **Negative Examples** (12% default): Teaches model to say "Not found in context" when appropriate
- **Answer Verification**: LLM-scored quality filtering with automatic correction
- **Semantic Deduplication**: Removes similar questions using embeddings
- **Multi-Turn Conversations** (20% default): Realistic follow-up Q&A dialogues
- **Difficulty Stratification**: Basic/Intermediate/Advanced question mix

### Post-Generation Reports
```bash
# Generate a quality dashboard after synthetic generation
python -m src.data.synth_report --input dataset/my_synthetic.jsonl --output report.md
```

### Model Evaluation Benchmark
```bash
# Create evaluation template
python -m src.training.evaluate_rpg --create-template evals/my_game_eval.yaml
# Edit with test questions, then benchmark your model
```

Configuration for quality features in `synthetic_generic.yaml`:
```yaml
negatives:
  enabled: true
  ratio: 0.12

verification:
  enabled: true
  threshold: 4

deduplication:
  enabled: true
  similarity_threshold: 0.85

multiturn:
  enabled: true
  ratio: 0.20
  min_turns: 2
  max_turns: 3

difficulty:
  enabled: true
  distribution:
    basic: 0.30
    intermediate: 0.50
    advanced: 0.20
```

## 🔎 Optional RAG (Chunks + Semantic Search)

This repo now supports an **optional** RAG ingestion path that creates **heading-aware, overlapping chunks with stable IDs** and can build a **FAISS** index for top-k semantic search.

- Chunk the PDF into stable artifacts:
  - Enable `rag_ingest.enabled: true` in `config/synthetic_generic.yaml`
  - Run: `python -m src.rag.chunk_pdf --config config/synthetic_generic.yaml`
- Build/query an index (optional):
  - Install: `pip install -r requirements_rag.txt`
  - Build: `python -m src.rag.build_index --config config/synthetic_generic.yaml`
  - Query: `python -m src.rag.query --config config/synthetic_generic.yaml --query "How does Overcharge work?"`

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
  raw_output_path: "/content/drive/MyDrive/llm_experiments/datasets/{project_name}_{dataset_tag}_raw.jsonl"
  output_format: "jsonl"
  flush_every: 50
debug:
  enabled: false                # Set true for quick end-to-end tests
  max_pages: 5                  # Limit ingest to first N pages
  max_samples: 10               # Limit synthetic samples
limits:
  enforce_max_samples: true     # Set false to ignore n_samples cap
generation:
  shuffle: true
  shuffle_seed: 1337
context:
  prev_chars: 400
  next_chars: 400
tables:
  enabled: true
  min_rows: 4
  min_cols: 2
  max_rows: 12
  max_cols: 5
  max_pairs: 5
  task_type: "rules_qa"
coverage:
  enabled: true
  min_text_len: 1400
  max_pairs: 2
  task_type: "rules_qa"
rag_mode:
  enabled: true
  emit_messages: true
  citation_style: "(p. {page})"
resume:
  enabled: true
  checkpoint_path: "/content/drive/MyDrive/llm_experiments/datasets/{project_name}_{dataset_tag}_resume.json"
  allow_mismatch: false
  force_restart: false
output:
  path: "/content/drive/MyDrive/llm_experiments/datasets/{project_name}_{dataset_tag}_synthetic_{run_id}.jsonl"
  run_id: "auto"
  append: false
  flush_every: 50
llm:
  model: "gpt-5-mini"
  temperature: null
  max_output_tokens: null
  max_completion_tokens: null
  repair_invalid_json: true
  invalid_response_log: "/content/drive/MyDrive/llm_experiments/datasets/invalid_synth_responses.log"
logging:
  quiet: false
  max_warnings: 10
  hud_every: 10
task_types:
  - "rules_qa"
  - "character_build"
  - "scenario_seed"
  - "gm_guidance"
  - "lore"
```
`tables` and `coverage` add lightweight passes to capture tabular stats and dense rule pages.

Validate a synthetic JSONL (optional):
```bash
python scripts/validate_synth.py --input /content/drive/MyDrive/llm_experiments/datasets/lancer_v1_ctx4096_synthetic_*.jsonl
```

Sample-audit grounding quality (optional):
```bash
python scripts/audit_synth.py --input /content/drive/MyDrive/llm_experiments/datasets/lancer_v1_ctx4096_synthetic_*.jsonl --sample 50
```

**`config/rpg_finetune.yaml`**:
```yaml
project_name: "lancer"
dataset_tag: "v1_ctx4096"
dataset:
  train_path: "/content/drive/MyDrive/llm_experiments/datasets/{project_name}_{dataset_tag}_synthetic*.jsonl"
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

### Preflight PDF Analysis (Optional)

Before generating synthetic data, you can analyze the PDF to estimate density, identify table-heavy pages, and estimate how many Q/A pairs a full run will produce.

```bash
python -m src.data.analyze_pdf --pdf_path "books/Lancer Core Book (PR2).pdf"
```

### Resumable Synthetic Generation

Synthetic generation writes each record to Drive as it is produced and periodically flushes, so partial runs persist through Colab disconnects. A lightweight resume checkpoint (based on your config + input file signature) lets you continue where you left off.

- Resume behavior is controlled in `resume` (see config above).
- If the book or settings change, the checkpoint signature will mismatch and a fresh run starts (unless you set `resume.allow_mismatch: true`).

### Phase 3: Execution on Google Colab

1.  Choose the notebook that matches your run:
    - Full pipeline: `colab/run_pipeline.ipynb`
    - Synthetic-only: `colab/run_synthetic_only.ipynb`
    - Train/Eval only: `colab/run_train_after_synth.ipynb`
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
