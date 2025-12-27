# Agents Documentation

## Project Context
This project, **TTRPG LLM Playground**, is a synthetic data generation and fine-tuning pipeline for Tabletop RPG systems. 
The goal is to create models capable of understanding game rules, lore, and generating consistent scenarios for various TTRPGs.

## Core Architecture
- **Hybrid Workflow**: 
  - **Local (Cursor)**: Code editing, configuration (`.yaml`), version control.
  - **Remote (Colab)**: Heavy compute execution (Training, Inference).
  - **Bridge**: GitHub for code sync, Google Drive for artifact storage (datasets, model weights).

## Key Files for Agents

### Configuration
- **`config/rpg_finetune.yaml`**: Training hyperparameters. Modify this instead of hardcoding values.
- **`config/synthetic_generic.yaml`**: Synthetic data generation settings including quality enhancements.
- **`config/templates/*.yaml`**: Pre-built configs for D&D 5e, Lancer, Blades in the Dark.

### Synthetic Data Pipeline
- **`src/data/generate_synthetic.py`**: Main generation orchestrator. Integrates all quality modules.
- **`src/data/synth_prompts.py`**: Configurable prompt templates (no more hardcoded system strings).
- **`src/data/synth_multiturn.py`**: Multi-turn conversation generation.
- **`src/data/synth_walkthrough.py`**: Guided step-by-step walkthrough conversations.
- **`src/data/synth_filter.py`**: Topic-based chunk filtering (keyword + semantic).
- **`src/data/synth_difficulty.py`**: Basic/Intermediate/Advanced stratification.
- **`src/data/synth_negatives.py`**: "Not found in context" example generation.
- **`src/data/synth_verify.py`**: LLM-based answer verification and correction.
- **`src/data/synth_dedup.py`**: Semantic deduplication via sentence embeddings.
- **`src/data/synth_report.py`**: Post-generation quality dashboard.

### Utilities
- **`scripts/merge_datasets.py`**: Combine multiple JSONL datasets with optional deduplication.

### Training & Evaluation
- **`src/training/finetune_lora.py`**: Unsloth/LoRA training. Preserve `FastLanguageModel` loading logic.
- **`src/training/evaluate_rpg.py`**: RPG-specific benchmark framework (accuracy, grounding, citations).
- **`src/training/evaluate.py`**: Quick sanity check with sample questions.
- **`scripts/run_eval_benchmark.py`**: CLI runner for evaluation benchmarks (Ollama or HF models).
- **`dataset/lancer_eval_benchmark.yaml`**: Curated evaluation set with 20 annotated examples.

### Local Inference
- **`scripts/local_chat.py`**: Local chat with Ollama + RAG. Supports CLI and Gradio UI modes.
- **`requirements_local.txt`**: Dependencies for local chat (sentence-transformers, faiss-cpu, ollama, gradio).
- **`docs/LOCAL_CHAT.md`**: Setup guide for local Ollama + RAG deployment.

### Notebooks
- **`colab/run_pipeline.ipynb`**: Full pipeline (ingest → synth → train → push to HF Hub).
- **`colab/run_synthetic_only.ipynb`**: Synthetic generation only.
- **`colab/run_train_after_synth.ipynb`**: Training + HF Hub push (assumes dataset exists).

## Conventions
- **Paths**: Use relative paths in code, assuming execution from the project root (`llm-playground/`).
- **Dependencies**: Keep `requirements.txt` minimal and compatible with Google Colab's environment. Quality features require `sentence-transformers` (in `requirements_synth.txt`).
- **Logging**: Scripts should print clear status updates to stdout for Colab cell monitoring.
- **Config-Driven**: All major settings should come from YAML configs, not hardcoded values.

## Completed Features
- ✅ **RAG Integration**: Heading-aware chunking with stable IDs, FAISS indexing (`src/rag/`).
- ✅ **Multi-Turn Generation**: Realistic follow-up conversations (`synth_multiturn.py`).
- ✅ **Quality Pipeline**: Verification, deduplication, negative examples.
- ✅ **Multi-RPG Templates**: D&D, Lancer, Blades configs ready to use.
- ✅ **Evaluation Benchmark**: RPG-specific accuracy/grounding/citation/refusal metrics with curated Lancer eval set.
- ✅ **HF Hub Integration**: Notebooks include cells for pushing models to Hugging Face.
- ✅ **GGUF Export**: Optional cell for Ollama/local deployment.
- ✅ **Local Chat Script**: `scripts/local_chat.py` with Ollama + RAG for local testing.
- ✅ **Topic-Focused Generation**: Filter chunks by section keywords or semantic similarity.
- ✅ **Walkthrough Mode**: Generate guided step-by-step conversations for processes like character creation.
- ✅ **Dataset Merging**: Combine focused datasets with general ones using `scripts/merge_datasets.py`.

## Deployment Options
1. **HF Spaces**: Gradio app with RAG (see `notes/hf_spaces_deployment.md` for template).
2. **Local Ollama**: GGUF model + `local_chat.py` for fully offline use.
3. **API Services**: Modal, Replicate, or dedicated GPU hosting.

## Future Ideas (If Needed)
- **Entity-Aware Coverage**: Extract named abilities/items and generate targeted questions
- **Expand Eval Benchmarks**: Add D&D 5e and Blades eval sets following the Lancer template
- **Automated Eval in CI**: Run benchmarks on push to track model quality over time
- **Desktop App**: Tauri/Electron wrapper for end-user distribution

