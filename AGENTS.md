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
- **`config/rpg_finetune.yaml`**: The source of truth for training hyperparameters. Modify this instead of hardcoding values in Python scripts.
- **`src/training/finetune_lora.py`**: The main training entry point. It uses `Unsloth` for optimization. When modifying, preserve the `FastLanguageModel` loading logic as it is specific to the hardware optimizations.
- **`colab/run_pipeline.ipynb`**: The execution driver. If you add new pipeline steps (e.g., a new data processing script), update this notebook to ensure it runs in the cloud environment.

## Conventions
- **Paths**: Use relative paths in code, assuming execution from the project root (`llm-playground/`).
- **Dependencies**: Keep `requirements.txt` minimal and compatible with Google Colab's environment.
- **Logging**: Scripts should print clear status updates to stdout, as this is the primary way the user monitors progress in the Colab notebook cell output.

## Future Roadmap
1. **RAG Integration**: Add a vector store ingestion script to `src/data/` for rule retrieval.
2. **Advanced Synthetic Data**: Integrate `distilabel` for multi-turn agentic data generation.
3. **Validation**: Add a structured output validator to ensure generated RPG stats follow JSON schemas.

