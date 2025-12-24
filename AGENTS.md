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
- **`src/data/synth_difficulty.py`**: Basic/Intermediate/Advanced stratification.
- **`src/data/synth_negatives.py`**: "Not found in context" example generation.
- **`src/data/synth_verify.py`**: LLM-based answer verification and correction.
- **`src/data/synth_dedup.py`**: Semantic deduplication via sentence embeddings.
- **`src/data/synth_report.py`**: Post-generation quality dashboard.

### Training & Evaluation
- **`src/training/finetune_lora.py`**: Unsloth/LoRA training. Preserve `FastLanguageModel` loading logic.
- **`src/training/evaluate_rpg.py`**: RPG-specific benchmark framework (accuracy, grounding, citations).

### Notebooks
- **`colab/run_pipeline.ipynb`**: Full pipeline execution driver.
- **`colab/run_synthetic_only.ipynb`**: Synthetic generation only.
- **`colab/run_train_after_synth.ipynb`**: Training after existing synthetic data.

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
- ✅ **Evaluation Benchmark**: RPG-specific accuracy/grounding/citation metrics.

## Future Roadmap
See `docs/SYNTH_ROADMAP.md` for detailed implementation status. Remaining items:
1. **Interactive Sample Review UI**: Web interface to approve/reject samples before training.
2. **Production RAG Pipeline**: Combine fine-tuned model with retrieval for deployment.
3. **Adversarial Examples**: Edge cases and trick questions for robustness.
4. **Entity-Aware Coverage**: Extract named abilities/items and generate targeted questions.
5. **Curriculum Learning**: Train on easy examples first, then progressively harder ones.

