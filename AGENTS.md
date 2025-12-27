# Agents Documentation

## Project Context
This project is a **monorepo** for Lancer TTRPG tooling with three domains:
1. **`/core`**: Type-driven game schemas using Pydantic v2
2. **`/llm`**: Synthetic data generation and LLM fine-tuning pipeline
3. **`/app`**: Future application layer (placeholder)

## Monorepo Structure

```
trpg-llm-playground/
├── core/                   # Type-driven Lancer schemas
│   ├── pilot/              # Pilot domain models
│   ├── shared/             # Shared types (dice, enums)
│   └── export.py           # JSON Schema export
├── llm/                    # LLM pipeline (moved from root)
│   ├── src/                # Data, RAG, training modules
│   ├── colab/              # Notebooks (run from llm/)
│   ├── scripts/            # CLI tools
│   ├── config/             # YAML configs
│   └── dataset/            # Generated data
├── app/                    # Future web application
├── books/                  # Source PDFs (shared)
├── models/                 # GGUF models (shared)
└── tests/                  # Cross-domain tests
```

## Core Type System (`/core`)

### Pilot Domain (`core/pilot/`)
- **`skill.py`**: 4 mech skills (HULL, AGI, SYS, ENG) with triggers
- **`background.py`**: Pilot backgrounds with starting triggers
- **`talent.py`**: 3-rank talent definitions
- **`license.py`**: Manufacturer licenses (IPS-N, SSC, HORUS, HA)
- **`core_bonus.py`**: Core bonuses earned from maxed licenses
- **`pilot.py`**: Main Pilot model composing all above

### Shared Types (`core/shared/`)
- **`enums.py`**: ActionType, DamageType, RangeType, StatusType, etc.
- **`dice.py`**: DiceExpression with parsing, rolling, and stats

### JSON Schema Export
- **`export.py`**: Export Pydantic models to JSON Schema
  ```bash
  python -m core.export --output-dir schemas/
  python -m core.export --combined  # Single combined schema
  ```

## LLM Pipeline (`/llm`)

### Configuration
- **`llm/config/rpg_finetune.yaml`**: Training hyperparameters
- **`llm/config/synthetic_generic.yaml`**: Synthetic data settings
- **`llm/config/templates/*.yaml`**: Pre-built RPG configs

### Synthetic Data Pipeline
- **`llm/src/data/generate_synthetic.py`**: Main orchestrator
- **`llm/src/data/synth_prompts.py`**: Prompt templates
- **`llm/src/data/synth_multiturn.py`**: Multi-turn conversations
- **`llm/src/data/synth_walkthrough.py`**: Step-by-step walkthroughs
- **`llm/src/data/synth_filter.py`**: Topic-based chunk filtering
- **`llm/src/data/synth_difficulty.py`**: Difficulty stratification
- **`llm/src/data/synth_negatives.py`**: "Not found" examples
- **`llm/src/data/synth_verify.py`**: Answer verification
- **`llm/src/data/synth_dedup.py`**: Semantic deduplication
- **`llm/src/data/synth_report.py`**: Quality dashboard

### Training & Evaluation
- **`llm/src/training/finetune_lora.py`**: Unsloth/LoRA training
- **`llm/src/training/evaluate_rpg.py`**: RPG-specific benchmarks
- **`llm/scripts/run_eval_benchmark.py`**: CLI for benchmarks
- **`llm/dataset/`**: Generated datasets (user-created eval sets go here)

### Local Inference
- **`llm/scripts/local_chat.py`**: Ollama + RAG chat (CLI/Gradio)
- **`llm/docs/LOCAL_CHAT.md`**: Setup guide

### Notebooks
- **`llm/colab/run_pipeline.ipynb`**: Full pipeline
- **`llm/colab/run_synthetic_only.ipynb`**: Synthetic only
- **`llm/colab/run_train_after_synth.ipynb`**: Training only

**Note**: Notebooks `cd` into `/llm` after cloning. All paths are relative to `llm/`.

## Conventions

### Core Domain
- **Pydantic v2**: Use `model_config = {"frozen": True}` for immutable game rules
- **Literal types**: Prefer `Literal["a", "b"]` over `Enum` for better IDE support
- **ID-based references**: Use string IDs for database normalization readiness

### LLM Pipeline
- **Paths**: Relative to `llm/` directory
- **Config-Driven**: All settings in YAML, no hardcoded values
- **Logging**: Clear status updates for Colab monitoring

## Completed Features

### Core
- ✅ **Pilot Schemas**: Skills, backgrounds, talents, licenses, core bonuses
- ✅ **Shared Types**: Dice expressions, action/damage enums
- ✅ **JSON Schema Export**: Individual and combined schema files

### LLM
- ✅ **RAG Integration**: Heading-aware chunking, FAISS indexing
- ✅ **Multi-Turn Generation**: Follow-up conversations
- ✅ **Quality Pipeline**: Verification, deduplication, negatives
- ✅ **Multi-RPG Templates**: D&D, Lancer, Blades configs
- ✅ **Evaluation Benchmark**: Accuracy/grounding/citation metrics
- ✅ **Local Chat**: Ollama + RAG with Gradio UI

## Roadmap

### Next: Mech Domain (`core/mech/`)
- Frame definitions (size, armor, HP, mounts)
- Mount types and weapon slots
- Systems and tech actions
- Manufacturer-specific gear

### Future
- Combat domain (actions, conditions, initiative)
- Database layer (SQLModel)
- Web application (FastAPI + frontend)
- Game engine with type dispatch
