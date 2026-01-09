# TTRPG LLM Playground

A **monorepo** for Lancer TTRPG tooling: type-driven game schemas, synthetic data generation, and LLM fine-tuning.

## Architecture

This project is organized into three domains:

```text
trpg-llm-playground/
├── core/                   # Type-driven Lancer schemas (Pydantic v2)
│   ├── pilot/              # Pilot domain (skills, talents, licenses, etc.)
│   │   └── tests/           # Pilot schema tests
│   ├── mech/               # Mech domain (frames, weapons, combat)
│   │   └── tests/           # Mech schema tests
│   ├── shared/             # Shared types (dice, enums, damage types)
│   │   └── tests/           # Shared schema tests
│   └── export.py           # JSON Schema export utility
│
├── llm/                    # LLM fine-tuning pipeline
│   ├── src/                # Data processing, RAG, training modules
│   ├── colab/              # Google Colab notebooks
│   ├── scripts/            # CLI tools
│   ├── config/             # YAML configuration
│   ├── dataset/            # Generated datasets
│   └── tests/              # LLM pipeline tests
│
├── app/                    # Future application layer (placeholder)
│
├── books/                  # Source PDFs (not committed)
└── models/                 # GGUF models for local inference
```

## Quick Start

### Using the Type System

```python
from core import Pilot, SkillSet, create_ll0_pilot, PilotTrigger, Talent
from core.pilot import STANDARD_BACKGROUNDS
from core.shared.ids import PilotId, TalentId

# Create a new pilot
pilot = create_ll0_pilot(
    callsign="PHOENIX",
    name="Alex Chen",
    background=STANDARD_BACKGROUNDS[0],
    skills=SkillSet(hull=1, agility=1),
    triggers=[
        PilotTrigger(trigger_id="assault", rank=2),
        PilotTrigger(trigger_id="threaten", rank=2),
        PilotTrigger(trigger_id="survive", rank=2),
        PilotTrigger(trigger_id="take_control", rank=2),
    ],
    talents=[
        Talent(talent_id=TalentId("ace"), rank=1),
        Talent(talent_id=TalentId("bonded"), rank=1),
        Talent(talent_id=TalentId("brutal"), rank=1),
    ],
)

print(f"{pilot.callsign} - LL{pilot.level}, HP: {pilot.hp}, Grit: {pilot.grit}")
# PHOENIX - LL0, HP: 6, Grit: 0

# Export to JSON Schema
from core.export import export_all_schemas
export_all_schemas("schemas/")  # Creates individual schema files
```

### Running the LLM Pipeline

```bash
cd llm

# Generate synthetic training data
python -m src.data.generate_synthetic --config config/synthetic_generic.yaml

# Run local chat with Ollama + RAG
python scripts/local_chat.py --chunks dataset/lancer_v1_chunks.jsonl --ui
```

### Google Colab Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riefer02/trpg-llm-playground/blob/main/llm/colab/run_pipeline.ipynb) Full Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riefer02/trpg-llm-playground/blob/main/llm/colab/run_synthetic_only.ipynb) Synthetic Only

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riefer02/trpg-llm-playground/blob/main/llm/colab/run_train_after_synth.ipynb) Train/Eval Only

---

## Core Type System

The `/core` directory contains Pydantic v2 models that define the Lancer game system:

### Pilot Domain (`core/pilot/`)

| Module | Contents |
|--------|----------|
| `skill.py` | 4 mech skills (HULL, AGI, SYS, ENG) + triggers |
| `background.py` | Pilot backgrounds with starting triggers |
| `talent.py` | 3-rank talent trees |
| `license.py` | Manufacturer licenses (IPS-N, SSC, HORUS, HA) |
| `core_bonus.py` | Core bonuses earned from maxed licenses |
| `pilot.py` | Main Pilot model composing all above |

### Shared Types (`core/shared/`)

| Module | Contents |
|--------|----------|
| `enums.py` | Action types, damage types, status effects |
| `dice.py` | Dice expressions with parsing and rolling |
| `ids.py` | 36 typed ID definitions (PilotId, MechId, WeaponId, etc.) |
| `id_helpers.py` | IdField[T] pattern for Pydantic model fields with coercion |

### Typed ID System

The project uses typed IDs for compile-time type safety:

```python
from core.shared.ids import PilotId, WeaponId, CombatantId
from core.shared.id_helpers import PilotIdField, CombatantIdField

class MountedWeapon(FrozenModel):
    weapon_id: WeaponId  # Typed ID for type checking

class AttackResult(FrozenModel):
    attacker_id: CombatantIdField  # Coerces "c1" → CombatantId("c1")
    target_id: CombatantIdField | None
```

Benefits:
- Type checkers catch ID mismatches (e.g., `WeaponId` where `SystemId` expected)
- IdField[T] pattern maintains backward compatibility with string inputs
- 36 typed IDs cover all game entities, equipment, and combat objects
- See `notes/phase_3b_implementation_plan.md` for migration details

### JSON Schema Export

```bash
# Export individual schemas
python -m core.export --output-dir schemas/

# Export combined schema
python -m core.export --output-dir schemas/ --combined

# Export single model
python -m core.export --model Pilot
```

### Coverage & Validation Status

- Coverage is focused on **mechanical rules** only (no flavor text), with shared effect primitives in `core/shared/effects.py`.
- Pilot domain (skills, triggers, gear, talents, licenses, core bonuses) is typed and validated; mech domain (frames, weapons, systems, combat) is actively expanding.
- Remaining gaps are tracked in `notes/mechanics_coverage_map.md`; new mechanics should extend typed effects rather than add untyped strings.
- Core validation runs through schema tests (`make test-core`) and example builds; JSON Schema export is supported for integration.

---

## LLM Fine-Tuning Pipeline

The `/llm` directory contains the complete pipeline for generating synthetic training data and fine-tuning models.

### Configuration

Edit configs in `llm/config/`:
- `synthetic_generic.yaml` - Synthetic data generation settings
- `rpg_finetune.yaml` - Training hyperparameters
- `templates/*.yaml` - Pre-built configs for different RPG systems

### Quality Features

- **Negative Examples** (12%): Teaches "Not found in context" responses
- **Answer Verification**: LLM-scored quality filtering
- **Semantic Deduplication**: Removes similar questions
- **Multi-Turn Conversations** (20%): Follow-up dialogues
- **Difficulty Stratification**: Basic/Intermediate/Advanced mix

### Local Development

```bash
# Run core schema tests
make test-core

# Run LLM tests
make test-llm

# Run all tests
make test

# Validate synthetic data
cd llm
python scripts/validate_synth.py --input dataset/*.jsonl

# Generate quality report
python -m src.data.synth_report --input dataset/my_synthetic.jsonl
```

### Testing Expectations

- Core changes should always run `make test-core` before committing.
- LLM changes should run `make test-llm` (no third-party calls; mock mode is supported).

See `llm/docs/` for detailed documentation:
- `CONFIG.md` - Configuration reference
- `LOCAL_CHAT.md` - Ollama + RAG setup
- `FOCUSED_GENERATION.md` - Topic filtering & dataset merging

---

## Deployment Options

### 1. HF Spaces (Quickest)

Push trained model to Hugging Face Hub, deploy Gradio app.

### 2. Local Ollama + RAG

```bash
cd llm
pip install -r requirements_local.txt
ollama create lancer-rules -f ../models/Modelfile.lancer
python scripts/local_chat.py --ui
```

### 3. API Deployment

Modal, Replicate, or dedicated GPU hosting (RunPod, Vast.ai).

---

## Roadmap

### Completed
- [x] Type-driven Pilot schemas with Pydantic v2
- [x] Typed ID system (36 NewType definitions with coercion)
- [x] Mech domain schemas (frames, systems, weapons)
- [x] Combat domain schemas (actions, conditions, initiative)
- [x] JSON Schema export for cross-language use
- [x] Synthetic data generation pipeline
- [x] Multi-turn conversation generation
- [x] Quality verification and deduplication
- [x] Local Ollama + RAG chat

### Planned
- [ ] Database layer (SQLModel)
- [ ] Web application (FastAPI + frontend)
- [ ] Game engine using type dispatch

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Type System | Pydantic v2, Python 3.11+ |
| Training | Unsloth, LoRA, Hugging Face |
| Inference | Ollama, GGUF quantization |
| Data | PyMuPDF, OpenAI API |
| Future App | FastAPI, SQLModel |
