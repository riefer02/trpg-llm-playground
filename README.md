# TTRPG LLM Playground

A monorepo for Lancer TTRPG tooling: a type-safe mechanical core, a full-stack app, and an LLM pipeline. The focus is on rules and mechanics only (no setting text), with types as the source of truth.

## Architecture

This project is organized into three domains:

```text
trpg-llm-playground/
├── core/                   # Type-driven Lancer mechanics (Pydantic v2)
│   ├── character/          # Unified character model (pilot + mech)
│   ├── pilot/              # Pilot domain (skills, talents, licenses)
│   ├── mech/               # Mech domain (frames, weapons, combat)
│   ├── shared/             # Shared types (dice, enums, effects)
│   ├── npc/                # NPC templates and behavior
│   ├── gm_toolkit/         # SITREPs, encounters, world helpers
│   └── export.py           # JSON Schema export utility
│
├── app/                    # Full-stack web app (FastAPI + TanStack Start)
│   ├── backend/            # API layer (thin wrapper over core)
│   └── frontend/           # React app with generated types
│
├── llm/                    # Synthetic data + fine-tuning pipeline
│   ├── src/                # Data generation, RAG, training modules
│   ├── colab/              # Google Colab notebooks
│   ├── scripts/            # CLI tools
│   ├── config/             # YAML configuration
│   ├── dataset/            # Generated datasets
│   └── tests/              # LLM pipeline tests
│
├── books/                  # Source PDFs (not committed)
└── models/                 # GGUF models for local inference
```

## Quick Start

### Using the Type System

```python
from core import SkillSet, create_ll0_character, PilotTrigger, Talent
from core.pilot import STANDARD_BACKGROUNDS
from core.shared.ids import CharacterId, TalentId

# Create a new LL0 character (pilot + mech)
character = create_ll0_character(
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

stats = character.active_mech_stats
print(
    f"{character.pilot.callsign} - LL{character.pilot.level}, "
    f"Mech HP: {stats.hp}, Grit: {character.pilot.grit}"
)
# PHOENIX - LL0, Mech HP: 10, Grit: 0

# Export to JSON Schema
from core.export import export_all_schemas
export_all_schemas("schemas/")  # Creates individual schema files
```

### Running the App

```bash
make install-app
cp .env.example .env
make db-up
make db-migrate
make dev
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

The `/core` directory is the source of truth for all mechanics. The app and pipeline consume these types directly.

### Character Domain (`core/character/`)

| Module | Contents |
|--------|----------|
| `character.py` | Unified Character model (pilot + mech configurations) |
| `factory.py` | Character creation helpers (LL0, empty, etc.) |
| `validation.py` | Holistic character validation (pilot + mech rules) |

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

- Coverage is focused on **mechanical rules** only (no flavor text), with shared effect primitives in `core/shared/effects/`.
- Character is the primary entry point (pilot + mech); pilot, mech, NPC, and GM toolkit domains are typed and validated with ongoing expansion tracked in `notes/mechanics_coverage_map.md`.
- Remaining gaps are tracked in `notes/mechanics_coverage_map.md`; new mechanics should extend typed effects rather than add untyped strings.
- Core validation runs through schema tests (`make test-core`) and example builds; JSON Schema export is supported for integration.

---

## App Layer

The `/app` directory is a full-stack web application:

- FastAPI backend that validates input by calling `model_validate()` on core types
- JSON blob storage pattern for core models
- TanStack Start frontend with types generated from core JSON Schema

If a type is missing in the frontend, add it to `core/export.py` and run `make generate-types`.

---

## LLM Fine-Tuning Pipeline

The `/llm` directory contains the pipeline for synthetic data generation and fine-tuning.
The next phase is focused on a compact, quantized model that uses the core's type primitives to generate structured content for the app. This is under construction and still exploratory.

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

### Current Focus
- [ ] Real-time multiplayer combat updates (WebSockets)
- [ ] Movement path drawing and multi-target selection
- [ ] System activation UI
- [ ] Type-primitive, quantized generation pipeline (exploratory)

### Completed
- [x] Type-driven core mechanics (pilot, mech, NPC, GM toolkit)
- [x] Typed ID system and JSON Schema export
- [x] Full-stack app with core-validated API and generated frontend types
- [x] Synthetic data pipeline with verification and deduplication
- [x] Local RAG chat and evaluation harness

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Type System | Pydantic v2, Python 3.11+ |
| Backend | FastAPI, SQLModel, PostgreSQL |
| Frontend | TanStack Start, React Query |
| Training | Unsloth, LoRA, Hugging Face |
| Inference | Ollama, GGUF quantization |
| Data | PyMuPDF, OpenAI API |
| Types | JSON Schema export, json-schema-to-typescript |
