# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monorepo for Lancer TTRPG tooling with three domains:
- **`/core`**: Type-driven game schemas using Pydantic v2 (3277+ tests)
- **`/llm`**: Synthetic data generation and LLM fine-tuning pipeline
- **`/app`**: Full-stack web application (FastAPI backend + TanStack Start frontend)

## Core-First Architecture

**Golden Rule: Core is the source of truth. Never duplicate what core provides.**

```
core/ (Pydantic v2 models)
  ↓ exports via JSON Schema
app/backend/ (thin FastAPI wrapper)
  ↓ generated types
app/frontend/ (TanStack Start + React)
```

### Correct Patterns

```python
# ✅ RIGHT: Use core validation via shared utility
from app.backend.utils import validate_core_model
pilot = validate_core_model(Pilot, request_data, "pilot")

# ✅ RIGHT: Store as JSON blob, validate on retrieval
db_record.data = pilot.model_dump(mode="json")
pilot = Pilot.model_validate(db_record.data)
```

### Anti-Patterns

```python
# ❌ WRONG: Duplicating validation in API layer
class SkillSetInput(BaseModel):
    hull: int = Field(ge=0, le=6)  # Core already does this!

# ❌ WRONG: Recomputing derived values
mech_hp = frame_hp + hull * 2 + grit  # Core's computed_field does this!
```

## Common Commands

```bash
# Testing
make test             # All tests (core + llm + app)
make test-core        # Core type system tests (3277+)
pytest core/pilot/tests/test_skill.py -v  # Single test file

# Development
make install-app      # Install Python + Node dependencies
make db-up            # Start PostgreSQL (Docker, port 5433)
make db-migrate       # Run Alembic migrations
make dev              # Start backend (8000) + frontend (5173)

# Type Generation (run after changing core/ models)
make generate-types   # Python → JSON Schema → TypeScript

# Database
make db-revision MSG="description"  # Create new migration
```

## Key Architectural Patterns

### Typed IDs for Compile-Time Safety
36 typed ID definitions prevent ID mismatches:
```python
from core.shared.ids import PilotId, WeaponId
from core.shared.id_helpers import WeaponIdField  # Coerces "w1" → WeaponId("w1")
```

### API Thin Wrapper Pattern
Endpoints accept raw dicts, delegate to core for validation:
```python
@router.post("")
async def create_pilot(body: dict, session=Depends(get_session)):
    pilot = validate_core_model(Pilot, body, "pilot")
    # ...
```

### JSON Blob Storage
Complex game objects stored as JSONB in PostgreSQL, validated via core on retrieval.

## Change Propagation

When modifying core models:
1. Update core model with tests → `make test-core`
2. Add new models to `core/export.py` EXPORTABLE_MODELS
3. Regenerate types → `make generate-types`
4. Update API/frontend → `make test-app`

## Project-Specific Notes

- PostgreSQL runs on port **5433** (not 5432) to avoid conflicts
- LLM pipeline paths are relative to `llm/` directory
- Backend tests use in-memory SQLite (no Docker needed)
- Frontend types at `app/frontend/src/lib/types/lancer.ts` are auto-generated

## Documentation

- **AGENTS.md**: Detailed architecture patterns and conventions
- **llm/docs/**: LLM pipeline configuration and usage
- **notes/**: Planning documents and implementation details
