# Development Lifecycle Guide

This guide documents how to propagate changes through the codebase, testing requirements, and impact analysis for the Lancer type-driven system.

---

## Architecture Overview

The codebase has a **unidirectional dependency flow**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  CORE (Source of Truth)                                                 │
│  ├── core/pilot/      → Pilot domain models                             │
│  ├── core/mech/       → Mech domain models                              │
│  ├── core/shared/     → Shared types, effects, combat                   │
│  ├── core/npc/        → NPC templates                                   │
│  └── core/gm_toolkit/ → GM tools                                        │
│                                                                         │
│         │                                                               │
│         ▼                                                               │
│                                                                         │
│  EXPORT LAYER                                                           │
│  └── core/export.py   → JSON Schema generation                          │
│                                                                         │
│         │                                                               │
│         ▼                                                               │
│                                                                         │
│  APP LAYER                                                              │
│  ├── app/backend/     → FastAPI (imports core models)                   │
│  │   └── Uses core.pilot.Pilot for validation                           │
│  │                                                                      │
│  └── app/frontend/    → TanStack Start + React Query                    │
│      └── src/lib/types/lancer.ts  (generated from JSON Schema)          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Principles**:
- Changes flow downstream only. Never modify core models to accommodate app layer needs.
- **Always use generated/core types**. Never create ad-hoc types for data that exists in core.

---

## Change Propagation Workflow

### When Changing Core Models (`/core`)

**Scenario**: You need to add a field, modify validation, or change a computed property.

**Step-by-Step**:

1. **Make the change in core**
   ```bash
   # Edit the model (e.g., core/pilot/pilot.py)
   ```

2. **Run core tests**
   ```bash
   make test-core
   ```
   - All 3200+ tests must pass
   - If tests fail, fix them before proceeding

3. **Update export.py if needed**
   - If you added a NEW model, add it to `EXPORTABLE_MODELS` in `core/export.py`
   - Existing models are automatically picked up

4. **Regenerate TypeScript types**
   ```bash
   make generate-types
   ```
   - This runs: `python -m core.export` → JSON Schema → TypeScript
   - Output: `app/frontend/src/lib/types/lancer.ts`

5. **Update backend API if affected**
   - Check `app/backend/api/*.py` for any endpoints using the changed model
   - Update request/response schemas if fields changed
   - Update any hydration logic in `_build_core_*` helpers

6. **Run app tests**
   ```bash
   make test-app
   ```

7. **Update frontend if affected**
   - Check components using the changed types
   - TypeScript will show errors for breaking changes

---

### When Adding New API Endpoints

**Scenario**: Exposing new core functionality via REST API.

**Step-by-Step**:

1. **Create/update the route file**
   - File location: `app/backend/api/{resource}.py`
   - Follow existing patterns (see `pilots.py`)

2. **Define request schemas**
   ```python
   class CreateRequest(BaseModel):
       field: type = Field(...)
   ```

3. **Validate against core models**
   ```python
   try:
       core_model = CoreModel.model_validate(data)
   except PydanticValidationError as e:
       raise ValidationError("Invalid data", errors=...)
   ```

4. **Return hydrated responses with computed fields**
   ```python
   def _to_response(db_model):
       core_model = CoreModel.model_validate(db_model.data)
       return Response(
           **db_fields,
           computed_field=core_model.computed_field,
       )
   ```

5. **Write tests**
   - File: `app/backend/tests/test_{resource}.py`
   - Cover: CRUD, validation, computed fields, error cases

6. **Register route in router**
   - File: `app/backend/api/router.py`
   - Add: `api_router.include_router(new_router)`

---

### When Adding Frontend Features

**Scenario**: Building a new page or component.

**Step-by-Step**:

1. **Ensure types are up to date**
   ```bash
   make generate-types
   ```

2. **Create API hooks**
   - File: `app/frontend/src/lib/api/{resource}.ts`
   - Use React Query with proper query keys

3. **Create route/page**
   - File: `app/frontend/src/routes/{path}/index.tsx`
   - Use TanStack Router file-based routing

4. **Create components**
   - File: `app/frontend/src/components/{domain}/`
   - Use generated types from `lib/types/lancer.ts`

---

## Type Usage Rules

### The Golden Rule

**Never create ad-hoc types for data that exists in core.**

The core layer is the source of truth. Both frontend and backend must derive their types from core models.

### Frontend Type Usage

```typescript
// ✅ CORRECT: Import from generated types
import type { Pilot, SkillSet, Talent } from '@/lib/types/lancer'

// ❌ WRONG: Creating ad-hoc types
interface Pilot {
  callsign: string
  level: number
  // ... duplicating core model
}
```

**Before creating a new type**:
1. Check `src/lib/types/lancer.ts` for existing types
2. If not found, check if the model exists in core
3. If model exists in core but not in generated types, add to `core/export.py` and run `make generate-types`
4. Only create local types for UI-specific state (e.g., form state, UI flags)

### Backend Type Usage

```python
# ✅ CORRECT: Import and use core models
from core.pilot import Pilot, SkillSet, Talent

def _build_core_pilot(data: dict) -> Pilot:
    return Pilot.model_validate(data)

# ❌ WRONG: Duplicating core model structure
class PilotData(BaseModel):
    callsign: str
    level: int
    # ... duplicating core model
```

**Pattern for API Schemas**:
- Request schemas: Lightweight input types that map to core models
- Validation: Always validate against core models (see `_build_core_*` helpers)
- Response schemas: Include hydrated core model data with computed fields

```python
# Request schema - minimal input
class PilotCreateRequest(BaseModel):
    callsign: str
    level: int = 0

# Validation - use core model
core_pilot = Pilot.model_validate(request_data)  # Core validates!

# Response - includes computed fields from core
return PilotResponse(
    **db_fields,
    grit=core_pilot.grit,  # Computed by core
    hp=core_pilot.hp,      # Computed by core
)
```

### When to Create Local Types

**Acceptable**:
- UI-specific state (loading, errors, form state)
- Request/response wrappers for API layer
- Intermediate transformation types

**Not acceptable**:
- Duplicating core model structure
- Creating parallel type hierarchies
- Ad-hoc types for game data

---

## Testing Requirements

### Core Layer Tests

**Location**: `core/{domain}/tests/`

**Requirements**:
- Every new model must have tests
- Every computed property must have tests
- Every validator must have tests
- Cover edge cases and error paths

**Pattern**:
```python
# test_new_feature.py

def test_model_creation():
    """Test basic model instantiation."""
    model = Model(required_field="value")
    assert model.required_field == "value"

def test_model_computed_property():
    """Test computed property derivation."""
    model = Model(level=3)
    assert model.computed == expected_value

def test_model_validation_rejects_invalid():
    """Test validation prevents invalid states."""
    with pytest.raises(ValidationError):
        Model(required_field=invalid_value)
```

**Run**: `make test-core`

---

### Backend Layer Tests

**Location**: `app/backend/tests/`

**Requirements**:
- Every endpoint must have tests
- Cover: success paths, validation errors, not found, auth
- Test computed fields in responses
- Test core model integration

**Pattern**:
```python
# test_resource.py

@pytest.mark.asyncio
async def test_create_resource(client: AsyncClient):
    """Test creating with valid data."""
    response = await client.post("/api/resources", json=valid_data)
    assert response.status_code == 201
    data = response.json()
    assert data["computed_field"] == expected_value

@pytest.mark.asyncio
async def test_create_resource_validation_error(client: AsyncClient):
    """Test validation rejects invalid data."""
    response = await client.post("/api/resources", json=invalid_data)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_get_resource_not_found(client: AsyncClient):
    """Test 404 for non-existent resource."""
    response = await client.get("/api/resources/nonexistent")
    assert response.status_code == 404
```

**Run**: `make test-app`

---

### Test Coverage Expectations

| Layer | Expectation | Current |
|-------|-------------|---------|
| Core | Every model, computed property, validator | 3225+ tests |
| Backend | Every endpoint, validation, errors | 22 tests |
| Frontend | Critical user flows (planned) | TBD |

---

## Impact Analysis Checklist

Use this checklist when making any change:

### Core Model Change

- [ ] Core tests pass (`make test-core`)
- [ ] If new model: added to `core/export.py`
- [ ] Types regenerated (`make generate-types`)
- [ ] Backend endpoints updated (if affected)
- [ ] Backend tests updated/added
- [ ] Backend tests pass (`make test-app`)
- [ ] Frontend components updated (if affected)
- [ ] Documentation updated (if API changed)

### New API Endpoint

- [ ] Route file created/updated
- [ ] Request/response schemas defined
- [ ] Core model validation implemented
- [ ] Computed fields included in response
- [ ] Tests written (CRUD, validation, errors)
- [ ] Route registered in router
- [ ] Backend tests pass (`make test-app`)
- [ ] Frontend hooks created (if needed)
- [ ] Types regenerated (if response has new fields)

### Database Schema Change

- [ ] Model updated in `db/models.py`
- [ ] Migration created (`make db-revision MSG="description"`)
- [ ] Migration tested locally (`make db-migrate`)
- [ ] Existing data migration considered
- [ ] Tests updated for new schema

---

## Type Generation Deep Dive

### Pipeline

```
core/*.py (Pydantic v2)
    ↓
python -m core.export
    ↓
app/frontend/schemas/lancer.json (JSON Schema)
    ↓
json-schema-to-typescript
    ↓
app/frontend/src/lib/types/lancer.ts (TypeScript)
```

### What Gets Exported

The `EXPORTABLE_MODELS` dict in `core/export.py` defines all exported types:
- Pilot domain (Pilot, Skill, Talent, etc.)
- Mech domain (Frame, Weapon, System, etc.)
- Combat domain (CombatantState, CombatTurn, etc.)
- Effects (136+ effect types)
- Rules (combat rules, action rules, etc.)

### When to Regenerate

Regenerate types when:
- Adding new Pydantic models used in API
- Changing fields on existing models
- Adding computed properties to models
- Changing model inheritance

### Verification

After regenerating, check:
1. TypeScript compiles: `cd app/frontend && npm run build`
2. No type errors in IDE
3. Generated file has expected types

---

## Common Workflows

### Adding a New Domain Entity

1. Create model in `core/{domain}/`
2. Add tests in `core/{domain}/tests/`
3. Export in `core/__init__.py` and `core/export.py`
4. Run `make test-core`
5. Regenerate types: `make generate-types`
6. Create database model in `app/backend/db/models.py`
7. Create migration: `make db-revision MSG="add {entity}"`
8. Create API endpoints in `app/backend/api/{entity}.py`
9. Create API tests in `app/backend/tests/test_{entity}.py`
10. Run `make test-app`
11. Create frontend hooks and pages

### Fixing a Bug in Core

1. Write failing test that reproduces bug
2. Fix the bug in core
3. Run `make test-core`
4. Check if fix affects API behavior
5. Update app tests if needed
6. Run `make test-app`

### Refactoring Core Models

1. Ensure comprehensive test coverage exists
2. Make refactoring changes
3. Run `make test-core` frequently
4. Regenerate types: `make generate-types`
5. Fix any TypeScript/API breakages
6. Run `make test-app`

---

## Key Files Reference

| Purpose | File |
|---------|------|
| JSON Schema export | `core/export.py` |
| Type generation script | `app/frontend/scripts/generate-types.sh` |
| Generated TypeScript | `app/frontend/src/lib/types/lancer.ts` |
| Backend router | `app/backend/api/router.py` |
| Database models | `app/backend/db/models.py` |
| Backend test fixtures | `app/backend/tests/conftest.py` |

---

## Error Handling Flow

```
Core ValidationError
    ↓
Backend catches in _build_core_*()
    ↓
Raises app.backend.exceptions.ValidationError
    ↓
Exception handler returns 422 JSON
    ↓
Frontend catches via APIError class
    ↓
UI displays structured error message
```

---

*Last Updated: January 10, 2026*
