# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision

**Lancer Tactics AI** - A single-player tactical mech combat game where you fight against an AI that actually thinks. Voice-controlled, accessible, built on a complete Lancer mechanical system.

**Tagline**: "Your mech. Your voice. An AI that fights back."

See `notes/mission_statement.md` for full vision document.

## Project Structure

Monorepo with three domains:
- **`/core`**: Complete Lancer mechanical system (4000+ tests) - the engine
- **`/llm`**: AI tactician and narrative generation (the brain)
- **`/app`**: Game interface (voice input, combat visualization)

### What We're Building

| Layer | Purpose | Status |
|-------|---------|--------|
| **Core Engine** | All Lancer rules encoded | ✅ Complete (4000+ tests) |
| **AI Tactician** | LLM that reasons about tactics and executes actions | ✅ Complete |
| **Voice Interface** | Speech-to-action, text-to-speech narration | ✅ Complete |
| **Mission Generator** | Procedural objectives with narrative | ✅ Complete |
| **Game Flow** | Title screen, quarters hub, mission select, debrief | ✅ Complete |
| **Combat Polish** | Action preview, confirmations, undo, statistics | ✅ Complete |
| **Visual Polish** | Enemy colors, terrain patterns, help system | 🔲 Next |

### What We're NOT Building

- Multiplayer VTT features
- GM tools for human game masters
- Campaign management for groups
- Features requiring other humans

## Core-First Architecture

**Golden Rule: Core is the source of truth. Never duplicate what core provides.**

```
core/ (Pydantic v2 models) ← All game rules live here
  ↓
llm/ (AI reasoning) ← Uses core to understand valid actions
  ↓
app/ (Interface) ← Renders state, captures input
```

### Correct Patterns

```python
# ✅ RIGHT: Use core validation
from app.backend.utils import validate_core_model
pilot = validate_core_model(Pilot, request_data, "pilot")

# ✅ RIGHT: AI uses core to get valid actions
available = combat_state.get_available_actions(actor_id)
# LLM picks from available, core validates execution
```

### Anti-Patterns

```python
# ❌ WRONG: Duplicating validation
class SkillSetInput(BaseModel):
    hull: int = Field(ge=0, le=6)  # Core already does this!

# ❌ WRONG: AI bypassing core rules
action = llm_picks_action()  # Must validate against core!
```

## Common Commands

```bash
# Testing
make test             # All tests (core + llm + app)
make test-core        # Core mechanical system (4000+ tests)
pytest core/pilot/tests/test_skill.py -v  # Single test file

# Development
make install-app      # Install Python + Node dependencies
make db-up            # Start PostgreSQL (Docker, port 5433)
make db-migrate       # Run Alembic migrations
make dev              # Start backend (8000) + frontend (5173)

# Type Generation (after changing core/ models)
make generate-types   # Python → JSON Schema → TypeScript

# Linting (catches React hooks errors before runtime!)
make lint             # Run all linters
make lint-frontend    # ESLint only (catches hooks ordering issues)
make lint-fix         # Auto-fix what can be fixed

# Autonomous Development (Ralph Loop)
./scripts/ralph/ralph.sh                      # Claude Code, 10 iterations
./scripts/ralph/ralph.sh --tool opencode      # OpenCode (open-source)
./scripts/ralph/ralph.sh --tool opencode --model ollama/llama3  # Local model
```

## Key Patterns

### Typed IDs for Safety
36 typed ID definitions prevent ID mismatches:
```python
from core.shared.ids import PilotId, WeaponId
```

### Combat State Machine
All combat flows through `MechCombatScenario`:
```python
# Get valid actions for current actor
actions = scenario.get_available_actions(actor_id)

# Execute an action (validates internally)
result = scenario.execute_action(action)

# Check for reactions
pending = scenario.get_pending_decisions()
```

### AI Integration Point
The AI tactician will interface at the action level:
```python
# AI sees: board state, available actions, actor capabilities
# AI decides: which action to take, with what parameters
# Core validates: action is legal, resolves effects
```

### Testing Patterns

**Deterministic Combat Tests**
Combat resolution uses dice rolls. To make tests deterministic, always use `ResolutionSettings`:

```python
from core.mech.combat_resolution import ResolutionSettings

# Force a specific d20 roll (e.g., to guarantee a miss)
settings = ResolutionSettings(forced_roll=5)
result = resolve_attack(..., settings=settings)

# Force accuracy/difficulty dice
settings = ResolutionSettings(
    forced_roll=10,
    forced_accuracy_rolls=[6, 4],
    forced_difficulty_rolls=[1, 2],
)
```

**Critical Hit Rule**: Natural 20 always hits regardless of defense. If testing a "miss" scenario, force a low roll (1-19) to avoid flaky tests.

**Environment**: Always use `make test` commands. They handle PYTHONPATH and venv activation. If running pytest manually:
```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest <path> -v
```

## Project-Specific Notes

- PostgreSQL runs on port **5433** (not 5432)
- Backend tests use in-memory SQLite (no Docker needed)
- Frontend types at `app/frontend/src/lib/types/lancer.ts` are auto-generated
- Accessibility is a core requirement (voice control, no fast clicking)

## MCP Tools

### Database Reader
A read-only database MCP tool is available for inspecting the PostgreSQL database state. Use it to:
- Verify migrations applied correctly
- Check table schemas and data
- Debug database-related issues

Available tools:
- `mcp__database-reader__health_check` - Check database connectivity
- `mcp__database-reader__list_tables` - List all tables in the database
- `mcp__database-reader__get_table_schema` - Get schema for a specific table
- `mcp__database-reader__get_all_schemas` - Get schemas for all tables with sample data
- `mcp__database-reader__database_query` - Execute read-only SQL queries

Example usage:
```
# Check if migration added a new column
mcp__database-reader__get_table_schema(table_name="combat_sessions")

# Verify data integrity
mcp__database-reader__database_query(query="SELECT id, mission_id FROM combat_sessions LIMIT 5")
```

## Common Gotchas

### Database Migrations
When adding fields to `app/backend/db/models.py`, you MUST create an Alembic migration or the app will crash. Migration revision IDs must be ≤32 characters (e.g., `007_mission_fields` not `007_combat_session_mission_fields`).

**Verification**: After creating a migration, use the database reader MCP tools to verify the schema changes were applied correctly.

### React Hook Ordering (Rules of Hooks)
React Hooks must be called in the same order on every render. This causes runtime errors that tests don't catch.

**Run `make lint-frontend` to catch these errors before runtime!**

**Most common error: Hooks after early returns**
```typescript
// ❌ WRONG: useEffect called after early return
function MyComponent({ isOpen }) {
  if (!isOpen) return null;  // Early return

  useEffect(() => { ... }, []); // ERROR: Hook after return!
}

// ✅ CORRECT: Move hooks before early returns, guard inside
function MyComponent({ isOpen }) {
  useEffect(() => {
    if (!isOpen) return;  // Guard INSIDE the hook
    // ... do work
  }, [isOpen]);

  if (!isOpen) return null;  // Early return AFTER hooks
}
```

**Hook ordering within component**
```typescript
// ❌ Crashes: useEffect uses currentActor before it's defined
useEffect(() => { ... currentActor?.id ... }, [currentActor?.id]);
const currentActor = useMemo(() => ..., []);

// ✅ Works: Define before use
const currentActor = useMemo(() => ..., []);
useEffect(() => { ... currentActor?.id ... }, [currentActor?.id]);
```

### Runtime Validation
Tests passing ≠ app working. After app changes, run `make dev` and check the browser console for errors.

### Circular Imports in Core

The `core/` module has deep interdependencies. When adding new modules, be careful of circular imports:

**Common Circular Import Pattern**:
```
combat_state.py → new_module.py → combat_models.py → damage.py → combat_state.py
```

**Prevention Strategies**:

1. **Don't re-export cross-dependent modules from `__init__.py`**:
```python
# ❌ WRONG: __init__.py re-exports module that imports from same package
from core.shared.combat.statistics_integration import (...)  # Creates cycle

# ✅ RIGHT: Document that users should import directly
# In __init__.py:
# Note: Import statistics_integration directly:
#   from core.shared.combat.statistics_integration import func
```

2. **Use TYPE_CHECKING for type hints that create cycles**:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.mech.combat_state import MechCombatScenario  # Only for type hints

def my_func(scenario: "MechCombatScenario") -> None:  # String annotation
    ...
```

3. **Move shared types to a lower-level module** that doesn't import from higher-level modules.

**Debugging Circular Imports**:
- Error: `ImportError: cannot import name 'X' from partially initialized module`
- Run `make test-core` - if ALL tests fail with import errors, it's a circular import
- Trace the import chain from the error message
- Fix by removing the cycle (usually via `__init__.py` changes or TYPE_CHECKING)

## Autonomous Development (Ralph)

The project supports autonomous AI development via the Ralph loop pattern. Ralph repeatedly runs an AI coding agent until all PRD items are complete.

### How It Works

1. **PRD** (`scripts/ralph/prd.json`) - Structured user stories with acceptance criteria
2. **Fresh context per iteration** - Each cycle spawns a new AI instance
3. **Memory persistence** - Git commits, `progress.txt`, and PRD status carry state
4. **Quality gates** - Tests must pass before commits

### Files

```
scripts/ralph/
├── ralph.sh       # Main loop script
├── PROMPT.md      # Instructions for each iteration
├── prd.json       # User stories (edit to add features)
├── progress.txt   # Append-only learnings log
└── archive/       # Previous PRD runs (auto-archived)
```

### Usage

```bash
# Claude Code (default)
./scripts/ralph/ralph.sh

# OpenCode (open-source, multi-provider)
./scripts/ralph/ralph.sh --tool opencode

# With specific model
./scripts/ralph/ralph.sh --tool opencode --model anthropic/claude-3-5-sonnet
./scripts/ralph/ralph.sh --tool opencode --model ollama/llama3

# Set max iterations
./scripts/ralph/ralph.sh 20
```

### Adding New Features

1. Edit `scripts/ralph/prd.json` to add user stories
2. Run `./scripts/ralph/ralph.sh`
3. Monitor progress in terminal and `progress.txt`
4. Loop exits when all stories have `passes: true`

## Documentation

- **notes/mission_statement.md**: Project vision and direction
- **notes/planning_next_steps.md**: Current priorities
- **notes/mechanics_coverage_map.md**: What's implemented in core
- **AGENTS.md**: Detailed architecture patterns
- **scripts/ralph/PROMPT.md**: Ralph agent instructions
