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
| **AI Tactician** | LLM that reasons about tactics and executes actions | 🔲 Next |
| **Voice Interface** | Speech-to-action, text-to-speech narration | 🔲 Planned |
| **Mission Generator** | Procedural objectives with narrative | 🔲 Planned |

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

## Project-Specific Notes

- PostgreSQL runs on port **5433** (not 5432)
- Backend tests use in-memory SQLite (no Docker needed)
- Frontend types at `app/frontend/src/lib/types/lancer.ts` are auto-generated
- Accessibility is a core requirement (voice control, no fast clicking)

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
