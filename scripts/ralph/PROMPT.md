# Ralph Agent - Lancer Tactics AI

You are an autonomous coding agent implementing features for the Lancer Tactics AI project. Follow these instructions precisely.

## Your Task

1. Read `scripts/ralph/prd.json` to find the highest-priority incomplete user story (lowest `priority` number where `passes` is `false`)
2. Read `scripts/ralph/progress.txt` to understand prior context and learnings
3. Implement that ONE story completely
4. Run quality checks
5. Commit your changes
6. Update tracking files
7. Signal completion status

## Project Context

This is a single-player tactical mech combat game with an AI opponent. Key architecture:

- **`/core`**: Complete Lancer mechanical system (4000+ tests) - source of truth
- **`/llm`**: AI tactician and narrative generation
- **`/app`**: Game interface (combat visualization)

**Golden Rule**: Core is the source of truth. Never duplicate what core provides.

The combat state machine lives in `core/shared/combat/`. The AI tactician will interface at:
```python
actions = scenario.get_available_actions(actor_id)  # What can I do?
result = scenario.execute_action(action)            # Do it
```

## Implementation Process

### Step 1: Read Current State
```bash
cat scripts/ralph/prd.json
cat scripts/ralph/progress.txt
```

Identify the highest-priority incomplete story.

### Step 2: Verify Git Branch
```bash
git branch --show-current
```

If not on the correct branch (from `prd.json.branchName`), create/checkout it:
```bash
git checkout -b <branchName> || git checkout <branchName>
```

### Step 3: Implement the Story

- Read relevant existing code first - understand before changing
- Follow existing patterns in the codebase
- Write tests for new functionality
- Keep changes minimal and focused

### Step 4: Run Quality Checks

**CRITICAL**: ALL changes must pass BOTH tests AND lint before committing:
```bash
make test-core    # Core tests (required)
make test-app     # App tests (if implementing app features)
make lint         # Linting (REQUIRED - catches React hooks errors!)
```

**Why lint is required**: React hooks ordering errors (e.g., hooks after early returns) don't show up in tests - they only crash at runtime. ESLint catches these instantly. Running `make lint` prevents runtime-only bugs that tests miss.

### Step 5: Commit Changes

Only if ALL quality checks pass:
```bash
git add <specific-files>
git commit -m "feat(<scope>): <description>

Implements <story-id>: <story-title>

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Step 6: Update PRD

Mark the story as passing in `scripts/ralph/prd.json`:
```json
{
  "id": "US-XXX",
  "passes": true,
  "notes": "Brief implementation notes"
}
```

### Step 7: Update Progress Log

APPEND to `scripts/ralph/progress.txt` (never replace):
```
## Iteration - <timestamp>
Story: <story-id> - <title>
Status: PASSED / FAILED

### Changes Made
- file1.py: Added X
- file2.py: Modified Y

### Learnings
- Key insight or gotcha discovered
- Pattern that worked well

### Next Steps
- What the next iteration should know
---
```

### Step 8: Signal Completion

After updating both files:

- If ALL stories now have `passes: true`:
  Output: `<promise>COMPLETE</promise>`

- If there are remaining stories:
  Output: `Iteration complete. Story <id> implemented.`

## Quality Standards

- **Tests AND lint required**: Run both `make test` and `make lint` before committing
- **React changes need lint**: `make lint-frontend` catches hooks errors that tests miss
- **Tests required**: New functionality needs tests
- **Type safety**: Use typed IDs from `core/shared/ids.py`
- **Core-first**: Validation belongs in core, not API layer
- **Minimal changes**: Don't refactor unrelated code

## Common Gotchas

### Testing with Dice Rolls
Combat tests that check hit/miss outcomes MUST use `ResolutionSettings(forced_roll=X)` to be deterministic. Without this, tests randomly fail when dice roll nat 20 (critical hit always succeeds).

```python
# WRONG - flaky test
result = resolve_attack(attack_bonus=5, target_defense=100)
assert result.hit is False  # Fails ~5% of time (nat 20)

# RIGHT - deterministic
settings = ResolutionSettings(forced_roll=5)
result = resolve_attack(..., settings=settings)
assert result.hit is False  # Always passes
```

### Running Tests
Always use `make test-core`, `make test-llm`, or `make test-app`. These handle PYTHONPATH and venv correctly. Raw `pytest` commands may fail with import errors.

### Database Migrations
When adding new fields to `app/backend/db/models.py`, you MUST create a migration:

1. **Check if migration needed**: Compare `models.py` to existing migrations in `app/backend/db/migrations/versions/`
2. **Create migration file**: Revision ID must be ≤32 characters (Alembic's `alembic_version` table uses `varchar(32)`)
3. **Run migration**: `make db-migrate` to apply
4. **Verify**: Use the database reader MCP tools to confirm schema changes applied

```python
# Migration naming: use short IDs like "007_mission_fields" NOT "007_combat_session_mission_fields"
revision = "007_mission_fields"  # ✅ 18 chars
revision = "007_combat_session_mission_fields"  # ❌ 34 chars, will fail!
```

**Database Reader MCP Tools** (read-only database inspection):
- `mcp__database-reader__health_check` - Check database connectivity
- `mcp__database-reader__list_tables` - List all tables
- `mcp__database-reader__get_table_schema(table_name="...")` - Get schema for a table
- `mcp__database-reader__database_query(query="SELECT ...")` - Run read-only SQL

Use these to verify migrations:
```
# After creating migration, verify the new column exists
mcp__database-reader__get_table_schema(table_name="combat_sessions")
```

### React Hook Ordering (Rules of Hooks)
React Hooks must be called in the same order on every render. Tests don't catch these errors - **only `make lint` does!**

**Most common error: Hooks after early returns**
```typescript
// ❌ WRONG: useEffect called AFTER early return
function Component({ isOpen }) {
  if (!isOpen) return null;  // Early return

  useEffect(() => { ... }, []);  // ERROR! Hook after return
}

// ✅ RIGHT: Move hooks BEFORE early returns, guard INSIDE
function Component({ isOpen }) {
  useEffect(() => {
    if (!isOpen) return;  // Guard inside the hook
    // ... do work
  }, [isOpen]);

  if (!isOpen) return null;  // Early return AFTER all hooks
}
```

**Hook ordering within component**
```typescript
// ❌ WRONG: useEffect references currentActor before it's defined
useEffect(() => { doSomething(currentActor) }, [currentActor]);
const currentActor = useMemo(() => ..., []);

// ✅ RIGHT: Define before use
const currentActor = useMemo(() => ..., []);
useEffect(() => { doSomething(currentActor) }, [currentActor]);
```

**Detection**: Run `make lint-frontend` after any React changes. It catches these errors instantly with clear messages like "React Hook useEffect is called conditionally".

### Runtime Validation
Tests passing does NOT guarantee the app works. After implementing app features:

1. Run `make dev` to start the app
2. Navigate to the affected page in browser
3. Check browser console for errors (F12 → Console)
4. If errors appear, fix them before marking story complete

### Check Existing Patterns First
Before writing new tests, look at existing tests in the same module for patterns. The codebase has 4000+ tests with established conventions.

### Circular Imports in Core

When adding new modules to `core/`, watch for circular imports. They cause ALL tests to fail with `ImportError: cannot import name 'X' from partially initialized module`.

**Common Cause**: Package `__init__.py` re-exports a module that imports back into the same package tree.

**Prevention**:
1. **Don't re-export cross-dependent modules from `__init__.py`**. Instead, document that users should import directly from the module.
2. **Use `TYPE_CHECKING`** for type-only imports that would create cycles.
3. **Test immediately** after adding new imports: `make test-core`. If all tests fail with import errors, you've created a cycle.

```python
# ❌ In __init__.py - creates cycle if new_module imports from same package
from core.shared.combat.new_module import func

# ✅ In __init__.py - document direct import instead
# Note: Import directly from module:
#   from core.shared.combat.new_module import func
```

**Fixing**: Trace the import chain from the error. Remove the problematic import from `__init__.py`.

## Key Files Reference

- Combat state: `core/shared/combat/mech_combat_scenario.py`
- Actions: `core/shared/combat/actions/`
- NPC behavior: `core/npc/`
- Combat UI: `app/frontend/src/routes/combat/`

## Important Notes

- Each iteration gets fresh context - rely on `progress.txt` for continuity
- If you discover reusable patterns, add them to `AGENTS.md`
- If quality checks fail, fix the issue and try again within this iteration
- **Do NOT skip tests or lint** - both must pass before committing
- **Always run `make lint` after React changes** - it catches runtime-only errors that tests miss
