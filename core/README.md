## Core Type System

The `core/` package encodes Lancer mechanics as Pydantic v2 models. These
models are intentionally **typed**, **immutable**, and **flavor-free** so that
other applications (rules engines, builders, validators) can consume them
without parsing natural language.

### Conventions

- Use `FrozenModel` (`core/shared/models.py`) for immutable rule data.
- Use `Literal[...]` for small enums to keep IDE auto-complete strong.
- Prefer shared aliases for repeated patterns (ex: effect targets, uses-per).
- Avoid untyped escape hatches; extend shared effects or add domain models
  when a mechanic doesn't fit existing primitives.
- Avoid flavor text; only encode mechanical rules.

### Key Modules

- `core/shared/effects.py`: Canonical mechanical building blocks. The
  `MechanicalEffect` container composes all effect primitives.
- `core/shared/enums.py`: Cross-domain literals (actions, damage, status).
- `core/shared/payloads.py`: Shared area/damage payloads used by pilot gear and mech systems.
- `core/shared/models.py`: Base model for frozen, rules-only data types.

### Adding New Rules

1. Add or extend a typed effect in `core/shared/effects.py` if the mechanic
   is broadly reusable.
2. Add domain-specific models in `core/pilot/` or `core/mech/` when the rule
   is localized to a subsystem.
3. Add validation/tests that prove the new mechanic is wired correctly.
