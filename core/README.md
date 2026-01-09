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

### Typed ID Conventions

All domain entities use typed IDs for compile-time safety:

```python
from core.shared.ids import PilotId, MechId, WeaponId, CombatantId
from core.shared.id_helpers import CombatantIdField, WeaponIdField

class MountedWeapon(FrozenModel):
    weapon_id: WeaponId  # NewType for static type checking

class AttackResult(FrozenModel):
    attacker_id: CombatantIdField  # Annotated[str] with coercion validator
    target_id: CombatantIdField | None
```

**Pattern**:
- `ids.py`: Define NewType IDs (e.g., `PilotId = NewType("PilotId", str)`)
- `id_helpers.py`: Use `IdField[T]` or convenience aliases for Pydantic model fields
- Backward compatible: `CombatantIdField` accepts strings and coerces to typed ID

This enables type checkers to catch bugs like passing `WeaponId` where `SystemId` is expected.

### Key Modules

- `core/shared/effects.py`: Canonical mechanical building blocks. The
  `MechanicalEffect` container composes all effect primitives.
- `core/shared/enums.py`: Cross-domain literals (actions, damage, status).
- `core/shared/ids.py`: 36 typed ID definitions (PilotId, MechId, WeaponId, etc.)
- `core/shared/id_helpers.py`: IdField[T] pattern with automatic string→typed coercion.
- `core/shared/payloads.py`: Shared area/damage payloads used by pilot gear and mech systems.
- `core/shared/models.py`: Base model for frozen, rules-only data types.

### Adding New Rules

1. Add or extend a typed effect in `core/shared/effects.py` if the mechanic
   is broadly reusable.
2. Add domain-specific models in `core/pilot/` or `core/mech/` when the rule
   is localized to a subsystem.
3. Add validation/tests that prove the new mechanic is wired correctly.
