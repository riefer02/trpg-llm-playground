"""Zone and area effect definitions.

Re-exports zone-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - ZoneEndCondition: Conditions for zone effects to end
    - AttackCaptureEffect: Captures attack in zone
    - ZoneEffect: Creates area effects with specific shapes
    - AreaSelectionEffect: Selects areas for effects

See Also:
    - PR2 3970-3974: Area attacks
"""

from __future__ import annotations

from core.shared.effects.core import (
    ZoneEndCondition,
    AttackCaptureEffect,
    ZoneEffect,
    AreaSelectionEffect,
)

__all__ = [
    "ZoneEndCondition",
    "AttackCaptureEffect",
    "ZoneEffect",
    "AreaSelectionEffect",
]
