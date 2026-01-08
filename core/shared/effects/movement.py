"""Movement and positioning effects.

Re-exports movement-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - MovementGrant: Grants movement or teleportation actions
    - MoveAdjacentEffect: Moves source adjacent to a target
    - PositionSwapEffect: Swaps positions between two targets
    - ForcedMovement: Push or pull effects on targets
    - MovementRestrictionEffect: Limits movement types or distance
    - MovementSurfaceEffect: Changes terrain movement rules
    - MovementModeAccessEffect: Grants flight, burrow, or other movement modes
    - JumpDistanceEffect: Modifies jump distances
    - MovementOverrideEffect: Overrides movement mechanics

See Also:
    - PR2 3729-3930: Movement rules
    - PR2 4132-4151: Movement and forced movement effects
"""

from __future__ import annotations

from core.shared.effects.core import (
    MovementGrant,
    MoveAdjacentEffect,
    PositionSwapEffect,
    ForcedMovement,
    MovementRestrictionEffect,
    MovementSurfaceEffect,
    MovementModeAccessEffect,
    JumpDistanceEffect,
    MovementOverrideEffect,
)

__all__ = [
    "MovementGrant",
    "MoveAdjacentEffect",
    "PositionSwapEffect",
    "ForcedMovement",
    "MovementRestrictionEffect",
    "MovementSurfaceEffect",
    "MovementModeAccessEffect",
    "JumpDistanceEffect",
    "MovementOverrideEffect",
]
