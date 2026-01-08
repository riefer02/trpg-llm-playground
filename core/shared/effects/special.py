"""Special combat and interaction effects.

Re-exports special combat effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - TetherEffect: Tethers targets together
    - GrappleEffect: Grapple and throw mechanics
    - SizeInteractionEffect: Size-based interactions
    - MountedAllyEffect: Mounted ally bonuses
    - EffectRemoval: Removes effects from targets
    - HolographicDuplicateEffect: Creates holographic duplicates
    - MovementTrailEffect: Effects along movement path
    - HologramTrailEffect: Hologram effects along path

See Also:
    - PR2 4156-4170: Grapple and shove
    - PR2 4484-4498: Jockey combat
"""

from __future__ import annotations

from core.shared.effects.core import (
    TetherEffect,
    GrappleEffect,
    SizeInteractionEffect,
    MountedAllyEffect,
    EffectRemoval,
    HolographicDuplicateEffect,
    MovementTrailEffect,
    HologramTrailEffect,
)

__all__ = [
    "TetherEffect",
    "GrappleEffect",
    "SizeInteractionEffect",
    "MountedAllyEffect",
    "EffectRemoval",
    "HolographicDuplicateEffect",
    "MovementTrailEffect",
    "HologramTrailEffect",
]
