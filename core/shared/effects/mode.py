"""Mode, protocol, and progression effects.

Re-exports mode/progression effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - ModeEffect: Activates or modifies modes
    - ProtocolEffect: Protocol activation and effects
    - CorePowerEffect: Core power activation
    - ProgressionState: Tracks progression states
    - GateProgressionEffect: Gates progression requirements
    - ProgressionEffect: General progression effects

See Also:
    - PR2 4500-4537: Protocols and modes
"""

from __future__ import annotations

from core.shared.effects.core import (
    ModeEffect,
    ProtocolEffect,
    CorePowerEffect,
    ProgressionState,
    GateProgressionEffect,
    ProgressionEffect,
)

__all__ = [
    "ModeEffect",
    "ProtocolEffect",
    "CorePowerEffect",
    "ProgressionState",
    "GateProgressionEffect",
    "ProgressionEffect",
]
