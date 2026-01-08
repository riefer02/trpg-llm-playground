"""Intel and targeting effects.

Re-exports intel and targeting effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - IntelEffect: Grants information about targets
    - TargetMarkEffect: Marks targets for bonuses

See Also:
    - PR2 4050-4059: Scanning and information
"""

from __future__ import annotations

from core.shared.effects.core import (
    IntelEffect,
    TargetMarkEffect,
)

__all__ = [
    "IntelEffect",
    "TargetMarkEffect",
]
