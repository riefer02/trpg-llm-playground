"""Status and condition effects.

Re-exports status-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - StatusGrant: Grants a status or condition
    - StatusClear: Removes a status or condition
    - StatusToggleEffect: Toggleable status at specific timing windows
    - StatusBreakCondition: Breaks a status at specific conditions
    - StatusStackLimit: Limits how many times a status can stack
    - MovementScopedStatus: Status that affects specific movement types
    - StatusRestriction: Restricts specific statuses
    - StatusActionOverrideEffect: Overrides action when statuses are present
    - StatusTrigger: Triggers effects when statuses change

See Also:
    - PR2 3985-4012: Conditions and statuses
"""

from __future__ import annotations

from core.shared.effects.core import (
    StatusGrant,
    StatusClear,
    StatusToggleEffect,
    StatusBreakCondition,
    StatusStackLimit,
    MovementScopedStatus,
    StatusRestriction,
    StatusActionOverrideEffect,
    StatusTrigger,
)

__all__ = [
    "StatusGrant",
    "StatusClear",
    "StatusToggleEffect",
    "StatusBreakCondition",
    "StatusStackLimit",
    "MovementScopedStatus",
    "StatusRestriction",
    "StatusActionOverrideEffect",
    "StatusTrigger",
]
