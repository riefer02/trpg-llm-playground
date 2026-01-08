"""Cover and line of sight effects.

Re-exports cover-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - CoverGrant: Grants soft or hard cover
    - CoverRestriction: Restricts cover benefits
    - LineOfSightRestriction: Affects line of sight

See Also:
    - PR2 3816-3817: Cover rules
"""

from __future__ import annotations

from core.shared.effects.core import (
    CoverGrant,
    CoverRestriction,
    LineOfSightRestriction,
)

__all__ = [
    "CoverGrant",
    "CoverRestriction",
    "LineOfSightRestriction",
]
