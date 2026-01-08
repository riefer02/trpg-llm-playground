"""Deployment and attachment effects.

Re-exports deployment-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - DeploymentEffect: Deploys drones, turrets, or other entities
    - AttachmentEffect: Attaches entities to targets
    - SystemLinkEffect: Links systems for combined effects

See Also:
    - PR2 5070-5088: Drones and deployables
"""

from __future__ import annotations

from core.shared.effects.core import (
    DeploymentEffect,
    AttachmentEffect,
    SystemLinkEffect,
)

__all__ = [
    "DeploymentEffect",
    "AttachmentEffect",
    "SystemLinkEffect",
]
