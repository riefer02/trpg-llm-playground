"""Campaign persistence layer for Lancer TTRPG.

Provides models and utilities for tracking persistent campaign state across sessions,
including pilots, mech assignments, mission history, and session metadata.
"""

from core.shared.campaign.campaign import (
    Campaign,
    Session,
    PilotMechAssignment,
    CampaignMissionRecord,
    ActiveSessionMission,
)

__all__ = [
    "Campaign",
    "Session",
    "PilotMechAssignment",
    "CampaignMissionRecord",
    "ActiveSessionMission",
]
