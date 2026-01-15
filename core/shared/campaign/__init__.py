"""Campaign persistence layer for Lancer TTRPG.

Provides models and utilities for tracking persistent campaign state across sessions,
including characters, mech assignments, mission history, and session metadata.
"""

from core.shared.campaign.campaign import (
    Campaign,
    Session,
    CharacterMechAssignment,
    CampaignMissionRecord,
    ActiveSessionMission,
    MissionPrepPlan,
    CampaignIdentity,
    CampaignLobbyState,
    MissionObjectiveBrief,
    MissionStakesBrief,
    ReservePlanEntry,
    SessionLifecycleCheckpoint,
    MissionOutcomeReport,
)

__all__ = [
    "Campaign",
    "Session",
    "CharacterMechAssignment",
    "CampaignMissionRecord",
    "ActiveSessionMission",
    "MissionPrepPlan",
    "CampaignIdentity",
    "CampaignLobbyState",
    "MissionObjectiveBrief",
    "MissionStakesBrief",
    "ReservePlanEntry",
    "SessionLifecycleCheckpoint",
    "MissionOutcomeReport",
]
