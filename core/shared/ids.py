"""Typed ID definitions for Lancer TTRPG.

This module provides NewType definitions for all ID types used across the codebase.
Using typed IDs enables type checkers (mypy/pyright) to catch ID mismatches at
compile time, preventing bugs like passing a WeaponId where a SystemId is expected.

Usage:
    from core.shared.ids import PilotId, MechId, WeaponId

    def equip_weapon(mech_id: MechId, weapon_id: WeaponId) -> None:
        ...

    # Type checker will catch this error:
    equip_weapon(weapon_id=WeaponId("w1"), mech_id=SystemId("s1"))  # Error!
"""

from typing import NewType

__all__ = [
    # Entity IDs
    "PilotId",
    "MechId",
    "CharacterId",
    "NpcId",
    "CombatantId",
    "ActorId",
    "EntityId",
    # Equipment IDs
    "FrameId",
    "WeaponId",
    "SystemId",
    "LicenseId",
    "TalentId",
    "CoreBonusId",
    # Combat IDs
    "ActionId",
    "EffectId",
    "StatusId",
    "ProtocolId",
    "TriggerId",
    "ReactionId",
    # Game Object IDs
    "DroneId",
    "DeployableId",
    "ObjectId",
    "ZoneId",
    # Mission/Campaign IDs
    "MissionId",
    "ObjectiveId",
    "SessionId",
    "SceneId",
    "ConsequenceId",
    # NPC IDs
    "TemplateId",
    # Initiative IDs
    "NominatorId",
    "NomineeId",
]

# -----------------------------------------------------------------------------
# Entity IDs - Characters and combatants
# -----------------------------------------------------------------------------

PilotId = NewType("PilotId", str)
"""Unique identifier for a pilot character."""

MechId = NewType("MechId", str)
"""Unique identifier for a mech unit."""

CharacterId = NewType("CharacterId", str)
"""Unique identifier for a character (pilot + mechs)."""

NpcId = NewType("NpcId", str)
"""Unique identifier for an NPC."""

CombatantId = NewType("CombatantId", str)
"""Generic identifier for any combat participant (mech, pilot, NPC, drone)."""

ActorId = NewType("ActorId", str)
"""Generic identifier for any actor taking actions."""

EntityId = NewType("EntityId", str)
"""Generic identifier for any game entity."""

# -----------------------------------------------------------------------------
# Equipment IDs - Gear and loadout components
# -----------------------------------------------------------------------------

FrameId = NewType("FrameId", str)
"""Unique identifier for a mech frame definition."""

WeaponId = NewType("WeaponId", str)
"""Unique identifier for a weapon definition or instance."""

SystemId = NewType("SystemId", str)
"""Unique identifier for a mech system definition or instance."""

LicenseId = NewType("LicenseId", str)
"""Unique identifier for a manufacturer license."""

TalentId = NewType("TalentId", str)
"""Unique identifier for a pilot talent."""

CoreBonusId = NewType("CoreBonusId", str)
"""Unique identifier for a core bonus."""

# -----------------------------------------------------------------------------
# Combat IDs - Actions and effects
# -----------------------------------------------------------------------------

ActionId = NewType("ActionId", str)
"""Unique identifier for an action definition."""

EffectId = NewType("EffectId", str)
"""Unique identifier for an effect instance."""

StatusId = NewType("StatusId", str)
"""Unique identifier for a status/condition."""

ProtocolId = NewType("ProtocolId", str)
"""Unique identifier for a protocol."""

TriggerId = NewType("TriggerId", str)
"""Unique identifier for a trigger."""

ReactionId = NewType("ReactionId", str)
"""Unique identifier for a reaction."""

# -----------------------------------------------------------------------------
# Game Object IDs - Deployables and battlefield objects
# -----------------------------------------------------------------------------

DroneId = NewType("DroneId", str)
"""Unique identifier for a drone."""

DeployableId = NewType("DeployableId", str)
"""Unique identifier for a deployable object."""

ObjectId = NewType("ObjectId", str)
"""Unique identifier for a battlefield object."""

ZoneId = NewType("ZoneId", str)
"""Unique identifier for a zone on the battlefield."""

# -----------------------------------------------------------------------------
# Mission/Campaign IDs - Session and scenario tracking
# -----------------------------------------------------------------------------

MissionId = NewType("MissionId", str)
"""Unique identifier for a mission."""

ObjectiveId = NewType("ObjectiveId", str)
"""Unique identifier for a mission objective."""

SessionId = NewType("SessionId", str)
"""Unique identifier for a game session."""

SceneId = NewType("SceneId", str)
"""Unique identifier for a narrative scene."""

ConsequenceId = NewType("ConsequenceId", str)
"""Unique identifier for a narrative consequence."""

# -----------------------------------------------------------------------------
# NPC IDs
# -----------------------------------------------------------------------------

TemplateId = NewType("TemplateId", str)
"""Unique identifier for an NPC template."""

# -----------------------------------------------------------------------------
# Initiative IDs - Turn order tracking
# -----------------------------------------------------------------------------

NominatorId = NewType("NominatorId", str)
"""ID of the actor nominating the next turn."""

NomineeId = NewType("NomineeId", str)
"""ID of the actor being nominated to act."""
