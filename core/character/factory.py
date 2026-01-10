"""Factory functions for creating Lancer characters.

Provides convenience functions for creating valid characters at various
license levels, with sensible defaults and validation.
"""

from __future__ import annotations

from uuid import uuid4

from core.character.character import Character, MechConfiguration
from core.pilot.pilot import Pilot
from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.background import Background
from core.pilot.talent import Talent
from core.mech.build import MechBuild


def _generate_character_id() -> str:
    """Generate a unique character ID."""
    return f"char_{uuid4().hex[:12]}"


def _generate_mech_id() -> str:
    """Generate a unique mech configuration ID."""
    return f"mech_{uuid4().hex[:12]}"


def create_ll0_character(
    callsign: str,
    name: str = "",
    background: Background | None = None,
    skills: SkillSet | None = None,
    triggers: list[PilotTrigger] | None = None,
    talents: list[Talent] | None = None,
    mech_name: str | None = None,
    mech_build: MechBuild | None = None,
    character_id: str | None = None,
    mech_id: str | None = None,
) -> Character:
    """Create a valid License Level 0 character.

    LL0 characters start with:
    - Pilot:
      - 2 mech skill points distributed across HASE
      - 4 triggers at +2 each (8 total points)
      - 3 talents at rank I (3 total points)
      - 0 license points
      - 0 core bonuses
    - Mech:
      - GMS Everest frame only
      - GMS weapons and systems only

    Args:
        callsign: Pilot callsign (required)
        name: Pilot real name (optional)
        background: Pilot background (optional)
        skills: HASE skill allocation (defaults to Hull +2)
        triggers: Pilot triggers (defaults to 4 basic triggers at +2)
        talents: Pilot talents (defaults to 3 rank I talents)
        mech_name: Custom mech name (defaults to callsign)
        mech_build: Mech loadout (defaults to empty Everest)
        character_id: Override character ID
        mech_id: Override mech ID

    Returns:
        A valid LL0 Character

    Raises:
        ValueError: If provided data doesn't meet LL0 requirements
    """
    # Default skills: +2 Hull (the example from the book)
    resolved_skills = skills or SkillSet(hull=2)

    # Validate skill points
    if resolved_skills.total_points() != 2:
        raise ValueError(
            f"LL0 characters must have exactly 2 mech skill points, "
            f"got {resolved_skills.total_points()}"
        )

    # Default triggers: 4 basic triggers at +2
    if triggers is None:
        resolved_triggers = [
            PilotTrigger(trigger_id="assault", rank=2),
            PilotTrigger(trigger_id="survive", rank=2),
            PilotTrigger(trigger_id="spot", rank=2),
            PilotTrigger(trigger_id="take_someone_out", rank=2),
        ]
    else:
        resolved_triggers = triggers

    # Validate triggers
    if len(resolved_triggers) != 4:
        raise ValueError(
            f"LL0 characters must have exactly 4 triggers, "
            f"got {len(resolved_triggers)}"
        )
    trigger_total = sum(t.rank for t in resolved_triggers)
    if trigger_total != 8:
        raise ValueError(
            f"LL0 characters must have 8 trigger points (4 triggers at +2), "
            f"got {trigger_total}"
        )

    # Default talents: 3 rank I talents
    if talents is None:
        resolved_talents = [
            Talent(talent_id="ace", rank=1),
            Talent(talent_id="combined_arms", rank=1),
            Talent(talent_id="crack_shot", rank=1),
        ]
    else:
        resolved_talents = talents

    # Validate talents
    talent_total = sum(t.rank for t in resolved_talents)
    if talent_total != 3:
        raise ValueError(
            f"LL0 characters must have exactly 3 talent points, "
            f"got {talent_total}"
        )
    if any(t.rank != 1 for t in resolved_talents):
        raise ValueError("LL0 characters can only have rank I talents")

    # Create pilot
    pilot = Pilot(
        id=f"pilot_{uuid4().hex[:12]}",
        callsign=callsign,
        name=name,
        background=background,
        level=0,
        skills=resolved_skills,
        triggers=resolved_triggers,
        talents=resolved_talents,
        licenses=[],
        core_bonuses=[],
    )

    # Default mech: GMS Everest with no loadout
    resolved_mech_name = mech_name or callsign
    resolved_mech_build = mech_build or MechBuild(frame_id="gms_everest")

    # Validate frame is GMS Everest
    if resolved_mech_build.frame_id != "gms_everest":
        raise ValueError(
            f"LL0 characters can only use GMS Everest frame, "
            f"got '{resolved_mech_build.frame_id}'"
        )

    mech = MechConfiguration(
        id=mech_id or _generate_mech_id(),
        name=resolved_mech_name,
        frame_id="gms_everest",
        build=resolved_mech_build,
    )

    return Character(
        id=character_id or _generate_character_id(),
        pilot=pilot,
        mechs=[mech],
        active_mech_id=mech.id,
    )


def create_empty_character(
    callsign: str,
    name: str = "",
    level: int = 0,
    character_id: str | None = None,
) -> Character:
    """Create a minimal character with just a pilot (no mechs).

    Useful for testing or as a starting point for manual configuration.
    Note: This character will have validation warnings for missing
    mech skill points, triggers, talents, and mechs.

    Args:
        callsign: Pilot callsign (required)
        name: Pilot real name (optional)
        level: License level (defaults to 0)
        character_id: Override character ID

    Returns:
        A minimal Character with only pilot data
    """
    pilot = Pilot(
        id=f"pilot_{uuid4().hex[:12]}",
        callsign=callsign,
        name=name,
        level=level,
        skills=SkillSet(),
        triggers=[],
        talents=[],
        licenses=[],
        core_bonuses=[],
    )

    return Character(
        id=character_id or _generate_character_id(),
        pilot=pilot,
        mechs=[],
        active_mech_id=None,
    )
