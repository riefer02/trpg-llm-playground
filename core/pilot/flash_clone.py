"""Flash cloning process for Lancer pilots.

Implements flash cloning mechanics per PR2 4789-4820:
- Only once per character (second death = permanent death)
- Clone "rewinds" to session start but keeps LL advancement
- Clone always returns with a Quirk (1d20)
- Can only rejoin after mission completion
"""

from pydantic import Field
from core.shared.models import FrozenModel
from core.pilot.clone_state import CloneState, CloneStatus, Quirk


class PilotSnapshot(FrozenModel):
    """Snapshot of pilot state at session start for clone rewind.

    Per PR2 4815-4818:
    - Clone "rewinds" to character state at beginning of session
    - Keeps License Level (LL) advancement
    - HP, triggers, and relationships reset to session start
    """

    hp: int = Field(..., gt=0, description="HP at session start")
    evasion: int = Field(..., ge=0, description="Evasion at session start")
    triggers_snapshot: list[dict] = Field(
        default_factory=list, description="Serialized trigger state at session start"
    )
    relationships_snapshot: list[dict] = Field(
        default_factory=list,
        description="Serialized relationship state at session start",
    )


class FlashCloneInput(FrozenModel):
    """Input for flash clone creation."""

    original_pilot_id: str = Field(..., description="ID of pilot being cloned")
    session_snapshot: PilotSnapshot = Field(
        ..., description="Session start snapshot for rewind"
    )
    current_level: int = Field(
        ..., ge=0, le=12, description="Current License Level (preserved)"
    )
    current_skills: dict = Field(..., description="Current skill state (preserved)")
    current_talents: list[dict] = Field(
        ..., description="Current talent ranks (preserved)"
    )
    current_licenses: list[dict] = Field(
        ..., description="Current license progress (preserved)"
    )
    current_core_bonuses: list[str] = Field(
        ..., description="Current core bonuses (preserved)"
    )
    current_gear: list[str] = Field(
        default_factory=list, description="Current gear loadout (preserved)"
    )
    clone_allowed: bool = Field(
        default=True, description="Whether cloning is allowed (GM discretion)"
    )
    party_aware: bool = Field(
        default=False, description="Whether party knows about cloning"
    )


class FlashCloneResult(FrozenModel):
    """Result of flash clone creation."""

    success: bool = Field(..., description="Whether clone was successful")
    new_pilot_id: str = Field(..., description="ID of the new clone pilot")
    clone_count: int = Field(..., description="Times cloned (now 1)")
    quirk: Quirk | None = Field(
        default=None, description="Quirk assigned to this clone"
    )
    rewind_applied: bool = Field(
        default=False, description="Whether session rewind was applied"
    )
    session_hp: int = Field(..., description="HP after session rewind")
    session_evasion: int = Field(..., description="Evasion after session rewind")
    level_preserved: int = Field(..., description="License Level preserved in clone")
    can_rejoin_after_mission: bool = Field(
        default=True, description="Whether clone can rejoin after current mission"
    )
    narrative_effects: list[str] = Field(
        default_factory=list, description="Narrative effects of the clone"
    )
    gm_notes: list[str] = Field(
        default_factory=list, description="Notes for GM about this clone"
    )
    failure_reason: str | None = Field(
        default=None, description="Reason for clone failure if not successful"
    )


def can_be_cloned(clone_state: CloneState | None) -> bool:
    """Check if a pilot can be cloned.

    Args:
        clone_state: Current clone state (None if not tracked)

    Returns:
        True if pilot can be cloned
    """
    if clone_state is None:
        return True
    return clone_state.can_be_cloned


def create_flash_clone(input_data: FlashCloneInput, quirk: Quirk) -> FlashCloneResult:
    """Create a flash clone of a dead pilot.

    Per PR2 4811-4825:
    - Clone can only rejoin after mission completion
    - Clone rewinds to session start state
    - Clone keeps all LL advancement
    - Clone always returns with a Quirk
    - This is the pilot's ONE clone

    Args:
        input_data: Flash clone input
        quirk: Quirk to assign to this clone (from 1d20 roll)

    Returns:
        Flash clone result with new pilot state
    """
    if not input_data.clone_allowed:
        return FlashCloneResult(
            success=False,
            new_pilot_id="",
            clone_count=0,
            quirk=None,
            rewind_applied=False,
            session_hp=0,
            session_evasion=0,
            level_preserved=input_data.current_level,
            can_rejoin_after_mission=False,
            narrative_effects=[],
            gm_notes=["Cloning is not available (GM discretion)"],
            failure_reason="Cloning not allowed by GM",
        )

    narrative_effects: list[str] = []
    gm_notes: list[str] = []

    narrative_effects.append(
        "Pilot has been flash cloned. The clone remembers suffering a major injury "
        "but will have to rebuild relationships from the start of this session."
    )
    narrative_effects.append(
        f"Despite the trauma, the clone retains all License Level {input_data.current_level} advancement."
    )
    narrative_effects.append(f"The clone has been marked by a Quirk: {quirk.name}")

    if not input_data.party_aware:
        gm_notes.append(
            "Party was unaware of cloning - the pilot's return will be a surprise"
        )

    gm_notes.append(f"Clone Quirk (1d20={quirk.roll}): {quirk.name}")
    gm_notes.append(f"Quirk description: {quirk.description}")
    gm_notes.append("This is the pilot's ONLY clone - second death is permanent")

    new_pilot_id = f"{input_data.original_pilot_id}_clone"

    return FlashCloneResult(
        success=True,
        new_pilot_id=new_pilot_id,
        clone_count=1,
        quirk=quirk,
        rewind_applied=True,
        session_hp=input_data.session_snapshot.hp,
        session_evasion=input_data.session_snapshot.evasion,
        level_preserved=input_data.current_level,
        can_rejoin_after_mission=True,
        narrative_effects=narrative_effects,
        gm_notes=gm_notes,
        failure_reason=None,
    )


class CloneSessionRewindInput(FrozenModel):
    """Input for applying session rewind to a cloned pilot."""

    pilot_level: int = Field(..., ge=0, le=12, description="License Level to preserve")
    pilot_skills: dict = Field(..., description="Skills to preserve")
    pilot_talents: list[dict] = Field(..., description="Talents to preserve")
    pilot_licenses: list[dict] = Field(..., description="Licenses to preserve")
    pilot_core_bonuses: list[str] = Field(..., description="Core bonuses to preserve")
    pilot_gear: list[str] = Field(default_factory=list, description="Gear to preserve")
    session_hp: int = Field(..., gt=0, description="Session start HP")
    session_evasion: int = Field(..., ge=0, description="Session start evasion")
    session_triggers: list[dict] = Field(
        default_factory=list, description="Session start triggers"
    )


class CloneSessionRewindResult(FrozenModel):
    """Result of applying session rewind to a cloned pilot."""

    hp: int = Field(..., description="HP after rewind")
    evasion: int = Field(..., description="Evasion after rewind")
    level_preserved: int = Field(..., description="License Level preserved")
    skills_preserved: bool = Field(..., description="Whether skills were preserved")
    talents_preserved: int = Field(..., description="Number of talent ranks preserved")
    licenses_preserved: int = Field(..., description="Number of licenses preserved")
    core_bonuses_preserved: int = Field(
        ..., description="Number of core bonuses preserved"
    )
    gear_preserved: int = Field(..., description="Number of gear items preserved")
    narrative_summary: str = Field(..., description="Human-readable summary")


def apply_session_rewind(
    input_data: CloneSessionRewindInput,
) -> CloneSessionRewindResult:
    """Apply session rewind to a cloned pilot.

    Per PR2 4815-4818:
    - Clone rewinds to session start state (HP, triggers, relationships)
    - Clone keeps all License Level (LL) advancement
    - All progression beyond session start is preserved

    Args:
        input_data: Session rewind input

    Returns:
        Session rewind result with preserved and reset values
    """
    talents_count = sum(t.get("rank", 1) for t in input_data.pilot_talents)
    licenses_count = len(input_data.pilot_licenses)
    core_bonuses_count = len(input_data.pilot_core_bonuses)
    gear_count = len(input_data.pilot_gear)

    narrative_summary = (
        f"Clone rewind complete:\n"
        f"  - HP reset to session start: {input_data.session_hp}\n"
        f"  - Evasion reset to session start: {input_data.session_evasion}\n"
        f"  - License Level {input_data.pilot_level} preserved\n"
        f"  - {talents_count} talent ranks preserved\n"
        f"  - {licenses_count} licenses preserved\n"
        f"  - {core_bonuses_count} core bonuses preserved\n"
        f"  - {gear_count} gear items preserved"
    )

    return CloneSessionRewindResult(
        hp=input_data.session_hp,
        evasion=input_data.session_evasion,
        level_preserved=input_data.pilot_level,
        skills_preserved=True,
        talents_preserved=talents_count,
        licenses_preserved=licenses_count,
        core_bonuses_preserved=core_bonuses_count,
        gear_preserved=gear_count,
        narrative_summary=narrative_summary,
    )


class SecondCloneCheckInput(FrozenModel):
    """Input for checking if second clone is possible."""

    pilot_id: str = Field(..., description="Pilot being checked")
    has_previous_clone: bool = Field(..., description="Whether pilot was cloned before")
    current_clone_state: CloneStatus | None = Field(
        default=None, description="Current clone status"
    )


class SecondCloneCheckResult(FrozenModel):
    """Result of second clone eligibility check."""

    can_be_cloned: bool = Field(..., description="Whether clone is allowed")
    is_permanent_death: bool = Field(..., description="Whether this is permanent death")
    narrative_reason: str = Field(..., description="Explanation of result")
    gm_guidance: list[str] = Field(default_factory=list, description="Guidance for GM")


def check_second_clone_eligibility(
    input_data: SecondCloneCheckInput,
) -> SecondCloneCheckResult:
    """Check if a pilot can be cloned a second time.

    Per PR2 4823-4825:
    - "If a cloned character would be cloned a second time, they can no longer
      be played as a player character. The trauma from being brought 'back to life'
      is too great. In other words, you're one and done."

    Args:
        input_data: Second clone check input

    Returns:
        Second clone eligibility result
    """
    can_be_cloned = not input_data.has_previous_clone

    if can_be_cloned:
        return SecondCloneCheckResult(
            can_be_cloned=True,
            is_permanent_death=False,
            narrative_reason="Pilot has not been cloned before - clone available",
            gm_guidance=[
                "Pilot is eligible for their first (and only) flash clone",
                "After cloning, pilot will have 1 clone count",
                "Second death will be permanent death",
            ],
        )
    else:
        return SecondCloneCheckResult(
            can_be_cloned=False,
            is_permanent_death=True,
            narrative_reason=(
                "Pilot has already been cloned once. The trauma of being brought "
                "back to life a second time would be too great - this is permanent death."
            ),
            gm_guidance=[
                "Pilot has exceeded their one-time clone limit",
                "Character cannot be revived - new character needed",
                "This is a significant narrative moment - handle with appropriate weight",
            ],
        )
