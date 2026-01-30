"""Down and Out resolution for Lancer pilots.

Implements the Down and Out check when a pilot reaches 0 HP per PR2 3179-3194:
- Roll 1d6: 6 = recover to 1 HP, 2-5 = Down and Out, 1 = dead
- Down and Out: 0 HP, stunned, evasion=5, any damage kills
- Voluntary death: Pilot can choose to die instead of Down and Out
- Rest 1 hour = 1/2 max HP, 10 hours = full HP
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel


DownAndOutOutcome = Literal["recovered", "down_and_out", "died", "voluntary_death"]


class DownAndOutInput(FrozenModel):
    """Input for Down and Out resolution."""

    pilot_id: str = Field(..., description="Unique pilot identifier")
    current_hp: int = Field(..., description="Current HP (0 or negative)")
    max_hp: int = Field(..., gt=0, description="Maximum HP for this pilot")
    base_evasion: int = Field(
        default=10, ge=0, description="Pilot's base evasion before Down and Out"
    )
    voluntary_death: bool = Field(
        default=False,
        description="If True, pilot chooses to die instead of rolling Down and Out",
    )
    damage_dealt: int = Field(
        default=0, ge=0, description="Damage that brought pilot to 0 HP"
    )


class DownAndOutEffect(FrozenModel):
    """An effect applied as part of Down and Out resolution."""

    effect_type: Literal["hp_change", "status_change", "stat_change", "death", "info"]
    description: str = Field(..., description="Human-readable description of effect")
    value: int | None = Field(default=None, description="Numeric value if applicable")


class DownAndOutResult(FrozenModel):
    """Result of Down and Out resolution."""

    outcome: DownAndOutOutcome = Field(..., description="Result of the check")
    roll_result: int | None = Field(
        default=None, description="1d6 roll result (None for voluntary death)"
    )
    hp_after: int = Field(..., description="HP after resolution")
    evasion_after: int = Field(..., description="Evasion after resolution")
    effects: list[DownAndOutEffect] = Field(
        default_factory=list, description="Effects applied during resolution"
    )
    can_be_cloned: bool = Field(
        default=True,
        description="Whether this pilot can be cloned (depends on previous clones)",
    )
    quirk_eligible: bool = Field(
        default=False,
        description="Whether a quirk can be assigned as trauma (Down and Out only)",
    )
    recovery_info: str | None = Field(
        default=None, description="How to recover (rest time, etc)"
    )
    narrative_notes: list[str] = Field(
        default_factory=list, description="Narrative context for GM"
    )


def resolve_down_and_out(
    input_data: DownAndOutInput, roll_result: int | None = None
) -> DownAndOutResult:
    """Resolve a Down and Out check when pilot reaches 0 HP.

    Per PR2 3179-3194:
    - Roll 1d6
    - 6: Recover to 1 HP
    - 2-5: Down and Out (0 HP, stunned, evasion=5)
    - 1: Dead
    - Voluntary death: Can choose to die instead

    Args:
        input_data: Down and Out resolution input
        roll_result: Optional specific roll result for testing (1-6)

    Returns:
        Down and Out result with outcome and effects
    """
    effects: list[DownAndOutEffect] = []
    narrative_notes: list[str] = []

    if input_data.voluntary_death:
        return DownAndOutResult(
            outcome="voluntary_death",
            roll_result=None,
            hp_after=0,
            evasion_after=0,
            effects=[
                DownAndOutEffect(
                    effect_type="death",
                    description="Pilot chose to die rather than be Down and Out",
                    value=None,
                )
            ],
            can_be_cloned=True,
            quirk_eligible=False,
            recovery_info=None,
            narrative_notes=[
                "Pilot voluntarily accepted death",
                "Clone may be available if not previously cloned",
            ],
        )

    if roll_result is None:
        import random

        roll_result = random.randint(1, 6)

    if roll_result == 6:
        effects.append(
            DownAndOutEffect(
                effect_type="hp_change",
                description="Rolled 6 - barely shrug off the hit",
                value=1,
            )
        )
        effects.append(
            DownAndOutEffect(
                effect_type="info",
                description="Return to 1 HP - continue fighting",
                value=None,
            )
        )
        narrative_notes.append("Close call! The pilot shakes off the damage")

        return DownAndOutResult(
            outcome="recovered",
            roll_result=roll_result,
            hp_after=1,
            evasion_after=input_data.base_evasion,
            effects=effects,
            can_be_cloned=True,
            quirk_eligible=False,
            recovery_info="No rest needed - already at 1 HP",
            narrative_notes=narrative_notes,
        )

    if roll_result >= 2 and roll_result <= 5:
        effects.append(
            DownAndOutEffect(
                effect_type="status_change",
                description=f"Rolled {roll_result} - Down and Out",
                value=None,
            )
        )
        effects.append(
            DownAndOutEffect(
                effect_type="info",
                description="Knocked out, pinned, bleeding out",
                value=None,
            )
        )
        effects.append(
            DownAndOutEffect(
                effect_type="info",
                description="Stunned - cannot take actions",
                value=None,
            )
        )
        effects.append(
            DownAndOutEffect(
                effect_type="stat_change",
                description=f"Evasion reduced to 5 (from {input_data.base_evasion})",
                value=5,
            )
        )
        effects.append(
            DownAndOutEffect(
                effect_type="info",
                description="Any additional damage will kill",
                value=None,
            )
        )

        recovery_info = (
            f"Rest 1 hour: recover {input_data.max_hp // 2} HP (½ max); "
            f"Rest 10 hours: recover all {input_data.max_hp} HP"
        )

        narrative_notes.append(f"Rolled {roll_result} - pilot is Down and Out")
        narrative_notes.append("Pilot is bleeding out, unable to act")
        narrative_notes.append("Any further damage will kill the pilot")

        return DownAndOutResult(
            outcome="down_and_out",
            roll_result=roll_result,
            hp_after=0,
            evasion_after=5,
            effects=effects,
            can_be_cloned=True,
            quirk_eligible=True,
            recovery_info=recovery_info,
            narrative_notes=narrative_notes,
        )

    effects.append(
        DownAndOutEffect(
            effect_type="death", description="Rolled 1 - luck has run out", value=None
        )
    )
    effects.append(
        DownAndOutEffect(effect_type="info", description="Pilot has died", value=None)
    )

    narrative_notes.append("Rolled 1 - pilot has died")
    if input_data.max_hp > 0:
        narrative_notes.append(
            "Flash cloning may be available if not previously cloned"
        )

    return DownAndOutResult(
        outcome="died",
        roll_result=roll_result,
        hp_after=0,
        evasion_after=0,
        effects=effects,
        can_be_cloned=True,
        quirk_eligible=False,
        recovery_info=None,
        narrative_notes=narrative_notes,
    )


class PilotRestInput(FrozenModel):
    """Input for pilot rest/recovery."""

    pilot_id: str = Field(..., description="Unique pilot identifier")
    current_hp: int = Field(..., description="Current HP before rest")
    max_hp: int = Field(..., gt=0, description="Maximum HP for this pilot")
    hours_rested: int = Field(..., ge=1, description="Hours of rest taken")
    is_down_and_out: bool = Field(
        default=False, description="Whether pilot is currently Down and Out"
    )


class PilotRestResult(FrozenModel):
    """Result of pilot rest/recovery."""

    hp_before: int = Field(..., description="HP before rest")
    hp_after: int = Field(..., description="HP after rest")
    hp_recovered: int = Field(..., description="Amount of HP recovered")
    is_recovered: bool = Field(
        default=False, description="Whether pilot is now fully recovered"
    )
    down_and_out_cleared: bool = Field(
        default=False, description="Whether Down and Out status is cleared"
    )
    recovery_message: str = Field(..., description="Human-readable result")


def apply_pilot_rest(input_data: PilotRestInput) -> PilotRestResult:
    """Apply rest recovery to a pilot.

    Per PR2 3189-3191:
    - 1 hour rest = ½ max HP
    - 10 hours rest = full HP
    - Rest also clears Down and Out status

    Args:
        input_data: Rest recovery input

    Returns:
        Rest result with HP changes
    """
    hp_before = input_data.current_hp

    if input_data.hours_rested >= 10:
        hp_after = input_data.max_hp
        recovery_message = (
            f"Full rest ({input_data.hours_rested} hours): recovered to {hp_after} HP"
        )
    else:
        half_hp = input_data.max_hp // 2
        hp_after = min(input_data.max_hp, input_data.current_hp + half_hp)
        recovery_message = (
            f"Short rest ({input_data.hours_rested} hour(s)): recovered {hp_after - hp_before} HP "
            f"({hp_before} -> {hp_after}, max {input_data.max_hp})"
        )

    hp_recovered = hp_after - hp_before
    is_recovered = hp_after >= input_data.max_hp
    down_and_out_cleared = input_data.is_down_and_out and hp_after > 0

    if down_and_out_cleared:
        recovery_message += " - Down and Out status cleared!"

    return PilotRestResult(
        hp_before=hp_before,
        hp_after=hp_after,
        hp_recovered=hp_recovered,
        is_recovered=is_recovered,
        down_and_out_cleared=down_and_out_cleared,
        recovery_message=recovery_message,
    )


class PilotDeathResolutionInput(FrozenModel):
    """Input for pilot death resolution (when roll was 1)."""

    pilot_id: str = Field(..., description="Unique pilot identifier")
    death_circumstances: str = Field(
        default="", description="Narrative context of how death occurred"
    )
    has_previous_clone: bool = Field(
        default=False, description="Whether pilot has been cloned before"
    )
    clone_allowed: bool = Field(
        default=True, description="Whether flash cloning is available (GM discretion)"
    )
    party_aware_of_cloning: bool = Field(
        default=False, description="Whether party knows cloning is possible"
    )


class PilotDeathResolutionResult(FrozenModel):
    """Result of pilot death resolution."""

    pilot_id: str = Field(..., description="Resolved pilot ID")
    is_permanent_death: bool = Field(..., description="True if death is permanent")
    clone_available: bool = Field(..., description="True if cloning is still available")
    clone_effects: list[str] = Field(
        default_factory=list, description="Effects if clone is created"
    )
    narrative_outcome: str = Field(
        ..., description="Narrative description of death result"
    )
    gm_notes: list[str] = Field(
        default_factory=list, description="Notes for GM about clone processing"
    )


def resolve_pilot_death(
    input_data: PilotDeathResolutionInput,
) -> PilotDeathResolutionResult:
    """Resolve what happens when a pilot dies.

    Per PR2 3182, 4811-4825:
    - If not previously cloned and clone available: can be flash cloned
    - If previously cloned: permanent death (cannot be cloned again)
    - Clone rewinds pilot to session start but keeps LL advancement
    - Clone always returns with a Quirk

    Args:
        input_data: Death resolution input

    Returns:
        Death resolution result with clone options
    """
    clone_available = not input_data.has_previous_clone and input_data.clone_allowed

    narrative_outcome: str
    clone_effects: list[str] = []
    gm_notes: list[str] = []

    if clone_available:
        narrative_outcome = (
            "Pilot has died. Flash cloning is available - the pilot can be revived, "
            "but will return with complications."
        )
        clone_effects = [
            "Pilot will be revived after mission completion",
            "Pilot 'rewinds' to session start state (HP, triggers, relationships)",
            "Pilot keeps all License Level (LL) advancement",
            "Pilot will be assigned a Quirk (1d20 roll)",
        ]
        gm_notes = [
            "Confirm party is aware of cloning possibility",
            "Roll for Quirk after clone is created",
            "Pilot can only rejoin after current mission ends",
            "This is the pilot's ONE clone - second death is permanent",
        ]
        if not input_data.party_aware_of_cloning:
            gm_notes.append("Party may be surprised by pilot's return")
    else:
        narrative_outcome = "Pilot has died permanently."
        clone_effects = []
        if input_data.has_previous_clone:
            gm_notes.append("Pilot has already been cloned - second death is permanent")
        else:
            gm_notes.append(
                "Cloning is not available (GM discretion or campaign rules)"
            )
        gm_notes.append("Character cannot be revived - new character needed")

    return PilotDeathResolutionResult(
        pilot_id=input_data.pilot_id,
        is_permanent_death=not clone_available,
        clone_available=clone_available,
        clone_effects=clone_effects,
        narrative_outcome=narrative_outcome,
        gm_notes=gm_notes,
    )
