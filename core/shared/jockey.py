"""Jockey action resolution primitives for Lancer TTRPG.

Implements resolution logic for the Jockey full action per PR2 4446-4458:

Jockey Mechanics:
- Full action to attempt jockeying an enemy mech
- Must be adjacent to the target mech
- Contested check: pilot GRIT vs mech HULL
- On success: pilot rides the mech, sharing its space and moving when it moves
- Mech can shake off by winning contested check as full action
- Pilot can jump off as part of movement any time
- On successful jockey turn + free follow-up, repeat 1 option/turn as full action

Follow-up Options:
- Distract: Inflict Impaired + Slowed until end of target's next turn
- Shred: Deal 2 heat to target
- Damage: Deal 4 kinetic damage to target

Resolution Pattern:
1. resolve_jockey() - Pure resolution logic
2. apply_jockey_result() - Apply to combatant state
3. resolve_shake_off() - Mech shake-off contested check
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType, DamageType
from core.mech.combat_state import CombatantState
from core.mech.grid import HexPosition
from core.mech.combat_resolution import DiceRollResult, ResolutionSettings


JockeyOptionType = Literal["distract", "shred", "damage"]


class JockeyOption(FrozenModel):
    """A jockey follow-up option."""

    option_type: JockeyOptionType
    inflicted_conditions: list[StatusType] = Field(default_factory=list)
    heat: int = Field(default=0, ge=0)
    damage: int = Field(default=0, ge=0)
    damage_type: DamageType | None = None


DEFAULT_JOCKEY_OPTIONS: list[JockeyOption] = [
    JockeyOption(
        option_type="distract",
        inflicted_conditions=["impaired", "slowed"],
    ),
    JockeyOption(
        option_type="shred",
        heat=2,
    ),
    JockeyOption(
        option_type="damage",
        damage=4,
        damage_type="kinetic",
    ),
]


class JockeyRule(FrozenModel):
    """Rule configuration for Jockey action."""

    contested_check_stat_attacker: Literal["grit", "skill_check"] = "grit"
    contested_check_stat_defender: Literal["hull"] = "hull"
    tie_breaker: Literal["attacker", "defender"] = "attacker"
    options: list[JockeyOption] = Field(default_factory=lambda: DEFAULT_JOCKEY_OPTIONS)
    free_option_on_success: bool = True
    repeat_one_option_per_turn: bool = True


DEFAULT_JOCKEY_RULES = JockeyRule()


class JockeyInput(FrozenModel):
    """Input for Jockey action resolution."""

    actor_id: str = Field(..., description="ID of the pilot attempting to jockey")
    target_mech_id: str = Field(..., description="ID of the mech being jockeyed")
    chosen_option: JockeyOptionType = Field(
        ..., description="Which follow-up option to use"
    )
    rules: JockeyRule | None = Field(
        default=None, description="Override resolution rules"
    )
    settings: ResolutionSettings | None = Field(
        default=None, description="Optional resolution settings for forced rolls"
    )


class ShakeOffInput(FrozenModel):
    """Input for Mech Shake Off action."""

    mech_id: str = Field(
        ..., description="ID of the mech attempting to shake off rider"
    )
    rider_id: str = Field(..., description="ID of the pilot being shaken off")
    rules: JockeyRule | None = Field(
        default=None, description="Override resolution rules"
    )
    settings: ResolutionSettings | None = Field(
        default=None, description="Optional resolution settings for forced rolls"
    )


class JockeyContestedResult(FrozenModel):
    """Result of the contested check for Jockey."""

    attacker_roll: DiceRollResult | None = Field(
        default=None, description="Pilot's contested roll result"
    )
    defender_roll: DiceRollResult | None = Field(
        default=None, description="Mech's contested roll result"
    )
    attacker_total: int | None = Field(default=None, description="Pilot's total (grit)")
    defender_total: int | None = Field(default=None, description="Mech's total (hull)")
    attacker_wins: bool = Field(
        default=False, description="Whether the attacker won the contest"
    )


class JockeyResolutionResult(FrozenModel):
    """Complete result of Jockey resolution (pure logic)."""

    actor_id: str = Field(..., description="ID of the pilot")
    target_mech_id: str = Field(..., description="ID of the mech being jockeyed")
    contested_result: JockeyContestedResult | None = Field(
        default=None, description="Result of the contested check"
    )
    chosen_option: JockeyOption | None = Field(
        default=None, description="The option being applied"
    )
    jockey_success: bool = Field(
        default=False, description="Whether the jockey attempt succeeded"
    )
    conditions_inflicted: list[StatusType] = Field(
        default_factory=list, description="Conditions inflicted on target"
    )
    heat_dealt: int = Field(default=0, ge=0, description="Heat dealt to target")
    damage_dealt: int | None = Field(default=None, description="Damage dealt to target")
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class ShakeOffResolutionResult(FrozenModel):
    """Complete result of Shake Off resolution (pure logic)."""

    mech_id: str = Field(..., description="ID of the mech")
    rider_id: str = Field(..., description="ID of the pilot")
    contested_result: JockeyContestedResult | None = Field(
        default=None, description="Result of the contested check"
    )
    shake_off_success: bool = Field(
        default=False, description="Whether the mech successfully shook off the rider"
    )
    rider_ejected: bool = Field(
        default=False, description="Whether the rider was ejected from the mech"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class JockeyApplicationResult(FrozenModel):
    """Result of applying Jockey result to combatant state."""

    updated_target: CombatantState = Field(
        ..., description="Target mech with jockey effects applied"
    )
    jockey_success: bool = Field(
        default=False, description="Whether jockey was successful"
    )
    conditions_applied: list[StatusType] = Field(
        default_factory=list, description="Conditions that were applied"
    )
    heat_dealt: int = Field(default=0, ge=0, description="Heat that was dealt")
    damage_dealt: int | None = Field(default=None, description="Damage that was dealt")


class ShakeOffApplicationResult(FrozenModel):
    """Result of applying Shake Off result to combatant state."""

    updated_mech: CombatantState = Field(
        ..., description="Mech with shake off result applied"
    )
    updated_rider: CombatantState | None = Field(
        default=None, description="Rider after being shaken off"
    )
    shake_off_success: bool = Field(
        default=False, description="Whether shake off succeeded"
    )
    rider_ejected: bool = Field(
        default=False, description="Whether rider was ejected to adjacent space"
    )


def _roll_contested_check(
    attacker_value: int,
    defender_value: int,
    settings: ResolutionSettings | None = None,
    attacker_name: str = "attacker",
    defender_name: str = "defender",
) -> tuple[list[int], list[int], int, int]:
    """Roll a contested check for jockey/shake-off.

    Args:
        attacker_value: Attacker's stat value (grit for pilot)
        defender_value: Defender's stat value (hull for mech)
        settings: Optional resolution settings for forced rolls
        attacker_name: Name for debugging
        defender_name: Name for debugging

    Returns:
        Tuple of (attacker_rolls, defender_rolls, attacker_total, defender_total)
    """
    from core.shared.dice import DiceExpression

    attacker_dice = DiceExpression.parse("1d6").roll()
    if settings and settings.forced_rolls:
        attacker_dice = list(settings.forced_rolls[:1])

    defender_dice = DiceExpression.parse("1d6").roll()
    if settings and settings.forced_rolls and len(settings.forced_rolls) > 1:
        defender_dice = list(settings.forced_rolls[1:2])

    attacker_total = sum(attacker_dice) + attacker_value
    defender_total = sum(defender_dice) + defender_value

    return attacker_dice, defender_dice, attacker_total, defender_total


def _get_option_by_type(
    options: list[JockeyOption], option_type: JockeyOptionType
) -> JockeyOption | None:
    """Get a jockey option by type."""
    for option in options:
        if option.option_type == option_type:
            return option
    return None


def resolve_jockey(
    input: JockeyInput,
    pilot_grit: int,
    mech_hull: int,
    is_adjacent: bool = True,
) -> JockeyResolutionResult:
    """Resolve a Jockey action per PR2 4446-4458.

    Jockey is a Full Action that allows a pilot to ride an enemy mech.
    Requires adjacency and a contested GRIT vs HULL check.

    Args:
        input: Jockey input with pilot, target, and option information
        pilot_grit: The pilot's grit value
        mech_hull: The mech's hull value
        is_adjacent: Whether pilot is adjacent to target mech

    Returns:
        Detailed breakdown of what should happen during jockeying
    """
    if input.rules is None:
        rules = DEFAULT_JOCKEY_RULES
    else:
        rules = input.rules

    errors: list[str] = []

    if not is_adjacent:
        errors.append("Pilot must be adjacent to the mech to jockey it")

    option = _get_option_by_type(rules.options, input.chosen_option)
    if option is None:
        errors.append(f"Unknown jockey option: {input.chosen_option}")

    if errors:
        return JockeyResolutionResult(
            actor_id=input.actor_id,
            target_mech_id=input.target_mech_id,
            contested_result=None,
            chosen_option=None,
            jockey_success=False,
            conditions_inflicted=[],
            heat_dealt=0,
            damage_dealt=None,
            validation_errors=errors,
        )

    if option is None:
        errors.append(f"Unknown jockey option: {input.chosen_option}")
        return JockeyResolutionResult(
            actor_id=input.actor_id,
            target_mech_id=input.target_mech_id,
            contested_result=None,
            chosen_option=None,
            jockey_success=False,
            conditions_inflicted=[],
            heat_dealt=0,
            damage_dealt=None,
            validation_errors=errors,
        )

    attacker_rolls, defender_rolls, attacker_total, defender_total = (
        _roll_contested_check(
            pilot_grit,
            mech_hull,
            input.settings,
            "pilot",
            "mech",
        )
    )

    from core.mech.combat_resolution import DiceRollResult

    contested_result = JockeyContestedResult(
        attacker_roll=DiceRollResult(rolls=attacker_rolls, chosen=attacker_rolls),
        defender_roll=DiceRollResult(rolls=defender_rolls, chosen=defender_rolls),
        attacker_total=attacker_total,
        defender_total=defender_total,
        attacker_wins=attacker_total >= defender_total,
    )

    jockey_success = contested_result.attacker_wins

    if jockey_success:
        conditions_inflicted = list(option.inflicted_conditions)
        heat_dealt = option.heat
        damage_dealt = option.damage if option.damage > 0 else None
    else:
        conditions_inflicted = []
        heat_dealt = 0
        damage_dealt = None

    return JockeyResolutionResult(
        actor_id=input.actor_id,
        target_mech_id=input.target_mech_id,
        contested_result=contested_result,
        chosen_option=option,
        jockey_success=jockey_success,
        conditions_inflicted=conditions_inflicted,
        heat_dealt=heat_dealt,
        damage_dealt=damage_dealt,
        validation_errors=errors,
    )


def resolve_shake_off(
    input: ShakeOffInput,
    mech_hull: int,
    rider_grit: int,
) -> ShakeOffResolutionResult:
    """Resolve a Mech Shake Off action per PR2 4451-4452.

    The mech can attempt to shake off a riding pilot by winning a contested
    HULL vs GRIT check as a full action.

    Args:
        input: Shake off input with mech and rider information
        mech_hull: The mech's hull value
        rider_grit: The rider's grit value

    Returns:
        Detailed breakdown of what should happen during shake off
    """
    if input.rules is None:
        rules = DEFAULT_JOCKEY_RULES
    else:
        rules = input.rules

    attacker_rolls, defender_rolls, attacker_total, defender_total = (
        _roll_contested_check(
            mech_hull,
            rider_grit,
            input.settings,
            "mech",
            "pilot",
        )
    )

    from core.mech.combat_resolution import DiceRollResult

    contested_result = JockeyContestedResult(
        attacker_roll=DiceRollResult(rolls=attacker_rolls, chosen=attacker_rolls),
        defender_roll=DiceRollResult(rolls=defender_rolls, chosen=defender_rolls),
        attacker_total=attacker_total,
        defender_total=defender_total,
        attacker_wins=attacker_total >= defender_total,
    )

    shake_off_success = contested_result.attacker_wins

    return ShakeOffResolutionResult(
        mech_id=input.mech_id,
        rider_id=input.rider_id,
        contested_result=contested_result,
        shake_off_success=shake_off_success,
        rider_ejected=shake_off_success,
        validation_errors=[],
    )


def apply_jockey_result(
    target: CombatantState,
    result: JockeyResolutionResult,
) -> JockeyApplicationResult:
    """Apply Jockey result to combatant state.

    Updates target mech with conditions, heat, and/or damage from jockey.

    Args:
        target: Current target mech state
        result: Resolution result to apply

    Returns:
        Updated target with jockey effects applied
    """
    if not result.jockey_success:
        return JockeyApplicationResult(
            updated_target=target,
            jockey_success=False,
            conditions_applied=[],
            heat_dealt=0,
            damage_dealt=None,
        )

    updated_statuses = list(target.statuses)
    updated_conditions = list(target.conditions)
    updated_resources = target.resources

    for condition in result.conditions_inflicted:
        if condition not in updated_conditions:
            updated_conditions.append(condition)

    if result.heat_dealt > 0:
        new_heat = min(
            updated_resources.heat_current + result.heat_dealt,
            updated_resources.heat_cap,
        )
        updated_resources = updated_resources.model_copy(
            update={"heat_current": new_heat}
        )

    updated_target = target.model_copy(
        update={
            "statuses": updated_statuses,
            "conditions": updated_conditions,
            "resources": updated_resources,
        }
    )

    return JockeyApplicationResult(
        updated_target=updated_target,
        jockey_success=True,
        conditions_applied=result.conditions_inflicted,
        heat_dealt=result.heat_dealt,
        damage_dealt=result.damage_dealt,
    )


def apply_shake_off_result(
    mech: CombatantState,
    rider: CombatantState | None,
    result: ShakeOffResolutionResult,
    rider_position: HexPosition | None = None,
) -> ShakeOffApplicationResult:
    """Apply Shake Off result to combatant state.

    If shake off succeeds, the rider is ejected from the mech.

    Args:
        mech: Current mech state
        rider: Current rider state (if still on mech)
        result: Resolution result to apply
        rider_position: Position to place rider after ejection

    Returns:
        Updated mech and rider with shake off effects applied
    """
    updated_rider = rider

    if result.shake_off_success:
        updated_rider = None

    return ShakeOffApplicationResult(
        updated_mech=mech,
        updated_rider=updated_rider,
        shake_off_success=result.shake_off_success,
        rider_ejected=result.rider_ejected,
    )


try:
    from core.shared.heat import MeltdownState
    from core.mech.combat_state import CombatantState

    CombatantState.model_rebuild(
        _types_namespace={"MeltdownState": MeltdownState, "StabilizeState": type(None)}
    )
except ImportError:
    pass
