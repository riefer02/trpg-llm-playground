"""Overheat resolution primitives for Lancer TTRPG.

Provides type-safe overheat resolution per PR2 4660-4706 rules:
- Overheat chart (emergency shunt, power plant destabilize, meltdown)
- Meltdown suboutcomes by remaining stress
- MeltdownState for tracking delayed meltdowns
- Integration with existing MechanicalEffect system

Resolution Pattern:
1. resolve_overheat() - Pure resolution logic, returns what SHOULD happen
2. apply_overheat_result() - Applies result to combatant state
3. decrement_meltdown_countdown() - Called at turn start for countdown management
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType, SizeClass
from core.shared.dice import DiceExpression
from core.shared.saves import SaveRequest
from core.shared.battlefield_objects import WreckageState
from core.mech.combat_rules import OverheatRules, DEFAULT_OVERHEAT_RULES
from core.mech.combat_state import CombatantState


OverheatOutcomeLiteral = Literal[
    "emergency_shunt",
    "power_plant_destabilize",
    "meltdown",
    "irreversible_meltdown",
]


class MeltdownState(FrozenModel):
    """Tracks delayed meltdown countdown per PR2.

    When a mech suffers meltdown at 2 stress and fails the engineering check,
    a countdown is started. The meltdown triggers at the end of the specified
    turn unless the engineering check is passed.
    """

    turns_remaining: int = Field(
        ..., ge=1, description="Turns until meltdown (1d6 rolled by GM)"
    )
    triggered_by_overheat: bool = Field(
        default=True,
        description="True if from overheat check, False if from check failure",
    )
    exposed_applied: bool = Field(
        default=True, description="Whether exposed status has been applied"
    )
    is_immediate: bool = Field(
        default=False,
        description="True if meltdown triggers at end of next turn (stress 1)",
    )


class OverheatInput(FrozenModel):
    """Input for overheat resolution.

    Represents the context needed to resolve overheat when a mech exceeds
    its heat capacity and must mark reactor stress.
    """

    stress_marked: int = Field(
        ..., ge=1, description="Total stress boxes marked (including just-marked)"
    )
    remaining_stress: int = Field(
        ..., ge=0, description="Stress boxes remaining BEFORE this marking"
    )
    rules: OverheatRules | None = Field(
        default=None, description="Override resolution rules"
    )


class OverheatResolutionResult(FrozenModel):
    """Complete result of overheat resolution.

    Provides detailed breakdown of what happened and what should be applied
    to the combatant state.
    """

    outcome: OverheatOutcomeLiteral = Field(..., description="Primary overheat outcome")
    dice_rolls: list[int] = Field(default_factory=list, description="All d6 rolls made")
    lowest_roll: int = Field(
        ..., description="Lowest roll (used for outcome determination)"
    )
    statuses_to_apply: list[StatusType] = Field(
        default_factory=list, description="Status effects to apply"
    )
    engineering_check_request: SaveRequest | None = Field(
        default=None, description="Engineering check required for Meltdown at 2 stress"
    )
    meltdown_state: MeltdownState | None = Field(
        default=None, description="Meltdown countdown state if applicable"
    )
    stress_damage: int = Field(default=1, description="Stress damage dealt (usually 1)")
    heat_cleared: bool = Field(
        default=True, description="Whether heat was cleared (per PR2: always clear)"
    )


class OverheatApplicationResult(FrozenModel):
    """Result of applying overheat result to combatant state.

    Returns the updated combatant with new stress, statuses, and meltdown state.
    """

    updated_combatant: CombatantState = Field(
        ..., description="Combatant with overheat result applied"
    )
    statuses_applied: list[StatusType] = Field(
        default_factory=list, description="Status effects that were applied"
    )
    meltdown_state: MeltdownState | None = Field(
        default=None, description="Updated meltdown state if applicable"
    )
    stress_current: int = Field(..., description="New stress value after marking")
    heat_current: int = Field(..., description="New heat value (should be 0 per PR2)")


def resolve_overheat(
    input: OverheatInput,
    force_roll: int | None = None,
) -> OverheatResolutionResult:
    """Resolve overheat check per PR2 rules.

    Resolution Order (PR2 4660-4706):
    1. Mark 1 reactor stress (if heat exceeds capacity)
    2. Roll 1d6 per point of stress marked (including current)
    3. Choose lowest result
    4. Determine outcome from table:
       - 5-6: Emergency Shunt (impaired)
       - 2-4: Power Plant Destabilize (exposed)
       - 1: Meltdown (see below)
       - 2 1s: Irreversible Meltdown (countdown)
    5. Meltdown suboutcome by remaining stress:
       - 3+: Exposed
       - 2: Engineering check or countdown; exposed on success
       - 1: Immediate meltdown at end of next turn
    6. Clear all heat (per PR2: erase all heat after overheat check)

    Args:
        input: Overheat input context
        force_roll: Optional forced roll value for deterministic testing

    Returns:
        Detailed breakdown of overheat resolution
    """
    rules = input.rules or DEFAULT_OVERHEAT_RULES
    remaining = input.remaining_stress

    dice_count = input.stress_marked if rules.roll_dice_per_stress else 1
    if force_roll is not None:
        rolls = [force_roll]
    else:
        rolls = DiceExpression.parse(f"{dice_count}d6").roll()

    lowest = min(rolls)
    num_ones = rolls.count(1)

    statuses: list[StatusType] = []
    meltdown_state: MeltdownState | None = None
    engineering_check: SaveRequest | None = None

    if num_ones >= 2 and rules.irreversible_meltdown_on_multiple_ones:
        outcome: OverheatOutcomeLiteral = "irreversible_meltdown"
        meltdown_state = MeltdownState(
            turns_remaining=1,
            triggered_by_overheat=True,
            exposed_applied=True,
            is_immediate=False,
        )
    else:
        outcome = _lookup_overheat_outcome(lowest, rules)

        if outcome == "emergency_shunt":
            statuses.append("impaired")

        elif outcome == "power_plant_destabilize":
            statuses.append("exposed")

        elif outcome == "meltdown":
            if remaining >= 3:
                statuses.append("exposed")
            elif remaining == 2:
                statuses.append("exposed")
                engineering_check = SaveRequest(
                    save_type="engineering",
                    save_target=10,
                    save_bonus=0,
                    target_conditions=[],
                )
            elif remaining == 1:
                meltdown_state = MeltdownState(
                    turns_remaining=1,
                    triggered_by_overheat=True,
                    exposed_applied=True,
                    is_immediate=True,
                )

    return OverheatResolutionResult(
        outcome=outcome,
        dice_rolls=rolls,
        lowest_roll=lowest,
        statuses_to_apply=statuses,
        engineering_check_request=engineering_check,
        meltdown_state=meltdown_state,
        stress_damage=rules.stress_per_overheat,
        heat_cleared=True,
    )


def apply_overheat_result(
    combatant: CombatantState,
    result: OverheatResolutionResult,
) -> OverheatApplicationResult:
    """Apply overheat result to combatant state.

    Updates combatant with new stress, statuses, and meltdown state.

    Args:
        combatant: Current combatant state
        result: Resolution result to apply

    Returns:
        Updated combatant with overheat effects applied
    """
    statuses_applied = result.statuses_to_apply.copy()

    updated_statuses = list(combatant.statuses)
    for status in statuses_applied:
        if status not in updated_statuses:
            updated_statuses.append(status)

    new_stress = max(0, combatant.resources.stress_current - result.stress_damage)
    new_heat = 0 if result.heat_cleared else combatant.resources.heat_current

    updated_combatant = combatant.model_copy(
        update={
            "statuses": updated_statuses,
            "resources": combatant.resources.model_copy(
                update={
                    "stress_current": new_stress,
                    "heat_current": new_heat,
                }
            ),
        }
    )

    return OverheatApplicationResult(
        updated_combatant=updated_combatant,
        statuses_applied=statuses_applied,
        meltdown_state=result.meltdown_state,
        stress_current=new_stress,
        heat_current=new_heat,
    )


def decrement_meltdown_countdown(
    combatant: CombatantState,
) -> tuple[CombatantState, bool]:
    """Decrement meltdown countdown at turn start.

    Called at the start of each turn for combatants with active meltdown state.

    Args:
        combatant: Combatant with potential meltdown countdown

    Returns:
        Tuple of (updated combatant, whether meltdown triggered)
    """
    meltdown = combatant.meltdown_state
    if not meltdown:
        return combatant, False

    remaining = meltdown.turns_remaining - 1

    if remaining <= 0:
        statuses_after = [s for s in combatant.statuses if s != "exposed"]
        updated_combatant = combatant.model_copy(
            update={
                "meltdown_state": None,
                "statuses": statuses_after,
            }
        )
        return updated_combatant, True

    updated_combatant = combatant.model_copy(
        update={
            "meltdown_state": meltdown.model_copy(update={"turns_remaining": remaining})
        }
    )

    return updated_combatant, False


def trigger_meltdown(
    combatant: CombatantState,
) -> tuple[CombatantState, "WreckageState"]:
    """Trigger immediate meltdown, destroying the mech.

    Per PR2: Reactor meltdown annihilates the mech and creates a burst 2
    explosion dealing 4d6 explosive damage (half on agility save).

    Args:
        combatant: Combatant suffering meltdown

    Returns:
        Tuple of (destroyed combatant, wreckage object)
    """
    from core.shared.battlefield_objects import WreckageState
    from core.shared.enums import SizeClass

    size_value = _get_size_value(combatant.stats.size)

    wreckage = WreckageState.from_meltdown(
        combatant_id=combatant.id,
        combatant_name=combatant.name,
        position=combatant.position,
        size_value=size_value,
        object_id=f"wreckage_{combatant.id}",
    )

    statuses_after = [
        s for s in combatant.statuses if s not in ["impaired", "exposed", "stunned"]
    ]
    statuses_after.append("out")

    updated_combatant = combatant.model_copy(
        update={
            "statuses": statuses_after,
            "resources": combatant.resources.model_copy(
                update={
                    "hp_current": 0,
                    "structure_current": 0,
                    "stress_current": 0,
                }
            ),
            "meltdown_state": None,
        }
    )

    return updated_combatant, wreckage


def _get_size_value(size: SizeClass | int) -> int:
    """Extract integer size value from SizeClass or return as-is."""
    if isinstance(size, int):
        return size
    size_mapping = {
        "size_half": 1,
        "size_1": 1,
        "size_2": 2,
        "size_3": 3,
        "size_4": 4,
        "size_5": 5,
    }
    return size_mapping.get(str(size), 1)


def _lookup_overheat_outcome(
    roll: int,
    rules: OverheatRules,
) -> OverheatOutcomeLiteral:
    """Lookup overheat outcome from table by roll."""
    for entry in rules.table:
        if entry.roll_min <= roll <= entry.roll_max:
            if entry.outcome.name == "emergency_shunt":
                return "emergency_shunt"
            elif entry.outcome.name == "power_plant_destabilize":
                return "power_plant_destabilize"
            elif entry.outcome.name == "meltdown":
                return "meltdown"
            elif entry.outcome.name == "irreversible_meltdown":
                return "irreversible_meltdown"
    raise ValueError(f"No overheat outcome for roll {roll}")


def _lookup_meltdown_outcome(
    remaining_stress: int,
    rules: OverheatRules,
) -> str:
    """Get description of meltdown outcome (for documentation purposes)."""
    for entry in rules.meltdown_outcomes:
        if remaining_stress < entry.remaining_stress_min:
            continue
        if (
            entry.remaining_stress_max is None
            or remaining_stress <= entry.remaining_stress_max
        ):
            if entry.outcome.exposed_until_cleared:
                return "meltdown_exposed"
            elif entry.outcome.meltdown_countdown:
                return "meltdown_countdown"
            elif entry.outcome.meltdown_immediate:
                return "meltdown_immediate"
            return "meltdown"
    return "meltdown"


def check_unshackle_on_overheat(
    combatant: "CombatantState",
    force_roll: int | None = None,
) -> dict:
    """Check for unshackle trigger after overheat check resolution.

    Per PR2 5081-5082: Each time you roll an overheating check, roll a d20.
    On a roll of 1, your NHP's casket has suffered a traumatic code incursion
    and your NHP becomes Unshackled.

    Args:
        combatant: Combatant that may have NHP
        force_roll: Optional forced d20 roll for testing

    Returns:
        Dictionary with unshackle check results
    """
    from core.shared.ai import (
        resolve_unshackle_check,
        resolve_unshackled_behavior,
        apply_unshackle,
        UnshackleCheckInput,
        UnshackledBehaviorInput,
    )

    result = {
        "unshackle_check_performed": False,
        "unshackle_occurred": False,
        "nhp_behavior": None,
        "pilot_ejected": False,
        "combatant_updated": None,
    }

    if combatant.ai_type != "nhp":
        return result

    unshackle_input = UnshackleCheckInput(
        actor_id=combatant.id,
        check_type="overheat",
        force_roll=force_roll,
    )

    unshackle_result = resolve_unshackle_check(unshackle_input, has_nhp=True)

    if not unshackle_result.unshackle_occurred:
        return result

    behavior_input = UnshackledBehaviorInput(actor_id=combatant.id)
    behavior_result = resolve_unshackled_behavior(behavior_input)

    apply_result = apply_unshackle(combatant, unshackle_result, behavior_result)

    return {
        "unshackle_check_performed": True,
        "unshackle_occurred": True,
        "nhp_behavior": apply_result.nhp_behavior,
        "pilot_ejected": apply_result.pilot_ejected,
        "combatant_updated": apply_result.updated_combatant,
    }


# Rebuild CombatantState to resolve forward references
# This must be done after CombatantState is defined and types are available
try:
    from core.mech.combat_state import CombatantState
    from core.shared.protocols import ProtocolState
    from core.shared.turn_end import TurnEndEffectState

    CombatantState.model_rebuild(
        _types_namespace={
            "MeltdownState": MeltdownState,
            "ProtocolState": ProtocolState,
            "TurnEndEffectState": TurnEndEffectState,
        }
    )
except ImportError:
    pass  # CombatantState not yet available during initial import
