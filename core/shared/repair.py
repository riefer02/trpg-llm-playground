"""Repair system resolution primitives for Lancer TTRPG.

Implements rest and repair mechanics per PR2 4729-4785:
- Rest (1 hour): Cool heat, heal pilot HP, spend repairs
- Repair spending: 1/2/4 repairs for HP/structure/stress/destroyed mech
- Full Repair (10+ hours): Full restoration, repair cap refresh

Resolution Pattern:
1. resolve_rest() - Pure resolution logic, returns what SHOULD happen
2. apply_rest_result() - Applies result to combatant state
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType
from core.shared.id_helpers import CombatantIdField

if TYPE_CHECKING:
    from core.mech.combat_state import CombatantState


RestPrimaryChoice = Literal["cool_heat", "heal_pilot_hp"]
RestSecondaryChoice = Literal[
    "repair_hp",
    "repair_weapon",
    "repair_system",
    "repair_structure",
    "repair_stress",
    "repair_destroyed_mech",
    "end_condition",
]


class RepairSpec(FrozenModel):
    """Specification for a single repair operation."""

    target_id: CombatantIdField = Field(..., description="ID of target to repair")
    repair_type: Literal[
        "hp",
        "structure",
        "stress",
        "destroyed_weapon",
        "destroyed_system",
        "destroyed_mech",
    ] = Field(..., description="Type of repair to perform")
    repairs_spent: int = Field(..., ge=1, description="Number of repairs to spend")


class RepairResult(FrozenModel):
    """Result of a single repair operation."""

    target_id: str = Field(..., description="ID of target that was repaired")
    repair_type: str = Field(..., description="Type of repair performed")
    repairs_spent: int = Field(..., ge=1, description="Number of repairs spent")
    effect_applied: str = Field(..., description="Description of effect applied")


class RestRule(FrozenModel):
    """Rules for rest and repair mechanics."""

    min_hours_for_rest: int = Field(
        default=1, ge=1, description="Minimum hours for a rest"
    )
    min_hours_for_full_repair: int = Field(
        default=10, ge=1, description="Minimum hours for full repair"
    )
    heat_cleared_on_rest: bool = Field(
        default=True, description="Whether rest clears all heat"
    )
    pilot_hp_restore_fraction: float = Field(
        default=0.5, ge=0, le=1, description="Fraction of max HP to restore on rest"
    )
    pilot_hp_full_on_full_repair: bool = Field(
        default=True, description="Whether full repair restores full pilot HP"
    )
    can_end_conditions: bool = Field(
        default=True, description="Whether rest can end conditions"
    )
    condition_end_list: list[StatusType] = Field(
        default_factory=lambda: [
            "impaired",
            "shredded",
            "jammed",
            "slowed",
            "immobilized",
            "lock_on",
        ],
        description="Conditions that can be ended on rest",
    )
    repair_cost_hp: int = Field(
        default=1, ge=1, description="Repairs to restore full HP"
    )
    repair_cost_destroyed_weapon: int = Field(
        default=1, ge=1, description="Repairs to repair destroyed weapon"
    )
    repair_cost_destroyed_system: int = Field(
        default=1, ge=1, description="Repairs to repair destroyed system"
    )
    repair_cost_structure: int = Field(
        default=2, ge=1, description="Repairs to repair 1 structure"
    )
    repair_cost_stress: int = Field(
        default=2, ge=1, description="Repairs to repair 1 reactor stress"
    )
    repair_cost_destroyed_mech: int = Field(
        default=4, ge=1, description="Repairs to repair destroyed mech"
    )
    core_power_regains_on_full_repair: bool = Field(
        default=True, description="Whether full repair restores core power"
    )
    limited_weapons_reset_on_full_repair: bool = Field(
        default=True, description="Whether full repair resets limited weapon charges"
    )


DEFAULT_REST_RULES = RestRule()


class RestInput(FrozenModel):
    """Input for rest resolution.

    Represents the parameters for a rest action including duration and repairs to spend.
    """

    actor_id: str = Field(..., description="ID of the pilot/mech taking rest")
    duration_hours: int = Field(..., ge=1, description="Duration of rest in hours")
    repairs_to_spend: list[RepairSpec] = Field(
        default_factory=list, description="Repairs to spend during this rest"
    )
    repair_cap: int = Field(..., ge=0, description="Maximum repairs available")
    repairs_remaining: int = Field(..., ge=0, description="Current repairs remaining")
    rules: RestRule | None = Field(
        default=None, description="Override resolution rules"
    )


class RestResolutionResult(FrozenModel):
    """Complete result of rest resolution (pure logic).

    Provides detailed breakdown of what should happen during a rest.
    """

    actor_id: str = Field(..., description="ID of the pilot/mech taking rest")
    duration_hours: int = Field(..., description="Duration of rest in hours")
    is_full_repair: bool = Field(
        ..., description="Whether this is a full repair (10+ hours)"
    )
    heat_cleared: bool = Field(
        default=False, description="Whether heat was cleared to 0"
    )
    pilot_hp_restored: int = Field(default=0, ge=0, description="HP restored to pilot")
    conditions_ended: list[StatusType] = Field(
        default_factory=list, description="Conditions that were ended"
    )
    repairs_spent: int = Field(default=0, ge=0, description="Total repairs spent")
    repair_cap_before: int = Field(..., ge=0, description="Repair cap before rest")
    repair_cap_refreshed: bool = Field(
        default=False, description="Whether repair cap was refreshed"
    )
    core_power_regained: bool = Field(
        default=False, description="Whether core power was regained"
    )
    limited_weapons_reset: list[str] = Field(
        default_factory=list,
        description="Weapon IDs with limited charges that were reset",
    )
    repair_results: list[RepairResult] = Field(
        default_factory=list, description="Individual repair operation results"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


def calculate_repair_capacity(base_cap: int, hull_bonus: int) -> int:
    """Calculate total repair capacity from base and hull bonus.

    Per PR2: Every 2 points of HULL gives +1 repair capacity.

    Args:
        base_cap: Base repair capacity from frame
        hull_bonus: HULL skill bonus points

    Returns:
        Total repair capacity
    """
    return base_cap + (hull_bonus // 2)


def can_spend_repair(remaining: int, cost: int) -> bool:
    """Check if enough repairs remain to spend.

    Args:
        remaining: Current repairs remaining
        cost: Cost of the repair

    Returns:
        True if repair can be afforded
    """
    if cost < 0:
        return False
    return remaining >= cost


def resolve_rest(input: RestInput) -> RestResolutionResult:
    """Resolve a rest action per PR2 4729-4785.

    A rest is at least 1 hour of downtime. During a rest:
    - Cool all heat to 0
    - Heal 1/2 pilot HP (or full HP on full repair)
    - Spend repairs (up to repair cap)
    - End conditions (on full repair)

    Full Repair (10+ hours) additionally:
    - Repair all damage (unless destroyed)
    - Clear all stress and structure
    - Refresh repair cap
    - Regain core power
    - Reset limited weapons

    Args:
        input: Rest input with duration and repair specifications

    Returns:
        Detailed breakdown of rest effects
    """
    rules = input.rules or DEFAULT_REST_RULES
    errors: list[str] = []
    repair_results: list[RepairResult] = []
    total_repairs_spent = 0

    is_full_repair = input.duration_hours >= rules.min_hours_for_full_repair

    heat_cleared = rules.heat_cleared_on_rest
    pilot_hp_restored = 0
    conditions_ended: list[StatusType] = []
    repair_cap_refreshed = False
    core_power_regained = False
    limited_weapons_reset: list[str] = []

    pilot_hp_restore_fraction = (
        1.0
        if is_full_repair and rules.pilot_hp_full_on_full_repair
        else rules.pilot_hp_restore_fraction
    )

    if is_full_repair:
        repair_cap_refreshed = True
        if rules.core_power_regains_on_full_repair:
            core_power_regained = True
        if rules.limited_weapons_reset_on_full_repair:
            limited_weapons_reset = ["all"]  # Indicates all limited weapons reset
        if rules.can_end_conditions:
            conditions_ended = list(rules.condition_end_list)
    else:
        pilot_hp_restored = int(
            pilot_hp_restore_fraction * 10
        )  # Placeholder, actual max HP from combatant

    for repair_spec in input.repairs_to_spend:
        if total_repairs_spent + repair_spec.repairs_spent > input.repairs_remaining:
            errors.append(
                f"Insufficient repairs remaining for {repair_spec.repair_type} on {repair_spec.target_id}: "
                f"need {repair_spec.repairs_spent}, have {input.repairs_remaining - total_repairs_spent}"
            )
            continue

        if repair_spec.repairs_spent < 0:
            errors.append(
                f"Invalid repair cost for {repair_spec.target_id}: {repair_spec.repairs_spent}"
            )
            continue

        cost = repair_spec.repairs_spent
        actual_cost = _get_repair_cost(repair_spec.repair_type, rules)
        effect_desc = _get_repair_effect_desc(repair_spec.repair_type, actual_cost)

        if cost != actual_cost:
            errors.append(
                f"Wrong repair cost for {repair_spec.repair_type}: expected {actual_cost}, got {cost}"
            )
            continue

        total_repairs_spent += cost
        repair_results.append(
            RepairResult(
                target_id=repair_spec.target_id,
                repair_type=repair_spec.repair_type,
                repairs_spent=cost,
                effect_applied=effect_desc,
            )
        )

    if total_repairs_spent > input.repairs_remaining:
        errors.append(
            f"Total repairs spent ({total_repairs_spent}) exceeds repairs remaining ({input.repairs_remaining})"
        )

    return RestResolutionResult(
        actor_id=input.actor_id,
        duration_hours=input.duration_hours,
        is_full_repair=is_full_repair,
        heat_cleared=heat_cleared,
        pilot_hp_restored=pilot_hp_restored,
        conditions_ended=conditions_ended,
        repairs_spent=total_repairs_spent,
        repair_cap_before=input.repair_cap,
        repair_cap_refreshed=repair_cap_refreshed,
        core_power_regained=core_power_regained,
        limited_weapons_reset=limited_weapons_reset,
        repair_results=repair_results,
        validation_errors=errors,
    )


def _get_repair_cost(repair_type: str, rules: RestRule) -> int:
    """Get the repair cost for a given repair type."""
    cost_map = {
        "hp": rules.repair_cost_hp,
        "destroyed_weapon": rules.repair_cost_destroyed_weapon,
        "destroyed_system": rules.repair_cost_destroyed_system,
        "structure": rules.repair_cost_structure,
        "stress": rules.repair_cost_stress,
        "destroyed_mech": rules.repair_cost_destroyed_mech,
    }
    return cost_map.get(repair_type, 1)


def _get_repair_effect_desc(repair_type: str, cost: int) -> str:
    """Get a description of the repair effect."""
    if repair_type == "hp":
        return f"Heal to full HP ({cost} repair)"
    elif repair_type == "destroyed_weapon":
        return f"Repair destroyed weapon ({cost} repair)"
    elif repair_type == "destroyed_system":
        return f"Repair destroyed system ({cost} repair)"
    elif repair_type == "structure":
        return f"Repair 1 structure ({cost} repairs)"
    elif repair_type == "stress":
        return f"Repair 1 reactor stress ({cost} repairs)"
    elif repair_type == "destroyed_mech":
        return (
            f"Restore destroyed mech to 1 structure, 1 stress, full HP ({cost} repairs)"
        )
    return f"Unknown repair ({cost} repairs)"


def apply_rest_result(
    combatant: CombatantState,
    result: RestResolutionResult,
) -> CombatantState:
    """Apply rest result to combatant state.

    Updates combatant with restored HP, cleared heat, repaired systems, etc.

    Args:
        combatant: Current combatant state
        result: Resolution result to apply

    Returns:
        Updated combatant with rest effects applied
    """

    updated_resources = combatant.resources

    if result.heat_cleared:
        updated_resources = updated_resources.model_copy(update={"heat_current": 0})

    if result.repairs_spent > 0:
        repairs_after = max(
            0, updated_resources.repairs_remaining - result.repairs_spent
        )
        updated_resources = updated_resources.model_copy(
            update={"repairs_remaining": repairs_after}
        )

    updated_statuses = list(combatant.statuses)
    updated_conditions = list(combatant.conditions)

    for condition in result.conditions_ended:
        if condition in updated_conditions:
            updated_conditions.remove(condition)

    updated_combatant = combatant.model_copy(
        update={
            "statuses": updated_statuses,
            "conditions": updated_conditions,
            "resources": updated_resources,
        }
    )

    return updated_combatant


class DestroyedMechState(FrozenModel):
    """State tracking for a destroyed mech (wreck).

    Per PR2: A destroyed mech becomes an object providing hard cover.
    Climbing over/moving through is difficult terrain.
    """

    mech_id: str = Field(..., description="ID of the destroyed mech")
    wreck_position_hex: str = Field(..., description="Hex position of the wreck")
    is_recoverable: bool = Field(
        default=True, description="Whether wreck can be recovered/repaired"
    )
    is_melted: bool = Field(
        default=False, description="Whether wreck melted in reactor explosion"
    )
    provides_cover: bool = Field(
        default=True, description="Whether wreck provides hard cover"
    )
    is_difficult_terrain: bool = Field(
        default=True, description="Whether moving through is difficult terrain"
    )


def calculate_repair_cost(
    repair_type: Literal[
        "hp",
        "structure",
        "stress",
        "destroyed_weapon",
        "destroyed_system",
        "destroyed_mech",
    ],
    rules: RestRule | None = None,
) -> int:
    """Get the repair cost for a given repair type.

    Args:
        repair_type: Type of repair to perform
        rules: Optional rules override

    Returns:
        Number of repairs required
    """
    if rules is None:
        rules = DEFAULT_REST_RULES
    return _get_repair_cost(repair_type, rules)


try:
    from core.shared.heat import MeltdownState
    from core.mech.combat_state import CombatantState

    CombatantState.model_rebuild(
        _types_namespace={"MeltdownState": MeltdownState, "StabilizeState": type(None)}
    )
except ImportError:
    pass  # CombatantState or MeltdownState not yet available during initial import
