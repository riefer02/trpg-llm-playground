"""Stabilize action resolution primitives for Lancer TTRPG.

Implements resolution logic for the Stabilize full action per PR2 4275-4286:
- Primary choice: Cool heat (reset to 0, clear exposed) OR Spend Repair (full HP)
- Secondary choice: Reload Loading weapons OR Clear Burn OR Clear condition

Resolution Pattern:
1. resolve_stabilize() - Pure resolution logic, returns what SHOULD happen
2. apply_stabilize_result() - Applies result to combatant state
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType
from core.shared.id_helpers import CombatantIdField
from core.mech.combat_state import CombatantState
from core.mech.weapon import WeaponTagType


StabilizePrimaryChoice = Literal["cool_heat", "spend_repair_full_hp"]
StabilizeSecondaryChoice = Literal["reload_loading", "clear_burn", "clear_condition"]


class StabilizeRule(FrozenModel):
    """Stabilize action options."""

    primary_options: list[StabilizePrimaryChoice] = Field(
        default_factory=lambda: ["cool_heat", "spend_repair_full_hp"]
    )
    repair_cost: int = Field(default=1, ge=0)
    cool_heat_clears_exposed: bool = True
    secondary_options: list[StabilizeSecondaryChoice] = Field(
        default_factory=lambda: ["reload_loading", "clear_burn", "clear_condition"]
    )
    condition_clear_allows_adjacent_ally: bool = True
    condition_clear_disallow_self_sourced: bool = True
    clearable_conditions: list[StatusType] = Field(
        default_factory=lambda: [
            "impaired",
            "shredded",
            "jammed",
            "slowed",
            "immobilized",
            "stunned",
            "lock_on",
        ]
    )


DEFAULT_STABILIZE_RULES = StabilizeRule()


class StabilizeInput(FrozenModel):
    """Input for stabilize action resolution.

    Represents the choices made when performing a Stabilize action.
    """

    primary_choice: StabilizePrimaryChoice = Field(
        ..., description="Primary stabilize option: cool heat or repair HP"
    )
    secondary_choice: StabilizeSecondaryChoice = Field(
        ...,
        description="Secondary stabilize option: reload, clear burn, or clear condition",
    )
    condition_target_id: CombatantIdField | None = Field(
        default=None,
        description="Target ID for condition clearing (required if secondary_choice is clear_condition)",
    )
    rules: StabilizeRule | None = Field(
        default=None, description="Override resolution rules"
    )


class StabilizeResolutionResult(FrozenModel):
    """Complete result of stabilize resolution (pure logic).

    Provides detailed breakdown of what should happen when stabilizing.
    """

    primary_choice: StabilizePrimaryChoice = Field(
        ..., description="Which primary option was chosen"
    )
    secondary_choice: StabilizeSecondaryChoice = Field(
        ..., description="Which secondary option was chosen"
    )
    heat_cleared: bool = Field(
        default=False, description="Whether heat was cleared to 0"
    )
    exposed_cleared: bool = Field(
        default=False, description="Whether exposed condition was ended"
    )
    hp_restored: int = Field(default=0, ge=0, description="HP restored to combatant")
    weapons_reloaded: list[str] = Field(
        default_factory=list, description="Weapon IDs that were reloaded"
    )
    burn_cleared: bool = Field(default=False, description="Whether burn was cleared")
    conditions_cleared: list[StatusType] = Field(
        default_factory=list, description="Conditions that were cleared"
    )
    condition_target_id: str | None = Field(
        default=None, description="Target ID for condition clearing"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class StabilizeApplicationResult(FrozenModel):
    """Result of applying stabilize result to combatant state.

    Returns the updated combatant with new heat, HP, statuses, and weapon states.
    """

    updated_combatant: CombatantState = Field(
        ..., description="Combatant with stabilize result applied"
    )
    heat_cleared: bool = Field(default=False, description="Whether heat was cleared")
    hp_restored_amount: int = Field(
        default=0, ge=0, description="Amount of HP restored"
    )
    hp_current_after: int = Field(default=0, ge=0, description="HP after stabilization")
    statuses_cleared: list[StatusType] = Field(
        default_factory=list, description="Status effects that were cleared"
    )
    conditions_cleared: list[StatusType] = Field(
        default_factory=list, description="Conditions that were cleared"
    )
    weapons_reloaded: list[str] = Field(
        default_factory=list, description="Weapon IDs that were reloaded"
    )


def resolve_stabilize(input: StabilizeInput) -> StabilizeResolutionResult:
    """Resolve a Stabilize action per PR2 4275-4286.

    Stabilize is a Full Action that allows the pilot to:
    - Primary: Cool heat (reset to 0, end exposed) OR Spend Repair (full HP)
    - Secondary: Reload Loading weapons OR Clear Burn OR Clear condition

    Args:
        input: Stabilize input with choices and target information

    Returns:
        Detailed breakdown of what should happen during stabilization
    """
    rules = input.rules or DEFAULT_STABILIZE_RULES
    errors: list[str] = []

    if input.secondary_choice == "clear_condition" and not input.condition_target_id:
        errors.append(
            "condition_target_id is required when secondary_choice is 'clear_condition'"
        )

    heat_cleared = False
    exposed_cleared = False
    hp_restored = 0
    weapons_reloaded: list[str] = []
    burn_cleared = False
    conditions_cleared: list[StatusType] = []

    if input.primary_choice == "cool_heat":
        heat_cleared = True
        if rules.cool_heat_clears_exposed:
            exposed_cleared = True
    elif input.primary_choice == "spend_repair_full_hp":
        hp_restored = rules.repair_cost

    if input.secondary_choice == "reload_loading":
        weapons_reloaded = []  # Populated during apply when we have inventory access
    elif input.secondary_choice == "clear_burn":
        burn_cleared = True
    elif input.secondary_choice == "clear_condition":
        conditions_cleared = list(rules.clearable_conditions)

    return StabilizeResolutionResult(
        primary_choice=input.primary_choice,
        secondary_choice=input.secondary_choice,
        heat_cleared=heat_cleared,
        exposed_cleared=exposed_cleared,
        hp_restored=hp_restored,
        weapons_reloaded=weapons_reloaded,
        burn_cleared=burn_cleared,
        conditions_cleared=conditions_cleared,
        condition_target_id=input.condition_target_id,
        validation_errors=errors,
    )


def apply_stabilize_result(
    combatant: CombatantState,
    result: StabilizeResolutionResult,
    target_combatant: CombatantState | None = None,
) -> StabilizeApplicationResult:
    """Apply stabilize result to combatant state.

    Updates combatant with new heat, HP, statuses, conditions, and weapon states.

    Args:
        combatant: Current combatant state
        result: Resolution result to apply
        target_combatant: Target for condition clearing (required if secondary_choice is clear_condition)

    Returns:
        Updated combatant with stabilize effects applied
    """
    statuses_cleared: list[StatusType] = []
    conditions_cleared: list[StatusType] = []
    weapons_reloaded: list[str] = []

    updated_statuses = list(combatant.statuses)
    updated_conditions = list(combatant.conditions)
    updated_resources = combatant.resources

    if result.heat_cleared:
        updated_resources = updated_resources.model_copy(update={"heat_current": 0})

    if result.exposed_cleared and "exposed" in updated_statuses:
        updated_statuses.remove("exposed")
        statuses_cleared.append("exposed")

    if result.hp_restored > 0:
        hp_restored_amount = result.hp_restored
        hp_max = combatant.stats.hp_max
        hp_after = min(updated_resources.hp_current + hp_restored_amount, hp_max)
        hp_restored_amount = hp_after - updated_resources.hp_current
        updated_resources = updated_resources.model_copy(
            update={"hp_current": hp_after}
        )
    else:
        hp_restored_amount = 0

    if result.burn_cleared:
        if "burn" in updated_statuses:
            updated_statuses.remove("burn")
            statuses_cleared.append("burn")

    if result.conditions_cleared and result.secondary_choice == "clear_condition":
        target = target_combatant if target_combatant else combatant
        target_conditions = list(target.conditions)
        for condition in result.conditions_cleared:
            if condition in target_conditions:
                target_conditions.remove(condition)
                conditions_cleared.append(condition)
        if target_combatant:
            target_combatant = target_combatant.model_copy(
                update={"conditions": target_conditions}
            )
        else:
            updated_conditions = target_conditions

    if result.weapons_reloaded or result.secondary_choice == "reload_loading":
        if combatant.inventory:
            for mount in combatant.inventory.mounts:
                for weapon in mount.weapons:
                    if not weapon.destroyed and _has_loading_tag(weapon.tags):
                        weapons_reloaded.append(weapon.weapon_id)

    updated_combatant = combatant.model_copy(
        update={
            "statuses": updated_statuses,
            "conditions": updated_conditions,
            "resources": updated_resources,
        }
    )

    return StabilizeApplicationResult(
        updated_combatant=updated_combatant,
        heat_cleared=result.heat_cleared,
        hp_restored_amount=hp_restored_amount,
        hp_current_after=updated_resources.hp_current,
        statuses_cleared=statuses_cleared,
        conditions_cleared=conditions_cleared,
        weapons_reloaded=weapons_reloaded,
    )


def _has_loading_tag(tags: list[WeaponTagType]) -> bool:
    """Check if a weapon has the Loading tag."""
    return "loading" in tags


try:
    from core.shared.heat import MeltdownState
    from core.mech.combat_state import CombatantState

    CombatantState.model_rebuild(
        _types_namespace={"MeltdownState": MeltdownState, "StabilizeState": type(None)}
    )
except ImportError:
    pass  # CombatantState or MeltdownState not yet available during initial import
