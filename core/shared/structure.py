"""Structure damage resolution primitives for Lancer TTRPG.

Provides type-safe structure damage resolution per PR2 4598-4643 rules:
- Structure damage chart (glancing blow, system trauma, direct hit, crushing hit)
- System trauma selection (mount destruction, system destruction)
- Direct hit outcomes based on remaining structure
- Integration with existing MechanicalEffect system

Resolution Pattern:
1. resolve_structure_damage() - Pure resolution logic, returns what SHOULD happen
2. apply_structure_result() - Applies result to combatant state, creates battlefield object if destroyed
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType, SizeClass
from core.shared.dice import DiceExpression
from core.shared.saves import SaveRequest
from core.shared.battlefield_objects import DestroyedMechObject
from core.mech.combat_rules import (
    StructureDamageRules,
    DEFAULT_STRUCTURE_DAMAGE_RULES,
    SystemTraumaRules,
)
from core.mech.combat_state import (
    MechInventory,
    WeaponMountState,
    MechSystemState,
    WeaponState,
    CombatantState,
)


StructureOutcomeLiteral = Literal[
    "glancing_blow", "system_trauma", "direct_hit", "crushing_hit"
]


class StructureInput(FrozenModel):
    """Input for structure damage resolution.

    Represents the context needed to resolve structure damage when a mech
    reaches 0 HP from incoming damage.
    """

    damage_dealt: int = Field(..., ge=0, description="Incoming damage that exceeded HP")
    remaining_structure: int = Field(
        ..., ge=1, le=4, description="Structure points BEFORE this damage"
    )
    inventory: MechInventory | None = Field(
        default=None, description="Mech inventory for system trauma selection"
    )
    rules: StructureDamageRules | None = Field(
        default=None, description="Override resolution rules"
    )


class SystemTraumaSelection(FrozenModel):
    """System trauma selection result.

    Per PR2: On system trauma (roll 2-4), roll 1d6:
    - 1-3: Destroy weapons on one mount
    - 4-6: Destroy one system
    - Fallback to Direct Hit if nothing available to destroy
    """

    trauma_roll: int = Field(..., ge=1, le=6, description="d6 roll for trauma type")
    initial_target: Literal["mount", "system"] = Field(
        ..., description="Initial target based on trauma roll"
    )
    resolved_target: Literal["mount", "system", "direct_hit"] = Field(
        ..., description="Final resolved target"
    )
    mount_index: int | None = Field(
        default=None, description="Mount index to destroy (if mount target)"
    )
    system_id: str | None = Field(
        default=None, description="System ID to destroy (if system target)"
    )
    eligible_mounts: list[int] = Field(
        default_factory=list, description="Mounts eligible for destruction"
    )
    eligible_systems: list[str] = Field(
        default_factory=list, description="Systems eligible for destruction"
    )
    fallback_reason: Literal["none", "no_mounts", "no_systems", "none_available"] = (
        Field(default="none", description="Reason for fallback to Direct Hit")
    )


class StructureResolutionResult(FrozenModel):
    """Complete result of structure damage resolution.

    Provides detailed breakdown of what happened and what should be applied
    to the combatant state.
    """

    outcome: StructureOutcomeLiteral = Field(
        ..., description="Primary structure outcome"
    )
    dice_rolls: list[int] = Field(default_factory=list, description="All d6 rolls made")
    lowest_roll: int = Field(
        ..., description="Lowest roll (used for outcome determination)"
    )
    system_trauma: SystemTraumaSelection | None = Field(
        default=None, description="System trauma details if applicable"
    )
    statuses_to_apply: list[StatusType] = Field(
        default_factory=list, description="Status effects to apply"
    )
    hull_check_request: SaveRequest | None = Field(
        default=None, description="Hull check required for Direct Hit at 2 structure"
    )
    inventory_update: MechInventory | None = Field(
        default=None, description="Updated inventory after system trauma"
    )
    mech_destroyed: bool = Field(..., description="Whether the mech is destroyed")
    spillover_damage: int = Field(
        default=0, description="Damage that spilled over HP to structure"
    )


class StructureApplicationResult(FrozenModel):
    """Result of applying structure damage to combatant state.

    Returns the updated combatant and any battlefield objects created.
    """

    updated_combatant: CombatantState = Field(
        ..., description="Combatant with structure damage applied"
    )
    statuses_applied: list[StatusType] = Field(
        default_factory=list, description="Status effects that were applied"
    )
    inventory_updated: MechInventory | None = Field(
        default=None, description="Final inventory state after trauma"
    )
    created_object: "DestroyedMechObject | None" = Field(
        default=None, description="Battlefield object if mech destroyed"
    )
    mech_destroyed: bool = Field(..., description="Whether the mech was destroyed")


def resolve_structure_damage(
    input: StructureInput,
    force_roll: int | None = None,
) -> StructureResolutionResult:
    """Resolve structure damage per PR2 rules.

    Resolution Order (PR2 4592-4637):
    1. Roll 1d6 per structure damage marked (including current)
    2. Choose lowest result
    3. Determine outcome from table:
       - 5-6: Glancing Blow (impaired)
       - 2-4: System Trauma (destroy mount/system)
       - 1: Direct Hit (see below)
       - 2+ 1s: Crushing Hit (destroyed)
    4. Direct Hit suboutcome by remaining structure:
       - 3+: Stunned
       - 2: Hull check or destroyed; stunned on success
       - 1: Destroyed

    Args:
        input: Structure damage input context
        force_roll: Optional forced roll value for deterministic testing

    Returns:
        Detailed breakdown of structure damage resolution
    """
    rules = input.rules or DEFAULT_STRUCTURE_DAMAGE_RULES
    remaining = input.remaining_structure

    dice_count = max(remaining, 1)
    if force_roll is not None:
        rolls = [force_roll]
    else:
        rolls = DiceExpression.parse(f"{dice_count}d6").roll()

    lowest = min(rolls)
    num_ones = rolls.count(1)

    statuses: list[StatusType] = []
    system_trauma_result: SystemTraumaSelection | None = None
    inventory_update: MechInventory | None = None
    hull_check: SaveRequest | None = None
    mech_destroyed = False
    spillover = max(input.damage_dealt, 0)

    if num_ones >= 2 and rules.multiple_ones_crushing:
        outcome: StructureOutcomeLiteral = "crushing_hit"
        mech_destroyed = True
    else:
        outcome = _lookup_structure_outcome(lowest, rules)

        if outcome == "glancing_blow":
            statuses.append("impaired")

        elif outcome == "system_trauma":
            if input.inventory:
                trauma_result = _resolve_system_trauma(
                    input.inventory, rules.system_trauma_rules
                )
                system_trauma_result = trauma_result

                if trauma_result.resolved_target == "direct_hit":
                    outcome = "direct_hit"
                    inventory_update = input.inventory
                else:
                    inventory_update = _apply_system_trauma(
                        input.inventory, trauma_result
                    )

        elif outcome == "direct_hit":
            if remaining >= 3:
                statuses.append("stunned")
            elif remaining == 2:
                statuses.append("stunned")
                hull_check = SaveRequest(
                    save_type="hull",
                    save_target=10,
                    save_bonus=0,
                    target_conditions=[],
                )
            elif remaining == 1:
                mech_destroyed = True

    return StructureResolutionResult(
        outcome=outcome,
        dice_rolls=rolls,
        lowest_roll=lowest,
        system_trauma=system_trauma_result,
        statuses_to_apply=statuses,
        hull_check_request=hull_check,
        inventory_update=inventory_update,
        mech_destroyed=mech_destroyed,
        spillover_damage=spillover,
    )


def apply_structure_result(
    combatant: CombatantState,
    result: StructureResolutionResult,
    object_id: str,
    is_wreckage: bool = False,
) -> StructureApplicationResult:
    """Apply structure damage result to combatant state.

    Updates combatant with new structure, statuses, and creates battlefield
    object if destroyed.

    Args:
        combatant: Current combatant state
        result: Resolution result to apply
        object_id: ID for battlefield object if destroyed
        is_wreckage: True if destroyed by reactor meltdown

    Returns:
        Updated combatant and created battlefield object if any
    """
    from core.shared.battlefield_objects import DestroyedMechObject

    statuses_applied = result.statuses_to_apply.copy()

    updated_statuses = list(combatant.statuses)
    for status in statuses_applied:
        if status not in updated_statuses:
            updated_statuses.append(status)

    new_structure = combatant.resources.structure_current
    if result.mech_destroyed:
        new_structure = 0
    else:
        new_structure = max(0, combatant.resources.structure_current - 1)

    updated_combatant = combatant.model_copy(
        update={
            "statuses": updated_statuses,
            "resources": combatant.resources.model_copy(
                update={"structure_current": new_structure}
            ),
            "inventory": result.inventory_update or combatant.inventory,
        }
    )

    created_object: DestroyedMechObject | None = None
    if result.mech_destroyed:
        size_value = _get_size_value(combatant.stats.size)
        created_object = DestroyedMechObject.from_combatant(
            combatant_id=combatant.id,
            combatant_name=combatant.name,
            position=combatant.position,
            size_value=size_value,
            object_id=object_id,
            is_wreckage=is_wreckage,
            owner_id=combatant.id,
        )

    return StructureApplicationResult(
        updated_combatant=updated_combatant,
        statuses_applied=statuses_applied,
        inventory_updated=result.inventory_update,
        created_object=created_object,
        mech_destroyed=result.mech_destroyed,
    )


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


def _resolve_system_trauma(
    inventory: MechInventory,
    rules: SystemTraumaRules,
) -> SystemTraumaSelection:
    """Resolve system trauma selection."""
    trauma_roll = rules.roll.roll()[0] if rules.roll.roll() else 1

    if rules.mount_on.roll_min <= trauma_roll <= rules.mount_on.roll_max:
        initial_target: Literal["mount", "system"] = "mount"
    elif rules.system_on.roll_min <= trauma_roll <= rules.system_on.roll_max:
        initial_target = "system"
    else:
        initial_target = "mount"

    eligible_mounts = _eligible_mounts(inventory, rules.exclude_limited_no_charges)
    eligible_systems = _eligible_systems(inventory, rules.exclude_limited_no_charges)

    resolved_target: Literal["mount", "system", "direct_hit"] = initial_target
    mount_index: int | None = None
    system_id: str | None = None
    fallback_reason: Literal["none", "no_mounts", "no_systems", "none_available"] = (
        "none"
    )

    if initial_target == "mount":
        if eligible_mounts:
            mount_index = eligible_mounts[0]
        elif eligible_systems and rules.fallback_to_other_if_none:
            resolved_target = "system"
            system_id = eligible_systems[0]
            fallback_reason = "no_mounts"
        elif rules.fallback_to_direct_hit_if_none:
            resolved_target = "direct_hit"
            fallback_reason = "none_available"
        else:
            fallback_reason = "no_mounts"
    else:
        if eligible_systems:
            system_id = eligible_systems[0]
        elif eligible_mounts and rules.fallback_to_other_if_none:
            resolved_target = "mount"
            mount_index = eligible_mounts[0]
            fallback_reason = "no_systems"
        elif rules.fallback_to_direct_hit_if_none:
            resolved_target = "direct_hit"
            fallback_reason = "none_available"
        else:
            fallback_reason = "no_systems"

    return SystemTraumaSelection(
        trauma_roll=trauma_roll,
        initial_target=initial_target,
        resolved_target=resolved_target,
        mount_index=mount_index,
        system_id=system_id,
        eligible_mounts=eligible_mounts,
        eligible_systems=eligible_systems,
        fallback_reason=fallback_reason,
    )


def _eligible_mounts(
    inventory: MechInventory,
    exclude_limited_no_charges: bool,
) -> list[int]:
    """Get mount indices eligible for destruction."""
    mounts: list[int] = []
    for mount in inventory.mounts:
        if mount.destroyed:
            continue
        has_valid_weapon = any(
            _weapon_valid(weapon, exclude_limited_no_charges)
            for weapon in mount.weapons
        )
        if has_valid_weapon:
            mounts.append(mount.mount_index)
    return sorted(mounts)


def _eligible_systems(
    inventory: MechInventory,
    exclude_limited_no_charges: bool,
) -> list[str]:
    """Get system IDs eligible for destruction."""
    systems: list[str] = []
    for system in inventory.systems:
        if system.destroyed:
            continue
        if exclude_limited_no_charges and system.limited_charges_remaining == 0:
            continue
        systems.append(system.system_id)
    return sorted(systems)


def _weapon_valid(weapon: WeaponState, exclude_limited_no_charges: bool) -> bool:
    """Check if weapon is valid for destruction."""
    if weapon.destroyed:
        return False
    if exclude_limited_no_charges and weapon.limited_charges_remaining == 0:
        return False
    return True


def _apply_system_trauma(
    inventory: MechInventory,
    selection: SystemTraumaSelection,
) -> MechInventory:
    """Apply system trauma to inventory."""
    if selection.resolved_target == "mount" and selection.mount_index is not None:
        mounts: list[WeaponMountState] = []
        for mount in inventory.mounts:
            if mount.mount_index == selection.mount_index:
                destroyed_weapons = [
                    weapon.model_copy(update={"destroyed": True})
                    for weapon in mount.weapons
                ]
                mounts.append(mount.model_copy(update={"weapons": destroyed_weapons}))
            else:
                mounts.append(mount)
        return inventory.model_copy(update={"mounts": mounts})

    if selection.resolved_target == "system" and selection.system_id is not None:
        systems: list[MechSystemState] = []
        for system in inventory.systems:
            if system.system_id == selection.system_id:
                systems.append(system.model_copy(update={"destroyed": True}))
            else:
                systems.append(system)
        return inventory.model_copy(update={"systems": systems})

    return inventory


def _lookup_structure_outcome(
    roll: int,
    rules: StructureDamageRules,
) -> StructureOutcomeLiteral:
    """Lookup structure outcome from table by roll."""
    for entry in rules.table:
        if entry.roll_min <= roll <= entry.roll_max:
            if entry.outcome.name == "glancing_blow":
                return "glancing_blow"
            elif entry.outcome.name == "system_trauma":
                return "system_trauma"
            elif entry.outcome.name == "direct_hit":
                return "direct_hit"
            elif entry.outcome.name == "crushing_hit":
                return "crushing_hit"
    raise ValueError(f"No structure outcome for roll {roll}")


def _lookup_direct_hit_outcome(
    remaining_structure: int,
    rules: StructureDamageRules,
) -> str:
    """Get description of direct hit outcome (for documentation purposes)."""
    for entry in rules.direct_hit_outcomes:
        if remaining_structure < entry.remaining_structure_min:
            continue
        if (
            entry.remaining_structure_max is None
            or remaining_structure <= entry.remaining_structure_max
        ):
            if entry.outcome.stunned_until_end_next_turn and remaining_structure >= 2:
                return "direct_hit_stunned"
            elif entry.outcome.hull_check_required:
                return "direct_hit_hull_check"
            elif entry.outcome.destroyed:
                return "direct_hit_destroyed"
            return "direct_hit"
    return "direct_hit"
