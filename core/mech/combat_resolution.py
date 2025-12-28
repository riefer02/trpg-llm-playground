"""Structure and overheat resolution helpers."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

from core.shared.dice import DiceExpression
from core.mech.combat_rules import (
    DEFAULT_STRUCTURE_DAMAGE_RULES,
    DEFAULT_OVERHEAT_RULES,
    StructureDamageRules,
    OverheatRules,
    StructureOutcomeType,
    OverheatOutcomeType,
)
from core.mech.combat_state import MechInventory, WeaponState, WeaponMountState, MechSystemState


class DiceRollResult(BaseModel):
    """Raw dice roll outcome."""

    rolls: list[int] = Field(default_factory=list)
    chosen: list[int] = Field(default_factory=list)

    model_config = {"frozen": True}


class StructureResolution(BaseModel):
    """Resolution result for structure damage."""

    outcome: StructureOutcomeType
    dice: DiceRollResult
    direct_hit_outcome: StructureOutcomeType | None = None
    system_trauma: SystemTraumaSelection | None = None
    updated_inventory: MechInventory | None = None
    structure_damage: int = 1
    spillover_damage: int = 0

    model_config = {"frozen": True}


class OverheatResolution(BaseModel):
    """Resolution result for overheat checks."""

    outcome: OverheatOutcomeType
    dice: DiceRollResult
    meltdown_outcome: OverheatOutcomeType | None = None
    stress_damage: int = 1

    model_config = {"frozen": True}


class ResolutionSettings(BaseModel):
    """Settings for deterministic or forced resolution."""

    forced_rolls: list[int] | None = None
    forced_system_trauma_roll: int | None = Field(default=None, ge=1, le=6)

    model_config = {"frozen": True}


class SystemTraumaSelection(BaseModel):
    """Resolved selection for system trauma results."""

    roll: int
    initial_target: Literal["mount", "system"]
    resolved_target: Literal["mount", "system", "direct_hit"]
    eligible_mounts: list[int] = Field(default_factory=list)
    eligible_systems: list[str] = Field(default_factory=list)
    destroyed_mount_index: int | None = None
    destroyed_system_id: str | None = None
    fallback_reason: Literal["none", "no_mounts", "no_systems", "none_available"] = "none"

    model_config = {"frozen": True}


def resolve_structure_damage(
    *,
    remaining_structure: int,
    incoming_damage: int,
    hp_before: int,
    structure_damage_marked: int = 1,
    inventory: MechInventory | None = None,
    rules: StructureDamageRules = DEFAULT_STRUCTURE_DAMAGE_RULES,
    settings: ResolutionSettings | None = None,
) -> StructureResolution:
    """
    Resolve a structure check, including spillover and direct hit outcomes.

    Args:
        structure_damage_marked: Total structure damage marked (including the one just taken).
        inventory: Inventory state used to resolve system trauma selections and fallbacks.
    """
    structure_damage = 1
    hp_after = max(hp_before - incoming_damage, 0)
    spillover = max(incoming_damage - hp_before, 0) if rules.spillover_damage_applies else 0

    dice_count = max(structure_damage_marked * rules.dice_per_structure_marked, 1)
    rolls = _roll_dice(dice_count, settings)
    chosen = _choose_lowest(rolls, 1)

    direct_hit_outcome = None
    system_trauma = None
    updated_inventory = None
    if rules.multiple_ones_crushing and rolls.count(1) >= 2:
        outcome = rules.crushing_hit_outcome
    else:
        outcome = _lookup_structure_outcome(chosen[0], rules)
        if outcome.name == "direct_hit":
            direct_hit_outcome = _lookup_direct_hit_outcome(remaining_structure, rules)
        elif outcome.name == "system_trauma" and inventory:
            system_trauma = _resolve_system_trauma(inventory, rules, settings)
            if system_trauma.resolved_target == "direct_hit":
                outcome = _direct_hit_outcome_base(rules)
                direct_hit_outcome = _lookup_direct_hit_outcome(remaining_structure, rules)
            else:
                updated_inventory = _apply_system_trauma(inventory, system_trauma)

    return StructureResolution(
        outcome=outcome,
        dice=DiceRollResult(rolls=rolls, chosen=chosen),
        direct_hit_outcome=direct_hit_outcome,
        system_trauma=system_trauma,
        updated_inventory=updated_inventory,
        structure_damage=structure_damage,
        spillover_damage=spillover,
    )


def resolve_overheat(
    *,
    stress_marked: int,
    remaining_stress: int,
    rules: OverheatRules = DEFAULT_OVERHEAT_RULES,
    settings: ResolutionSettings | None = None,
) -> OverheatResolution:
    """
    Resolve an overheat check, including meltdown subtable outcomes.

    Args:
        stress_marked: Total stress boxes marked (including the one just taken).
        remaining_stress: Stress boxes remaining after marking.
    """
    dice_count = max(stress_marked, 1) if rules.roll_dice_per_stress else 1
    rolls = _roll_dice(dice_count, settings)
    chosen = _choose_lowest(rolls, 1)

    if rules.irreversible_meltdown_on_multiple_ones and rolls.count(1) >= 2:
        outcome = rules.irreversible_meltdown_outcome
    else:
        outcome = _lookup_overheat_outcome(chosen[0], rules)
    meltdown_outcome = None
    if outcome.name == "meltdown":
        meltdown_outcome = _lookup_meltdown_outcome(remaining_stress, rules)

    return OverheatResolution(
        outcome=outcome,
        dice=DiceRollResult(rolls=rolls, chosen=chosen),
        meltdown_outcome=meltdown_outcome,
        stress_damage=rules.stress_per_overheat,
    )


def _roll_dice(count: int, settings: ResolutionSettings | None) -> list[int]:
    if settings and settings.forced_rolls:
        if len(settings.forced_rolls) < count:
            raise ValueError("Forced rolls provided fewer results than dice count")
        return list(settings.forced_rolls[:count])
    return DiceExpression.parse(f"{count}d6").roll()


def _choose_lowest(rolls: list[int], count: int) -> list[int]:
    return sorted(rolls)[:count]


def _direct_hit_outcome_base(rules: StructureDamageRules) -> StructureOutcomeType:
    for entry in rules.table:
        if entry.outcome.name == "direct_hit":
            return entry.outcome
    return StructureOutcomeType(name="direct_hit")


def _resolve_system_trauma(
    inventory: MechInventory,
    rules: StructureDamageRules,
    settings: ResolutionSettings | None,
) -> SystemTraumaSelection:
    trauma_rules = rules.system_trauma_rules
    roll = settings.forced_system_trauma_roll if settings and settings.forced_system_trauma_roll else None
    if roll is None:
        roll_result = trauma_rules.roll.roll()
        roll = roll_result[0] if roll_result else 1

    if trauma_rules.mount_on.roll_min <= roll <= trauma_rules.mount_on.roll_max:
        initial_target: Literal["mount", "system"] = "mount"
    elif trauma_rules.system_on.roll_min <= roll <= trauma_rules.system_on.roll_max:
        initial_target = "system"
    else:
        raise ValueError("System trauma roll outside defined ranges")

    eligible_mounts = _eligible_mounts(inventory, trauma_rules.exclude_limited_no_charges)
    eligible_systems = _eligible_systems(inventory, trauma_rules.exclude_limited_no_charges)

    resolved_target: Literal["mount", "system", "direct_hit"] = initial_target
    destroyed_mount_index = None
    destroyed_system_id = None
    fallback_reason: Literal["none", "no_mounts", "no_systems", "none_available"] = "none"

    if initial_target == "mount":
        if eligible_mounts:
            destroyed_mount_index = eligible_mounts[0]
        elif trauma_rules.fallback_to_other_if_none and eligible_systems:
            resolved_target = "system"
            destroyed_system_id = eligible_systems[0]
            fallback_reason = "no_mounts"
        elif trauma_rules.fallback_to_direct_hit_if_none:
            resolved_target = "direct_hit"
            fallback_reason = "none_available"
        else:
            fallback_reason = "no_mounts"
    else:
        if eligible_systems:
            destroyed_system_id = eligible_systems[0]
        elif trauma_rules.fallback_to_other_if_none and eligible_mounts:
            resolved_target = "mount"
            destroyed_mount_index = eligible_mounts[0]
            fallback_reason = "no_systems"
        elif trauma_rules.fallback_to_direct_hit_if_none:
            resolved_target = "direct_hit"
            fallback_reason = "none_available"
        else:
            fallback_reason = "no_systems"

    return SystemTraumaSelection(
        roll=roll,
        initial_target=initial_target,
        resolved_target=resolved_target,
        eligible_mounts=eligible_mounts,
        eligible_systems=eligible_systems,
        destroyed_mount_index=destroyed_mount_index,
        destroyed_system_id=destroyed_system_id,
        fallback_reason=fallback_reason,
    )


def _eligible_mounts(
    inventory: MechInventory,
    exclude_limited_no_charges: bool,
) -> list[int]:
    mounts: list[int] = []
    for mount in inventory.mounts:
        if mount.destroyed:
            continue
        has_valid_weapon = any(
            _weapon_valid(weapon, exclude_limited_no_charges) for weapon in mount.weapons
        )
        if has_valid_weapon:
            mounts.append(mount.mount_index)
    return sorted(mounts)


def _eligible_systems(
    inventory: MechInventory,
    exclude_limited_no_charges: bool,
) -> list[str]:
    systems: list[str] = []
    for system in inventory.systems:
        if system.destroyed:
            continue
        if exclude_limited_no_charges and system.limited_charges_remaining == 0:
            continue
        systems.append(system.system_id)
    return sorted(systems)


def _weapon_valid(weapon: WeaponState, exclude_limited_no_charges: bool) -> bool:
    if weapon.destroyed:
        return False
    if exclude_limited_no_charges and weapon.limited_charges_remaining == 0:
        return False
    return True


def _apply_system_trauma(
    inventory: MechInventory,
    selection: SystemTraumaSelection,
) -> MechInventory:
    if selection.resolved_target == "mount" and selection.destroyed_mount_index is not None:
        mounts: list[WeaponMountState] = []
        for mount in inventory.mounts:
            if mount.mount_index == selection.destroyed_mount_index:
                destroyed_weapons = [
                    weapon.model_copy(update={"destroyed": True}) for weapon in mount.weapons
                ]
                mounts.append(mount.model_copy(update={"weapons": destroyed_weapons}))
            else:
                mounts.append(mount)
        return inventory.model_copy(update={"mounts": mounts})

    if selection.resolved_target == "system" and selection.destroyed_system_id is not None:
        systems: list[MechSystemState] = []
        for system in inventory.systems:
            if system.system_id == selection.destroyed_system_id:
                systems.append(system.model_copy(update={"destroyed": True}))
            else:
                systems.append(system)
        return inventory.model_copy(update={"systems": systems})

    return inventory


def _lookup_structure_outcome(roll: int, rules: StructureDamageRules) -> StructureOutcomeType:
    for entry in rules.table:
        if entry.roll_min <= roll <= entry.roll_max:
            return entry.outcome
    raise ValueError("No structure outcome for roll")


def _lookup_direct_hit_outcome(
    remaining_structure: int,
    rules: StructureDamageRules,
) -> StructureOutcomeType:
    for entry in rules.direct_hit_outcomes:
        if remaining_structure < entry.remaining_structure_min:
            continue
        if entry.remaining_structure_max is None or remaining_structure <= entry.remaining_structure_max:
            return entry.outcome
    raise ValueError("No direct hit outcome for remaining structure")


def _lookup_overheat_outcome(roll: int, rules: OverheatRules) -> OverheatOutcomeType:
    for entry in rules.table:
        if entry.roll_min <= roll <= entry.roll_max:
            return entry.outcome
    raise ValueError("No overheat outcome for roll")


def _lookup_meltdown_outcome(
    remaining_stress: int,
    rules: OverheatRules,
) -> OverheatOutcomeType:
    for entry in rules.meltdown_outcomes:
        if remaining_stress < entry.remaining_stress_min:
            continue
        if entry.remaining_stress_max is None or remaining_stress <= entry.remaining_stress_max:
            return entry.outcome
    raise ValueError("No meltdown outcome for remaining stress")
