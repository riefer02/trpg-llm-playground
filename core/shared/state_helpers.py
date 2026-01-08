"""State mutation helpers for CombatantState and related models.

Provides type-safe, immutable state updates using Pydantic's model_copy pattern.
All functions return new instances without mutating the original.

Per PR2 rules:
- HP: 0 = destroyed (but structure check first)
- Heat: Clear all heat on overheat check
- Structure: 0 = destroyed (create wreckage)
- Stress: 0 = no stress damage
- Statuses: Never auto-clear at turn end
- Reactions: Per-round reset at ROUND start (not turn end)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType

if TYPE_CHECKING:
    from core.mech.combat_state import (
        CombatantState,
        CombatResources,
        MechInventory,
        WeaponMountState,
        MechSystemState,
        WeaponState,
    )
    from core.shared.heat import MeltdownState
    from core.shared.turn_end import TurnEndEffectState
    from core.shared.effects import EffectDuration


# Runtime imports - OverchargeState doesn't have forward references that need TYPE_CHECKING
from core.mech.combat_state import OverchargeState


__all__ = [
    "set_hp",
    "decrement_hp",
    "increment_hp",
    "set_heat",
    "increment_heat",
    "clear_heat",
    "set_structure",
    "decrement_structure",
    "set_stress",
    "decrement_stress",
    "add_status",
    "add_statuses",
    "remove_status",
    "clear_statuses",
    "destroy_weapon",
    "destroy_mount",
    "destroy_system",
    "consume_limited_charge",
    "clear_per_round_reactions",
    "increment_reaction_use",
    "set_meltdown_state",
    "decrement_meltdown_countdown",
    "create_overcharge_state",
    "use_overcharge",
    "reset_overcharge_uses",
    "set_effect_duration",
    "advance_effect_to_next_turn",
    "apply_damage",
    "apply_heat_damage",
    "apply_structure_damage",
    "apply_overheat_result",
    "StateUpdateResult",
]


# =============================================================================
# Change Tracking Result
# =============================================================================


class StateUpdateResult(FrozenModel):
    """Result of applying state updates with change tracking."""

    updated_combatant: CombatantState
    changes_summary: dict[str, tuple[int, int]] = Field(
        default_factory=dict,
        description="Summary of changes {field: (old_value, new_value)}",
    )


# =============================================================================
# CombatResources Helpers
# =============================================================================


def set_hp(state: CombatantState, new_hp: int) -> CombatantState:
    """Set current HP to exact value (clamped to 0).

    Args:
        state: Current combatant state
        new_hp: New HP value (will be clamped to >= 0)

    Returns:
        Updated combatant with new HP
    """
    clamped_hp = max(0, new_hp)
    new_resources = state.resources.model_copy(update={"hp_current": clamped_hp})
    return state.model_copy(update={"resources": new_resources})


def decrement_hp(state: CombatantState, amount: int) -> CombatantState:
    """Decrease HP by amount (minimum 0).

    Args:
        state: Current combatant state
        amount: Amount to decrease (default 1)

    Returns:
        Updated combatant with decreased HP
    """
    return set_hp(state, state.resources.hp_current - amount)


def increment_hp(
    state: CombatantState, amount: int, max_hp: int | None = None
) -> CombatantState:
    """Increase HP by amount (optionally clamped to max_hp).

    Args:
        state: Current combatant state
        amount: Amount to increase
        max_hp: Optional maximum HP to clamp to

    Returns:
        Updated combatant with increased HP
    """
    new_hp = state.resources.hp_current + amount
    if max_hp is not None:
        new_hp = min(new_hp, max_hp)
    return set_hp(state, new_hp)


def set_heat(state: CombatantState, new_heat: int) -> CombatantState:
    """Set current heat to exact value (clamped to 0).

    Args:
        state: Current combatant state
        new_heat: New heat value (will be clamped to >= 0)

    Returns:
        Updated combatant with new heat
    """
    clamped_heat = max(0, new_heat)
    new_resources = state.resources.model_copy(update={"heat_current": clamped_heat})
    return state.model_copy(update={"resources": new_resources})


def increment_heat(state: CombatantState, amount: int) -> CombatantState:
    """Increase heat by amount.

    Args:
        state: Current combatant state
        amount: Amount to increase

    Returns:
        Updated combatant with increased heat
    """
    return set_heat(state, state.resources.heat_current + amount)


def clear_heat(state: CombatantState) -> CombatantState:
    """Clear all heat (per PR2: happens on overheat check).

    Args:
        state: Current combatant state

    Returns:
        Updated combatant with heat set to 0
    """
    return set_heat(state, 0)


def set_structure(state: CombatantState, new_structure: int) -> CombatantState:
    """Set structure points to exact value (clamped to 0-4).

    Args:
        state: Current combatant state
        new_structure: New structure value (will be clamped to 0-4)

    Returns:
        Updated combatant with new structure
    """
    clamped_structure = max(0, min(4, new_structure))
    new_resources = state.resources.model_copy(
        update={"structure_current": clamped_structure}
    )
    return state.model_copy(update={"resources": new_resources})


def decrement_structure(state: CombatantState) -> CombatantState:
    """Decrease structure by 1 (for structure damage marking).

    Args:
        state: Current combatant state

    Returns:
        Updated combatant with structure decreased by 1
    """
    return set_structure(state, state.resources.structure_current - 1)


def set_stress(state: CombatantState, new_stress: int) -> CombatantState:
    """Set stress boxes to exact value (clamped to 0-4).

    Args:
        state: Current combatant state
        new_stress: New stress value (will be clamped to 0-4)

    Returns:
        Updated combatant with new stress
    """
    clamped_stress = max(0, min(4, new_stress))
    new_resources = state.resources.model_copy(
        update={"stress_current": clamped_stress}
    )
    return state.model_copy(update={"resources": new_resources})


def decrement_stress(state: CombatantState) -> CombatantState:
    """Decrease stress by 1 (for stress healing).

    Args:
        state: Current combatant state

    Returns:
        Updated combatant with stress decreased by 1
    """
    return set_stress(state, state.resources.stress_current - 1)


# =============================================================================
# Status/Condition Helpers
# =============================================================================


def add_status(state: CombatantState, status: StatusType) -> CombatantState:
    """Add a status to the combatant (no duplicates).

    Args:
        state: Current combatant state
        status: Status to add

    Returns:
        Updated combatant with status added (if not already present)
    """
    if status in state.statuses:
        return state
    return state.model_copy(update={"statuses": state.statuses + [status]})


def add_statuses(state: CombatantState, statuses: list[StatusType]) -> CombatantState:
    """Add multiple statuses (no duplicates).

    Args:
        state: Current combatant state
        statuses: Statuses to add

    Returns:
        Updated combatant with statuses added (excluding duplicates)
    """
    current = set(state.statuses)
    new_statuses = [s for s in statuses if s not in current]
    if not new_statuses:
        return state
    return state.model_copy(update={"statuses": state.statuses + new_statuses})


def remove_status(state: CombatantState, status: StatusType) -> CombatantState:
    """Remove a status from the combatant.

    Args:
        state: Current combatant state
        status: Status to remove

    Returns:
        Updated combatant with status removed (if present)
    """
    if status not in state.statuses:
        return state
    return state.model_copy(
        update={"statuses": [s for s in state.statuses if s != status]}
    )


def clear_statuses(
    state: CombatantState, statuses_to_clear: list[StatusType] | None = None
) -> CombatantState:
    """Clear statuses (all or specific ones).

    Note: Per PR2, no conditions auto-clear at turn end. Use this only
    for effects that explicitly clear conditions.

    Args:
        state: Current combatant state
        statuses_to_clear: Specific statuses to clear (None = clear all)

    Returns:
        Updated combatant with specified statuses removed
    """
    if statuses_to_clear is None:
        return state.model_copy(update={"statuses": []})
    return state.model_copy(
        update={"statuses": [s for s in state.statuses if s not in statuses_to_clear]}
    )


# =============================================================================
# Inventory Helpers
# =============================================================================


def destroy_weapon(mount: WeaponMountState, weapon_index: int) -> WeaponMountState:
    """Mark a weapon as destroyed on a mount.

    Args:
        mount: Current mount state
        weapon_index: Index of weapon to destroy

    Returns:
        Updated mount with weapon destroyed
    """
    if weapon_index < 0 or weapon_index >= len(mount.weapons):
        return mount
    destroyed_weapons = [
        w.model_copy(update={"destroyed": True}) if i == weapon_index else w
        for i, w in enumerate(mount.weapons)
    ]
    return mount.model_copy(update={"weapons": destroyed_weapons})


def destroy_mount(inventory: MechInventory, mount_index: int) -> MechInventory:
    """Destroy all weapons on a mount (for system trauma).

    Args:
        inventory: Current inventory state
        mount_index: Index of mount to destroy weapons on

    Returns:
        Updated inventory with mount weapons destroyed
    """
    updated_mounts = []
    for mount in inventory.mounts:
        if mount.mount_index == mount_index:
            destroyed_weapons = [
                w.model_copy(update={"destroyed": True}) for w in mount.weapons
            ]
            updated_mounts.append(
                mount.model_copy(update={"weapons": destroyed_weapons})
            )
        else:
            updated_mounts.append(mount)
    return inventory.model_copy(update={"mounts": updated_mounts})


def destroy_system(inventory: MechInventory, system_id: str) -> MechInventory:
    """Mark a system as destroyed (for system trauma).

    Args:
        inventory: Current inventory state
        system_id: ID of system to destroy

    Returns:
        Updated inventory with system destroyed
    """
    updated_systems = [
        s.model_copy(update={"destroyed": True}) if s.system_id == system_id else s
        for s in inventory.systems
    ]
    return inventory.model_copy(update={"systems": updated_systems})


def consume_limited_charge(
    inventory: MechInventory, item_id: str
) -> tuple[MechInventory, bool]:
    """Consume one limited charge from a weapon or system.

    Args:
        inventory: Current inventory state
        item_id: ID of weapon or system to consume charge from

    Returns:
        Tuple of (updated_inventory, was_successful)
    """
    for mount in inventory.mounts:
        for i, weapon in enumerate(mount.weapons):
            if weapon.weapon_id == item_id and weapon.limited_charges_remaining:
                if weapon.limited_charges_remaining > 0:
                    new_charges = weapon.limited_charges_remaining - 1
                    new_weapon = weapon.model_copy(
                        update={"limited_charges_remaining": new_charges}
                    )
                    new_weapons = [
                        new_weapon if j == i else w for j, w in enumerate(mount.weapons)
                    ]
                    new_mount = mount.model_copy(update={"weapons": new_weapons})
                    new_mounts = [
                        new_mount if m.mount_index == mount.mount_index else m
                        for m in inventory.mounts
                    ]
                    return inventory.model_copy(update={"mounts": new_mounts}), True
                return inventory, False

    for i, system in enumerate(inventory.systems):
        if system.system_id == item_id and system.limited_charges_remaining:
            if system.limited_charges_remaining > 0:
                new_charges = system.limited_charges_remaining - 1
                new_system = system.model_copy(
                    update={"limited_charges_remaining": new_charges}
                )
                new_systems = [
                    new_system if j == i else s for j, s in enumerate(inventory.systems)
                ]
                return inventory.model_copy(update={"systems": new_systems}), True
            return inventory, False

    return inventory, False


# =============================================================================
# Per-Round/Reaction Helpers
# =============================================================================


def clear_per_round_reactions(state: CombatantState) -> CombatantState:
    """Clear per-round reaction counts (reset at ROUND start per PR2).

    Args:
        state: Current combatant state

    Returns:
        Updated combatant with per_round_reactions cleared
    """
    return state.model_copy(update={"per_round_reactions": {}})


def increment_reaction_use(state: CombatantState, action_id: str) -> CombatantState:
    """Increment the use count for a per-round reaction.

    Args:
        state: Current combatant state
        action_id: ID of the reaction action

    Returns:
        Updated combatant with reaction use incremented
    """
    current = state.per_round_reactions.get(action_id, 0)
    new_reactions = {**state.per_round_reactions, action_id: current + 1}
    return state.model_copy(update={"per_round_reactions": new_reactions})


# =============================================================================
# Meltdown Helpers
# =============================================================================


def set_meltdown_state(
    state: CombatantState, meltdown: MeltdownState | None
) -> CombatantState:
    """Set or clear the meltdown state.

    Args:
        state: Current combatant state
        meltdown: New meltdown state (or None to clear)

    Returns:
        Updated combatant with new meltdown state
    """
    return state.model_copy(update={"meltdown_state": meltdown})


def decrement_meltdown_countdown(state: CombatantState) -> tuple[CombatantState, bool]:
    """Decrement meltdown countdown at turn start.

    Args:
        state: Current combatant state

    Returns:
        Tuple of (updated_combatant, whether_meltdown_triggered)
    """
    meltdown = state.meltdown_state
    if not meltdown:
        return state, False

    remaining = meltdown.turns_remaining - 1
    if remaining <= 0:
        updated = clear_statuses(state, ["exposed"])
        updated = updated.model_copy(update={"meltdown_state": None})
        return updated, True

    new_meltdown = meltdown.model_copy(update={"turns_remaining": remaining})
    return state.model_copy(update={"meltdown_state": new_meltdown}), False


# =============================================================================
# Overcharge Helpers
# =============================================================================


def create_overcharge_state(level: int = 0, uses: int = 0) -> OverchargeState:
    """Create a new overcharge state.

    Args:
        level: Current escalation level (0-3)
        uses: Uses this turn

    Returns:
        New OverchargeState
    """
    return OverchargeState(current_level=level, uses_this_turn=uses)


def use_overcharge(state: CombatantState) -> tuple[CombatantState, OverchargeState]:
    """Increment overcharge uses for the turn.

    Args:
        state: Current combatant state

    Returns:
        Tuple of (updated_combatant, new_overcharge_state)
    """
    current = state.overcharge_state or OverchargeState()
    new_state = OverchargeState(
        current_level=min(current.current_level + 1, 3),
        uses_this_turn=current.uses_this_turn + 1,
    )
    return state.model_copy(update={"overcharge_state": new_state}), new_state


def reset_overcharge_uses(state: CombatantState) -> CombatantState:
    """Reset overcharge uses at turn start (keeps level).

    Args:
        state: Current combatant state

    Returns:
        Updated combatant with overcharge uses reset to 0
    """
    if not state.overcharge_state:
        return state
    new_state = state.overcharge_state.model_copy(update={"uses_this_turn": 0})
    return state.model_copy(update={"overcharge_state": new_state})


# =============================================================================
# Turn-End Effect Helpers
# =============================================================================


def set_effect_duration(
    effect: TurnEndEffectState, new_duration: EffectDuration
) -> TurnEndEffectState:
    """Update the duration type of a turn-end effect.

    Args:
        effect: Current turn-end effect state
        new_duration: New duration type

    Returns:
        Updated effect with new duration
    """
    return effect.model_copy(update={"duration_type": new_duration})


def advance_effect_to_next_turn(effect: TurnEndEffectState) -> TurnEndEffectState:
    """Advance an end_of_next_turn effect to end_of_turn.

    Args:
        effect: Current turn-end effect state

    Returns:
        Updated effect with duration advanced
    """
    if effect.duration_type != "end_of_next_turn":
        return effect
    return effect.model_copy(update={"duration_type": "end_of_turn"})


# =============================================================================
# Damage Application Helpers
# =============================================================================


def apply_damage(
    state: CombatantState,
    damage: int,
    armor_piercing: int = 0,
) -> StateUpdateResult:
    """Apply damage to a combatant, accounting for armor.

    Per PR2: Damage is reduced by armor, then applied to HP.

    Args:
        state: Current combatant state
        damage: Incoming damage amount
        armor_piercing: Armor piercing value (reduces effective armor)

    Returns:
        StateUpdateResult with updated combatant and change summary
    """
    old_hp = state.resources.hp_current
    effective_armor = max(0, state.stats.armor - armor_piercing)
    net_damage = max(0, damage - effective_armor)
    new_hp = max(0, old_hp - net_damage)

    updated = set_hp(state, new_hp)
    return StateUpdateResult(
        updated_combatant=updated,
        changes_summary={"hp": (old_hp, new_hp), "damage_net": (0, net_damage)},
    )


def apply_heat_damage(state: CombatantState, heat_amount: int) -> StateUpdateResult:
    """Apply heat damage and return updated combatant.

    Args:
        state: Current combatant state
        heat_amount: Amount of heat to add

    Returns:
        StateUpdateResult with updated combatant and change summary
    """
    old_heat = state.resources.heat_current
    new_heat = old_heat + heat_amount
    updated = set_heat(state, new_heat)
    return StateUpdateResult(
        updated_combatant=updated,
        changes_summary={"heat": (old_heat, new_heat)},
    )


def apply_structure_damage(state: CombatantState) -> StateUpdateResult:
    """Apply 1 structure damage (for structure check resolution).

    Args:
        state: Current combatant state

    Returns:
        StateUpdateResult with updated combatant and change summary
    """
    old_struct = state.resources.structure_current
    new_struct = max(0, old_struct - 1)
    updated = set_structure(state, new_struct)
    return StateUpdateResult(
        updated_combatant=updated,
        changes_summary={"structure": (old_struct, new_struct)},
    )


def apply_overheat_result(
    state: CombatantState,
    stress_after: int,
    heat_cleared: bool = True,
    statuses_to_add: list[StatusType] | None = None,
    meltdown_state: MeltdownState | None = None,
) -> StateUpdateResult:
    """Apply full overheat result to combatant state.

    Per PR2 4654-4706: Clear all heat, mark stress, apply statuses.

    Args:
        state: Current combatant state
        stress_after: Stress value after marking damage
        heat_cleared: Whether to clear all heat (default True)
        statuses_to_add: Statuses to add from overheat result
        meltdown_state: Meltdown state if countdown triggered

    Returns:
        StateUpdateResult with updated combatant and change summary
    """
    old_stress = state.resources.stress_current
    new_stress = stress_after
    new_heat = 0 if heat_cleared else state.resources.heat_current

    resource_updates = {
        "stress_current": new_stress,
        "heat_current": new_heat,
    }

    statuses = list(state.statuses)
    if statuses_to_add:
        for s in statuses_to_add:
            if s not in statuses:
                statuses.append(s)

    updates: dict[str, object] = {
        "statuses": statuses,
        "resources": state.resources.model_copy(update=resource_updates),
    }

    if meltdown_state is not None:
        updates["meltdown_state"] = meltdown_state

    updated = state.model_copy(update=updates)
    return StateUpdateResult(
        updated_combatant=updated,
        changes_summary={
            "stress": (old_stress, new_stress),
            "heat": (state.resources.heat_current, new_heat),
        },
    )


# Rebuild CombatantState to resolve forward references
# This must be done after CombatantState is defined and types are available
try:
    from core.mech.combat_state import CombatantState
    from core.shared.turn_end import TurnEndEffectState
    from core.shared.heat import MeltdownState
    from core.shared.protocols import ProtocolState

    StateUpdateResult.model_rebuild(
        _types_namespace={
            "CombatantState": CombatantState,
            "TurnEndEffectState": TurnEndEffectState,
            "MeltdownState": MeltdownState,
            "ProtocolState": ProtocolState,
        }
    )
except ImportError:
    pass  # CombatantState not yet available during initial import
