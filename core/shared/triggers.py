"""Universal trigger resolution system for typed Lancer mechanics.

This module provides a unified trigger resolution system that works for
both player mechs and NPCs, ensuring consistent trigger handling across
the entire game system.

Triggers are defined in TriggerType (core.shared.effects) and this module
provides resolution logic for all trigger types.
"""

from typing import TYPE_CHECKING
from core.shared.effects import TriggerType, MechanicalEffect

if TYPE_CHECKING:
    pass


class TriggerContext:
    """Context for trigger resolution.

    Contains information about the event that triggered the ability.
    Used by both player mech and NPC trigger resolution.
    """

    def __init__(
        self,
        trigger_type: TriggerType,
        source_id: str | None = None,
        target_id: str | None = None,
        damage_dealt: int | None = None,
        hp_before: int | None = None,
        hp_after: int | None = None,
        is_critical: bool = False,
        adjacent_allies: int = 0,
        round_number: int | None = None,
    ):
        self.trigger_type = trigger_type
        self.source_id = source_id
        self.target_id = target_id
        self.damage_dealt = damage_dealt
        self.hp_before = hp_before
        self.hp_after = hp_after
        self.is_critical = is_critical
        self.adjacent_allies = adjacent_allies
        self.round_number = round_number


class TriggerResolution:
    """Result of trigger resolution."""

    def __init__(
        self,
        triggered: bool,
        effects: list[MechanicalEffect] | None = None,
        message: str = "",
    ):
        self.triggered = triggered
        self.effects = effects or []
        self.message = message


def check_hp_below_half(hp_before: int | None, hp_after: int | None) -> bool:
    """Check if HP is below half of maximum.

    Args:
        hp_before: HP before the triggering event
        hp_after: HP after the triggering event

    Returns:
        True if HP is below half of maximum
    """
    if hp_before is not None and hp_after is not None:
        max_hp = hp_before
        if max_hp > 0:
            return hp_after < (max_hp / 2)
    elif hp_after is not None:
        return hp_after < 10  # Default threshold
    return False


def check_adjacent(adjacent_allies: int) -> bool:
    """Check if there are adjacent allies.

    Args:
        adjacent_allies: Number of adjacent allied units

    Returns:
        True if there are adjacent allies
    """
    return adjacent_allies > 0


def check_damage_dealt(damage_dealt: int | None) -> bool:
    """Check if damage was dealt.

    Args:
        damage_dealt: Amount of damage dealt

    Returns:
        True if damage was dealt
    """
    return damage_dealt is not None and damage_dealt > 0


def check_kill(damage_dealt: int | None) -> bool:
    """Check if a kill occurred.

    Args:
        damage_dealt: Amount of damage dealt

    Returns:
        True if damage was dealt (kill is implied)
    """
    return damage_dealt is not None and damage_dealt >= 0


TRIGGER_CONDITION_CHECKS: dict[TriggerType, callable] = {
    "on_hp_below_half": lambda ctx: check_hp_below_half(ctx.hp_before, ctx.hp_after),
    "on_adjacent": lambda ctx: check_adjacent(ctx.adjacent_allies),
    "on_damage_dealt": lambda ctx: check_damage_dealt(ctx.damage_dealt),
    "on_kill": lambda ctx: check_kill(ctx.damage_dealt),
}


NPC_ONLY_TRIGGERS: set[TriggerType] = {
    "on_deploy",
    "on_destroyed",
    "on_adjacent",
    "on_ally_killed",
    "on_hp_below_half",
    "on_first_adjacent_turn",
}


def is_valid_for_player(trigger: TriggerType) -> bool:
    """Check if a trigger is valid for player mechs.

    Player mechs cannot have NPC-specific triggers like on_deploy or on_ally_killed.

    Args:
        trigger: The trigger type to check

    Returns:
        True if the trigger is valid for player mechs
    """
    return trigger not in NPC_ONLY_TRIGGERS


def is_valid_for_npc(trigger: TriggerType) -> bool:
    """Check if a trigger is valid for NPCs.

    NPCs can have all triggers including NPC-specific ones.

    Args:
        trigger: The trigger type to check

    Returns:
        True if the trigger is valid for NPCs (always True for now)
    """
    return True


def check_trigger_condition(trigger: TriggerType, context: TriggerContext) -> bool:
    """Check if a trigger's additional conditions are met.

    Some triggers require additional context beyond just the trigger type.
    For example, on_hp_below_half requires checking HP values.

    Args:
        trigger: The trigger type to check
        context: The trigger context with additional information

    Returns:
        True if the trigger should fire
    """
    checker = TRIGGER_CONDITION_CHECKS.get(trigger)
    if checker:
        return checker(context)
    return True


def resolve_trigger(
    trigger: TriggerType,
    context: TriggerContext,
    is_npc: bool = False,
) -> TriggerResolution:
    """Resolve a trigger and return effects to apply.

    This is the main entry point for trigger resolution, used by both
    player mech and NPC systems.

    Args:
        trigger: The trigger type that occurred
        context: Additional context about the triggering event
        is_npc: Whether the entity is an NPC (affects valid triggers)

    Returns:
        TriggerResolution with triggered flag and effects
    """
    if is_npc:
        if not is_valid_for_npc(trigger):
            return TriggerResolution(
                triggered=False,
                message=f"Trigger {trigger} is not valid for NPCs",
            )
    else:
        if not is_valid_for_player(trigger):
            return TriggerResolution(
                triggered=False,
                message=f"Trigger {trigger} is not valid for player mechs",
            )

    if not check_trigger_condition(trigger, context):
        return TriggerResolution(
            triggered=False,
            message=f"Trigger condition not met for {trigger}",
        )

    return TriggerResolution(
        triggered=True,
        message=f"Trigger {trigger} resolved successfully",
    )
