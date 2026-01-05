"""NPC combat resolution for typed Lancer mechanics.

This module provides ability trigger resolution for NPCs during combat,
integrating with the broader combat state system.
"""

from typing import Literal, TYPE_CHECKING
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.effects import MechanicalEffect

if TYPE_CHECKING:
    from core.npc.models import NPCState, NPCAbility


TriggerContextType = Literal[
    "on_hit",
    "on_miss",
    "on_crit",
    "on_kill",
    "on_turn_start",
    "on_turn_end",
    "on_damaged",
    "on_attacked",
    "on_adjacent",
    "on_deploy",
    "on_destroyed",
    "on_initiative",
    "on_ally_killed",
    "on_hp_below_half",
    "on_damage_dealt",
]


class TriggerContext(FrozenModel):
    """Context for an NPC ability trigger.

    Contains information about the event that triggered the ability.
    """

    trigger_type: TriggerContextType
    source_id: str | None = None
    target_id: str | None = None
    damage_dealt: int | None = None
    hp_before: int | None = None
    hp_after: int | None = None
    is_critical: bool = False
    adjacent_allies: int = 0
    round_number: int | None = None


class NPCAbilityResolution(FrozenModel):
    """Result of triggering an NPC ability."""

    ability_id: str
    ability_name: str
    triggered: bool
    effect_result: MechanicalEffect | None = None
    uses_remaining: int | None = None
    message: str = ""


class NPCAbilityTracker(FrozenModel):
    """Tracks ability usage for an NPC in combat."""

    npc_id: str
    abilities_used: dict[str, int] = Field(default_factory=dict)

    def can_use_ability(self, ability: "NPCAbility") -> bool:
        """Check if an ability can still be used.

        Args:
            ability: The ability to check

        Returns:
            True if the ability can be used
        """
        if ability.uses_per_combat is None:
            return True
        used = self.abilities_used.get(ability.id, 0)
        return used < ability.uses_per_combat

    def mark_ability_used(self, ability: "NPCAbility") -> int:
        """Mark an ability as used and return remaining uses.

        Args:
            ability: The ability that was used

        Returns:
            Number of uses remaining
        """
        if ability.uses_per_combat is None:
            return -1  # Unlimited
        used = self.abilities_used.get(ability.id, 0)
        self.abilities_used[ability.id] = used + 1
        return ability.uses_per_combat - (used + 1)

    def get_uses_remaining(self, ability: "NPCAbility") -> int | None:
        """Get remaining uses for an ability.

        Args:
            ability: The ability to check

        Returns:
            Number of uses remaining, or None if unlimited
        """
        if ability.uses_per_combat is None:
            return None
        used = self.abilities_used.get(ability.id, 0)
        return ability.uses_per_combat - used


def get_abilities_for_trigger(
    npc: "NPCState",
    trigger_type: TriggerContextType,
) -> list["NPCAbility"]:
    """Get all abilities on an NPC that match a trigger type.

    Args:
        npc: The NPC to check
        trigger_type: The trigger type to match

    Returns:
        List of matching abilities
    """
    from core.npc.compendium import get_npc_template

    template = get_npc_template(npc.template_id) if npc.template_id else None
    if not template:
        return []
    return [a for a in template.abilities if a.trigger == trigger_type]


def resolve_npc_trigger(
    npc: "NPCState",
    tracker: NPCAbilityTracker,
    context: TriggerContext,
) -> list[NPCAbilityResolution]:
    """Resolve applicable ability triggers for an NPC.

    Args:
        npc: The NPC whose abilities are being checked
        tracker: Ability usage tracker for this NPC
        context: The trigger context

    Returns:
        List of resolution results for each triggered ability
    """
    results: list[NPCAbilityResolution] = []

    matching_abilities = get_abilities_for_trigger(npc, context.trigger_type)

    for ability in matching_abilities:
        if not tracker.can_use_ability(ability):
            results.append(
                NPCAbilityResolution(
                    ability_id=ability.id,
                    ability_name=ability.name,
                    triggered=False,
                    message="Ability already used maximum times",
                )
            )
            continue

        remaining = tracker.mark_ability_used(ability)

        results.append(
            NPCAbilityResolution(
                ability_id=ability.id,
                ability_name=ability.name,
                triggered=True,
                effect_result=ability.effect if ability.effect else None,
                uses_remaining=remaining if remaining >= 0 else None,
                message="Ability triggered successfully",
            )
        )

    return results


def create_npc_ability_tracker(npc: "NPCState") -> NPCAbilityTracker:
    """Create a new ability tracker for an NPC.

    Args:
        npc: The NPC to create a tracker for

    Returns:
        A new tracker with no abilities used
    """
    return NPCAbilityTracker(npc_id=npc.id)


def check_trigger_condition(
    ability: "NPCAbility",
    context: TriggerContext,
) -> bool:
    """Check if an ability's additional conditions are met.

    This handles triggers that have additional requirements beyond
    just the trigger type (e.g., on_hp_below_half requires HP check).

    Args:
        ability: The ability being checked
        context: The trigger context

    Returns:
        True if the ability should trigger
    """
    trigger = ability.trigger

    if trigger == "on_hp_below_half":
        if context.hp_before is not None and context.hp_after is not None:
            max_hp = context.hp_before
            if max_hp > 0:
                return context.hp_after < (max_hp / 2)
        elif context.hp_after is not None:
            return context.hp_after < 10  # Default threshold

    elif trigger == "on_adjacent":
        return context.adjacent_allies > 0

    elif trigger == "on_kill":
        return context.damage_dealt is not None and context.damage_dealt >= 0

    elif trigger == "on_damage_dealt":
        return context.damage_dealt is not None and context.damage_dealt > 0

    return True


def apply_ability_effect(
    effect: MechanicalEffect,
    combat_state: "CombatState",
    source_id: str,
    target_id: str | None = None,
) -> None:
    """Apply a MechanicalEffect to the combat state.

    This is a placeholder that integrates with the broader combat system.
    The actual effect application depends on the effect type.

    Args:
        effect: The effect to apply
        combat_state: The current combat state
        source_id: The NPC applying the effect
        target_id: Optional specific target
    """
    pass
