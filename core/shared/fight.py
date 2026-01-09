"""Pilot Fight action resolution for Lancer TTRPG.

Implements the Fight action per PR2 4441-4442:
- Full action to make a melee or ranged attack with one weapon
- Roll 1d20 + grit bonus
- Ranged attacks: +1 difficulty if engaged, take cover into account
- Sidearm tag weapons can be fired as quick action instead of full action

Resolution Pattern:
1. resolve_fight() - Pure resolution logic using resolve_attack()
2. apply_fight_result() - Apply damage to combatant state
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import DamageType
from core.shared.id_helpers import CombatantIdField
from core.shared.rolls import resolve_attack, AttackResolutionResult

if TYPE_CHECKING:
    from core.mech.combat_state import CombatantState


ActionTypeLiteral = Literal["full", "quick"]


class FightInput(FrozenModel):
    """Input for Fight action resolution.

    Captures all information needed to resolve a pilot Fight action.
    """

    actor_id: str = Field(..., description="ID of the pilot making the attack")
    target_id: CombatantIdField = Field(..., description="ID of the target")
    weapon_id: str | None = Field(
        default=None,
        description="Pilot weapon ID (None for unarmed/improvised)",
    )
    is_ranged: bool = Field(
        default=False,
        description="Whether this is a ranged attack (vs melee)",
    )
    is_engaged: bool = Field(
        default=False,
        description="Whether the actor is engaged (affects ranged attacks)",
    )
    target_has_cover: bool = Field(
        default=False,
        description="Whether the target has soft cover",
    )
    target_is_mech: bool = Field(
        default=False,
        description="Whether the target is a mech (affects archaic weapons)",
    )
    grit_bonus: int = Field(
        default=0,
        description="Pilot's GRIT bonus added to attack roll",
    )
    target_evasion: int = Field(
        default=10,
        ge=0,
        description="Target's evasion value (for melee/ranged attacks)",
    )
    target_e_defense: int = Field(
        default=10,
        ge=0,
        description="Target's e-defense value (for tech attacks)",
    )
    accuracy_bonus: int = Field(
        default=0,
        ge=0,
        description="Additional accuracy dice from systems/talents",
    )
    damage_flat: int = Field(
        default=0,
        description="Flat damage bonus (from weapon profile)",
    )


class FightResolutionResult(FrozenModel):
    """Complete result of Fight action resolution (pure logic).

    Provides detailed breakdown of attack resolution and damage calculation.
    """

    actor_id: str = Field(..., description="ID of the attacking pilot")
    target_id: str = Field(..., description="ID of the target")
    weapon_id: str | None = Field(
        default=None,
        description="Weapon used (None for unarmed)",
    )
    action_type: ActionTypeLiteral = Field(
        default="full",
        description="Action type required (quick for sidearms)",
    )
    attack_result: AttackResolutionResult = Field(
        ...,
        description="Detailed attack resolution from resolve_attack()",
    )
    damage_flat: int = Field(
        default=0,
        description="Flat damage from weapon",
    )
    damage_type: DamageType = Field(
        default="kinetic",
        description="Damage type of the attack",
    )
    damage_on_hit: int | None = Field(
        default=None,
        description="Damage dealt if attack hits (None if miss)",
    )
    is_critical: bool = Field(
        default=False,
        description="Whether attack was a critical hit",
    )
    hit: bool = Field(
        default=False,
        description="Whether attack hit the target",
    )
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Any validation errors encountered",
    )
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal validation issues",
    )


class FightApplicationResult(FrozenModel):
    """Result of applying Fight action result to combatant state."""

    target_hit: bool = Field(
        default=False,
        description="Whether the attack hit the target",
    )
    damage_dealt: int = Field(
        default=0,
        ge=0,
        description="Damage dealt to target HP",
    )
    heat_damage: int = Field(
        default=0,
        ge=0,
        description="Heat dealt to target",
    )
    damage_type: DamageType = Field(
        default="kinetic",
        description="Type of damage dealt",
    )
    target_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions inflicted on target",
    )
    target_statuses: list[str] = Field(
        default_factory=list,
        description="Statuses inflicted on target",
    )


def resolve_fight(
    input: FightInput,
    forced_roll: int | None = None,
    forced_accuracy_rolls: list[int] | None = None,
    forced_difficulty_rolls: list[int] | None = None,
) -> FightResolutionResult:
    """Resolve a pilot Fight action per PR2 4441-4442.

    Fight is a Full Action that makes one attack with a pilot weapon.
    Sidearm weapons can be used as a Quick Action instead.

    Attack Resolution:
    - Roll 1d20 + grit_bonus
    - Apply difficulty modifiers: +1 if engaged (ranged only), +1 if target has cover
    - Use resolve_attack() for consistent hit detection

    Args:
        input: Fight action input with attacker, target, and weapon information
        forced_roll: Optional forced 1d20 roll for deterministic testing
        forced_accuracy_rolls: Optional forced accuracy die rolls for testing
        forced_difficulty_rolls: Optional forced difficulty die rolls for testing

    Returns:
        Detailed breakdown of the Fight action resolution
    """
    from core.pilot.gear import (
        get_pilot_gear_definition,
        is_sidearm_weapon,
        is_archaic_weapon,
        can_pilot_weapon_damage_target,
        get_pilot_weapon_difficulty_modifier,
    )

    errors: list[str] = []
    warnings: list[str] = []

    weapon_damage_flat = input.damage_flat
    weapon_damage_type: DamageType = "kinetic"
    is_sidearm = False
    weapon_inaccuracy_modifier = 0

    if input.weapon_id:
        weapon = get_pilot_gear_definition(input.weapon_id)
        if weapon is None:
            errors.append(f"Unknown weapon ID: {input.weapon_id}")
        elif weapon.weapon_profile is None:
            errors.append(f"Weapon '{input.weapon_id}' has no weapon profile")
        else:
            is_sidearm = is_sidearm_weapon(weapon.tags)
            weapon_damage_flat = weapon.weapon_profile.damage.flat
            weapon_damage_type = weapon.weapon_profile.damage.damage_type
            weapon_inaccuracy_modifier = get_pilot_weapon_difficulty_modifier(
                weapon.tags
            )

            can_damage, damage_reason = can_pilot_weapon_damage_target(
                weapon, input.target_is_mech
            )
            if not can_damage:
                errors.append(damage_reason)

    if errors:
        return FightResolutionResult(
            actor_id=input.actor_id,
            target_id=input.target_id,
            weapon_id=input.weapon_id,
            action_type="full",
            attack_result=AttackResolutionResult(
                roll=1,
                attack_bonus=input.grit_bonus,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=1 + input.grit_bonus,
                target_defense=input.target_evasion,
                hit=False,
                is_critical=False,
                miss_by=input.target_evasion - (1 + input.grit_bonus),
            ),
            damage_flat=weapon_damage_flat,
            damage_type=weapon_damage_type,
            damage_on_hit=None,
            is_critical=False,
            hit=False,
            validation_errors=errors,
            validation_warnings=warnings,
        )

    action_type: ActionTypeLiteral = "quick" if is_sidearm else "full"

    difficulty_bonus = weapon_inaccuracy_modifier
    if input.is_ranged:
        if input.is_engaged:
            difficulty_bonus += 1
        if input.target_has_cover:
            difficulty_bonus += 1

    target_defense = input.target_e_defense if input.is_ranged else input.target_evasion

    attack_result = resolve_attack(
        attack_bonus=input.grit_bonus,
        target_defense=target_defense,
        accuracy_bonus=input.accuracy_bonus,
        difficulty_bonus=difficulty_bonus,
        forced_roll=forced_roll,
        forced_accuracy_rolls=forced_accuracy_rolls,
        forced_difficulty_rolls=forced_difficulty_rolls,
    )

    damage_on_hit = None
    if attack_result.hit:
        damage_on_hit = weapon_damage_flat
        if attack_result.is_critical:
            damage_on_hit *= 2

    return FightResolutionResult(
        actor_id=input.actor_id,
        target_id=input.target_id,
        weapon_id=input.weapon_id,
        action_type=action_type,
        attack_result=attack_result,
        damage_flat=weapon_damage_flat,
        damage_type=weapon_damage_type,
        damage_on_hit=damage_on_hit,
        is_critical=attack_result.is_critical,
        hit=attack_result.hit,
        validation_errors=errors,
        validation_warnings=warnings,
    )


def apply_fight_result(
    target: CombatantState,
    result: FightResolutionResult,
) -> FightApplicationResult:
    """Apply Fight action result to combatant state.

    Updates target with damage from the attack.

    Args:
        target: Current target combatant state
        result: Resolution result to apply

    Returns:
        Updated target with attack effects applied
    """
    if not result.hit or result.damage_on_hit is None:
        return FightApplicationResult(
            target_hit=False,
            damage_dealt=0,
            heat_damage=0,
            damage_type=result.damage_type,
            target_conditions=[],
            target_statuses=[],
        )

    new_hp = max(0, target.resources.hp_current - result.damage_on_hit)
    updated_resources = target.resources.model_copy(update={"hp_current": new_hp})
    updated_target = target.model_copy(update={"resources": updated_resources})

    return FightApplicationResult(
        target_hit=True,
        damage_dealt=result.damage_on_hit,
        heat_damage=0,
        damage_type=result.damage_type,
        target_conditions=list(updated_target.conditions),
        target_statuses=list(updated_target.statuses),
    )
