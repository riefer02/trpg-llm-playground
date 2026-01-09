"""Improvised Attack action resolution primitives for Lancer TTRPG.

Implements resolution logic for Improvised Attack per PR2 4265-4271:
- Full action melee attack when mech is unarmed
- Counts as a melee attack (uses standard attack resolution)
- Deals 1d6 kinetic damage on hit

Resolution Pattern:
1. resolve_improvised_attack() - Pure resolution logic
2. apply_improvised_result() - Apply to combatant state
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import DamageType, AttackType
from core.shared.dice import DiceExpression, round_up
from core.shared.id_helpers import CombatantIdField
from core.mech.combat_state import CombatantState
from core.shared.rolls import resolve_attack


class ImprovisedRule(FrozenModel):
    """Rule configuration for Improvised Attack action."""

    damage: DiceExpression = Field(
        default_factory=lambda: DiceExpression.parse("1d6"),
        description="Damage expression for improvised attack",
    )
    damage_type: DamageType = "kinetic"
    requires_unarmed: bool = True
    attack_type: AttackType = "melee"


DEFAULT_IMPROVISED_RULES = ImprovisedRule()


class ImprovisedInput(FrozenModel):
    """Input for Improvised Attack resolution."""

    actor_id: str = Field(
        ..., description="ID of the mech making the improvised attack"
    )
    target_id: CombatantIdField = Field(..., description="ID of the target")
    is_unarmed: bool = Field(..., description="Whether the actor is unarmed")
    rules: ImprovisedRule | None = Field(
        default=None, description="Override resolution rules"
    )


class ImprovisedResolutionResult(FrozenModel):
    """Complete result of Improvised Attack resolution (pure logic)."""

    actor_id: str = Field(..., description="ID of the attacker")
    target_id: str = Field(..., description="ID of the target")
    is_unarmed: bool = Field(..., description="Whether attacker was unarmed")
    attack_success: bool = Field(
        default=False, description="Whether the attack roll succeeded"
    )
    accuracy_roll: int | None = Field(default=None, description="Accuracy die roll")
    accuracy_bonus: int = Field(
        default=0, description="Accuracy bonus from stats/conditions"
    )
    total_accuracy: int | None = Field(
        default=None, description="Total accuracy (roll + bonus)"
    )
    target_evasion: int | None = Field(default=None, description="Target's evasion")
    target_e_defense: int | None = Field(default=None, description="Target's e-defense")
    hit: bool = Field(default=False, description="Whether attack hit")
    damage_expression: DiceExpression = Field(
        default_factory=lambda: DiceExpression.parse("1d6"),
        description="Damage expression to roll",
    )
    damage_type: DamageType = "kinetic"
    damage_on_hit: int | None = Field(
        default=None, description="Damage amount if hit (after dice roll)"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class ImprovisedApplicationResult(FrozenModel):
    """Result of applying Improvised Attack result to combatant state."""

    target_hit: bool = Field(
        default=False, description="Whether the attack hit the target"
    )
    damage_dealt: int = Field(default=0, ge=0, description="Damage dealt to target")
    damage_type: DamageType = "kinetic"
    target_conditions: list[str] = Field(
        default_factory=list, description="Conditions inflicted on target"
    )


def resolve_improvised_attack(
    input: ImprovisedInput,
    actor_accuracy_bonus: int = 0,
    target_evasion: int = 10,
    target_e_defense: int = 10,
    forced_roll: int | None = None,
) -> ImprovisedResolutionResult:
    """Resolve an Improvised Attack per PR2 4265-4271.

    Improvised Attack is a Full Action melee attack that deals 1d6 kinetic
    damage on hit. It uses standard attack resolution (accuracy vs evasion).

    Args:
        input: Improvised Attack input with attacker and target information
        actor_accuracy_bonus: Attacker's accuracy bonus (from systems, talents, etc.)
        target_evasion: Target's evasion value
        target_e_defense: Target's e-defense value (used if attack is tech)
        forced_roll: Optional forced roll value for deterministic testing

    Returns:
        Detailed breakdown of what should happen during the attack
    """
    if input.rules is None:
        rules = DEFAULT_IMPROVISED_RULES
    else:
        rules = input.rules

    errors: list[str] = []

    if rules.requires_unarmed and not input.is_unarmed:
        errors.append("Mech is not unarmed - cannot use improvised attack")

    if errors:
        return ImprovisedResolutionResult(
            actor_id=input.actor_id,
            target_id=input.target_id,
            is_unarmed=input.is_unarmed,
            attack_success=False,
            hit=False,
            damage_expression=rules.damage,
            damage_type=rules.damage_type,
            damage_on_hit=None,
            validation_errors=errors,
        )

    if rules.attack_type == "melee":
        target_defense = target_evasion
    else:
        target_defense = target_e_defense

    attack_result = resolve_attack(
        attack_bonus=actor_accuracy_bonus,
        target_defense=target_defense,
        forced_roll=forced_roll,
    )

    hit = attack_result.hit

    damage_on_hit = None
    if hit:
        damage_dice = rules.damage.roll()
        damage_on_hit = sum(damage_dice)

    return ImprovisedResolutionResult(
        actor_id=input.actor_id,
        target_id=input.target_id,
        is_unarmed=input.is_unarmed,
        attack_success=True,
        accuracy_roll=attack_result.roll,
        accuracy_bonus=actor_accuracy_bonus,
        total_accuracy=attack_result.total_accuracy,
        target_evasion=target_evasion,
        target_e_defense=target_e_defense,
        hit=hit,
        damage_expression=rules.damage,
        damage_type=rules.damage_type,
        damage_on_hit=damage_on_hit,
        validation_errors=errors,
    )


def apply_improvised_result(
    target: CombatantState,
    result: ImprovisedResolutionResult,
) -> ImprovisedApplicationResult:
    """Apply Improvised Attack result to combatant state.

    Updates target with damage from the attack.

    Args:
        target: Current target combatant state
        result: Resolution result to apply

    Returns:
        Updated target with attack effects applied
    """
    if not result.hit or result.damage_on_hit is None:
        return ImprovisedApplicationResult(
            target_hit=False,
            damage_dealt=0,
            damage_type=result.damage_type,
            target_conditions=[],
        )

    new_hp = max(0, target.resources.hp_current - result.damage_on_hit)
    updated_resources = target.resources.model_copy(update={"hp_current": new_hp})
    updated_target = target.model_copy(update={"resources": updated_resources})

    return ImprovisedApplicationResult(
        target_hit=True,
        damage_dealt=result.damage_on_hit,
        damage_type=result.damage_type,
        target_conditions=list(updated_target.conditions),
    )


try:
    from core.shared.heat import MeltdownState
    from core.mech.combat_state import CombatantState

    CombatantState.model_rebuild(
        _types_namespace={"MeltdownState": MeltdownState, "StabilizeState": type(None)}
    )
except ImportError:
    pass
