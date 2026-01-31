"""
Damage and hit prediction utilities for combat actions.

Core-first: all prediction logic lives in core, using core models.
"""

from __future__ import annotations

from typing import Any
from core.mech.combat_state import CombatantState
from core.mech.compendium import get_weapon_definition
from core.mech.weapon import WeaponDamage, resolve_weapon_profile
from core.shared.rolls import resolve_attack, AttackResolutionResult
from core.shared.dice import DiceExpression
import random


def compute_damage_stats(damage_components: list[WeaponDamage]) -> dict[str, Any]:
    """Compute min, max, average damage across all damage components."""
    total_min = 0
    total_max = 0
    total_avg = 0.0
    damage_types = []
    for dmg in damage_components:
        # Dice component
        dice_min = dmg.dice.min_value() if dmg.dice else 0
        dice_max = dmg.dice.max_value() if dmg.dice else 0
        dice_avg = dmg.dice.average() if dmg.dice else 0.0
        # Flat component
        flat = dmg.flat
        total_min += dice_min + flat
        total_max += dice_max + flat
        total_avg += dice_avg + flat
        if dmg.damage_type:
            damage_types.append(dmg.damage_type)
    # Deduplicate damage types
    unique_types = list(set(damage_types))
    return {
        "min": total_min,
        "max": total_max,
        "average": total_avg,
        "damage_types": unique_types,
    }


def compute_average_weapon_damage(combatant: CombatantState) -> dict[str, Any] | None:
    """Compute average damage for the combatant's first usable weapon.

    Returns dict with min, max, average damage, or None if no weapons.
    """
    if combatant.inventory is None:
        return None
    for mount in combatant.inventory.mounts:
        if mount.destroyed:
            continue
        for weapon_state in mount.weapons:
            if weapon_state.destroyed:
                continue
            try:
                weapon_def = get_weapon_definition(weapon_state.weapon_id)
                if weapon_def is None:
                    continue
                # Resolve to weapon profile
                profile = resolve_weapon_profile(weapon_def)
                damage_stats = compute_damage_stats(profile.damage)
                # Include weapon name and id for reference
                damage_stats["weapon_id"] = weapon_state.weapon_id
                damage_stats["weapon_name"] = profile.name
                return damage_stats
            except Exception:
                # Weapon not found or other error
                continue
    return None


def estimate_hit_probability(
    attack_bonus: int,
    target_defense: int,
    accuracy_bonus: int = 0,
    difficulty_bonus: int = 0,
    samples: int = 10000,
) -> float:
    """Estimate probability of hitting a target using Monte Carlo simulation.

    Uses resolve_attack with random rolls, returns proportion of hits.
    For deterministic results, pass a fixed random seed.
    """
    if samples <= 0:
        samples = 10000
    hits = 0
    for _ in range(samples):
        # Use forced_roll to simulate random roll (1-20)
        forced_roll = random.randint(1, 20)
        result = resolve_attack(
            attack_bonus=attack_bonus,
            target_defense=target_defense,
            accuracy_bonus=accuracy_bonus,
            difficulty_bonus=difficulty_bonus,
            forced_roll=forced_roll,
        )
        if result.hit:
            hits += 1
    return hits / samples


def predict_action_preview(
    attacker: CombatantState,
    target: CombatantState,
    action_id: str,
    weapon_id: str | None = None,
) -> dict[str, Any]:
    """Generate preview of action outcome (damage, hit chance, etc.)

    Args:
        attacker: The actor performing the action.
        target: The target combatant (if required).
        action_id: Action identifier (e.g., 'skirmish', 'barrage').
        weapon_id: Optional specific weapon ID (if action requires weapon).

    Returns:
        Dictionary with prediction data.
    """
    # TODO: Implement based on action type
    # For now, placeholder
    return {}
