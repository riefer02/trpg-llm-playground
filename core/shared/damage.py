"""Damage resolution primitives for Lancer TTRPG.

Provides centralized, composable damage resolution per PR2 4538-4558 rules:
- Damage types (kinetic, explosive, energy, burn)
- Armor reduction (max 4 for mechs, burn ignores armor)
- Resistance application (half damage, rounded up, after armor)
- Condition modifiers (shredded = no armor/resistance, exposed = 2x damage)
- Integration with existing MechanicalEffect system
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import DamageType, StatusType
from core.shared.dice import round_up
from core.shared.id_helpers import CombatantIdField
from core.mech.combat_rules import DamageResolutionRules
from core.mech.combat_state import CombatantState, CombatStats
from core.mech.grid import HexCoord


class DamageInput(FrozenModel):
    """Input specification for damage resolution.

    Represents a single source of damage to be resolved against a target.
    """

    damage: int = Field(..., ge=0, description="Base damage amount")
    damage_type: DamageType = Field(
        ..., description="Type of damage (kinetic/explosive/energy/burn)"
    )
    armor_piercing: int = Field(
        default=0, ge=0, description="AP value that ignores armor"
    )
    bonus_damage: int = Field(
        default=0, ge=0, description="Additional kinetic/explosive/energy damage"
    )
    source: str | None = Field(
        default=None, description="Source identifier for effect tracking"
    )


class DamageResolutionContext(FrozenModel):
    """Context for damage resolution including attacker and target information.

    Provides situational modifiers and state information needed for
    accurate damage resolution.
    """

    attacker_id: CombatantIdField = Field(..., description="ID of the damage source")
    target: CombatantState = Field(..., description="Target combatant state")
    target_position: HexCoord | None = Field(
        default=None, description="Target grid position"
    )
    accuracy_bonus: int = Field(
        default=0, ge=0, description="Accuracy bonus affecting attack"
    )
    is_critical: bool = Field(
        default=False, description="Whether attack was a critical hit (20+)"
    )
    multi_target: bool = Field(
        default=False, description="Whether attack targets multiple entities"
    )
    rules: DamageResolutionRules | None = Field(
        default=None, description="Override resolution rules"
    )
    resistances: list[DamageType] = Field(
        default_factory=list,
        description="Damage types target is resistant to (half damage)",
    )


class DamageResolutionResult(FrozenModel):
    """Complete result of damage resolution.

    Provides detailed breakdown of damage calculation for debugging,
    logging, and integration with other systems.
    """

    raw_damage: int = Field(..., description="Damage after multipliers (exposed 2x)")
    bonus_damage_applied: int = Field(
        ..., description="Bonus damage after multi-target halving"
    )
    armor_reduction: int = Field(..., description="Amount reduced by armor")
    resistance_reduction: int = Field(..., description="Amount reduced by resistance")
    net_damage: int = Field(..., description="Final damage after all reductions")
    damage_to_hp: int = Field(..., description="Damage applied to HP")
    damage_to_heat: int = Field(default=0, description="Heat added from damage resolution")
    damage_type: DamageType = Field(..., description="Original damage type")
    is_shredded: bool = Field(
        ..., description="Target was shredded (no armor/resistance)"
    )
    is_exposed: bool = Field(..., description="Target was exposed (2x damage)")
    armor_ignored: bool = Field(..., description="AP ignored armor completely")
    burn_ignored_armor: bool = Field(
        ..., description="Burn ignored armor per PR2 rules"
    )


class DamageBreakdown(FrozenModel):
    """Net damage totals by type (heat tracked separately from damage)."""

    kinetic: int = Field(default=0, ge=0)
    explosive: int = Field(default=0, ge=0)
    energy: int = Field(default=0, ge=0)
    burn: int = Field(default=0, ge=0)
    heat: int = Field(default=0, ge=0)


def resolve_damage_on_target(
    input: DamageInput,
    context: DamageResolutionContext,
) -> DamageResolutionResult:
    """Resolve damage against a target per PR2 rules.

    Resolution order (PR2 4089-4093):
    1. Apply damage multipliers (exposed = 2x) before any reductions
    2. Apply armor reduction (max 4 for mechs, burn ignores armor)
    3. Apply resistance (half damage, rounded up, after armor)
    4. Apply other reductions (systems, talents, reactions)
    5. Deal remaining damage to HP

    Shredded condition prevents benefit from armor or resistance.

    Args:
        input: Damage specification to resolve
        context: Resolution context including target state

    Returns:
        Detailed breakdown of damage resolution
    """
    rules = context.rules or DamageResolutionRules()

    target = context.target
    target_stats = target.stats
    target_statuses = target.statuses

    is_exposed = "exposed" in target_statuses
    is_shredded = "shredded" in target_statuses

    raw_damage = input.damage

    if is_exposed:
        raw_damage = raw_damage * 2

    bonus_damage = input.bonus_damage
    if context.multi_target and bonus_damage > 0:
        bonus_damage = round_up(bonus_damage / 2)

    combined_damage = raw_damage + bonus_damage

    damage_type = input.damage_type
    armor_value = target_stats.armor
    max_armor = 4

    effective_armor = min(armor_value, max_armor)

    burn_ignored_armor = damage_type == "burn"

    armor_reduction = 0
    armor_ignored = False

    if not burn_ignored_armor and effective_armor > 0 and not is_shredded:
        if input.armor_piercing > 0:
            armor_ignored = True
            armor_reduction = 0
        else:
            armor_reduction = min(combined_damage, effective_armor)

    damage_after_armor = max(0, combined_damage - armor_reduction)

    resistance_reduction = 0
    if not is_shredded:
        if damage_type in context.resistances:
            resistance_reduction = round_up(damage_after_armor / 2)

    damage_after_resistance = max(0, damage_after_armor - resistance_reduction)

    damage_to_hp = damage_after_resistance
    damage_to_heat = 0

    return DamageResolutionResult(
        raw_damage=raw_damage,
        bonus_damage_applied=bonus_damage,
        armor_reduction=armor_reduction,
        resistance_reduction=resistance_reduction,
        net_damage=damage_after_resistance,
        damage_to_hp=damage_to_hp,
        damage_to_heat=damage_to_heat,
        damage_type=damage_type,
        is_shredded=is_shredded,
        is_exposed=is_exposed,
        armor_ignored=armor_ignored if not burn_ignored_armor else True,
        burn_ignored_armor=burn_ignored_armor,
    )


def apply_damage_to_combatant(
    input: DamageInput,
    context: DamageResolutionContext,
) -> tuple[CombatantState, DamageResolutionResult]:
    """Apply resolved damage to a combatant's state.

    Updates the target's HP and heat tracks based on damage resolution.

    Args:
        input: Damage specification to apply
        context: Resolution context including target state

    Returns:
        Tuple of (updated combatant state, damage resolution result)
    """
    result = resolve_damage_on_target(input, context)

    target = context.target

    new_hp = max(0, target.resources.hp_current - result.damage_to_hp)
    new_heat = target.resources.heat_current + result.damage_to_heat

    updated_resources = target.resources.model_copy(
        update={"hp_current": new_hp, "heat_current": new_heat}
    )

    updated_target = target.model_copy(update={"resources": updated_resources})

    return updated_target, result


def compute_damage_before_reductions(
    base_damage: int,
    damage_type: DamageType,
    is_exposed: bool,
    bonus_damage: int = 0,
    multi_target: bool = False,
) -> int:
    """Compute damage before armor/resistance reductions.

    Applies exposed multiplier and halves bonus damage for multi-target attacks.

    Args:
        base_damage: Base damage from weapon/system
        damage_type: Type of damage being dealt
        is_exposed: Whether target is exposed
        bonus_damage: Additional bonus damage
        multi_target: Whether attack targets multiple entities

    Returns:
        Damage amount before armor/resistance
    """
    damage = base_damage

    if is_exposed:
        damage = damage * 2

    if multi_target and bonus_damage > 0:
        bonus_damage = round_up(bonus_damage / 2)

    return damage + bonus_damage


def compute_armor_reduction(
    damage: int,
    damage_type: DamageType,
    armor: int,
    armor_piercing: int,
    is_shredded: bool,
) -> tuple[int, bool]:
    """Compute armor reduction for incoming damage.

    Args:
        damage: Damage amount before armor
        damage_type: Type of damage
        armor: Target's armor value
        armor_piercing: Attack's AP value
        is_shredded: Whether target is shredded

    Returns:
        Tuple of (armor reduction amount, whether armor was ignored)
    """
    if damage_type == "burn":
        return 0, True

    if is_shredded:
        return 0, False

    max_armor = 4
    effective_armor = min(armor, max_armor)

    if effective_armor <= 0:
        return 0, False

    if armor_piercing > 0:
        return 0, True

    return min(damage, effective_armor), False


def compute_resistance_reduction(
    damage: int,
    damage_type: DamageType,
    is_shredded: bool,
    resistances: list[DamageType] | None = None,
) -> int:
    """Compute resistance reduction for incoming damage.

    Per PR2: Resistance reduces damage by half, rounded up.
    Resistance does not stack (only one reduction applies).
    Args:
        damage: Damage amount after armor
        damage_type: Type of damage
        is_shredded: Whether target is shredded (resistance ignored)
        resistances: List of damage types target is resistant to

    Returns:
        Resistance reduction amount
    """
    if is_shredded:
        return 0

    if resistances is None:
        resistances = []

    if damage_type not in resistances:
        return 0

    return round_up(damage / 2)
