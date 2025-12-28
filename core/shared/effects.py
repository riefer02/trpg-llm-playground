"""
Structured mechanical effect primitives for Lancer TTRPG.

This module defines composable effect building blocks that encode
game mechanics as structured data rather than description strings.

Legal Note: These represent pure game mechanics (allowed under the
Lancer Third Party License), not copyrighted expression/flavor text.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

from core.shared.enums import ActionType, DamageType, StatusType
from core.shared.dice import DiceExpression


# Stats that can be modified
StatType = Literal[
    # Pilot stats
    "hp",
    "armor", 
    "evasion",
    "e_defense",
    "speed",
    # Mech stats
    "heat_cap",
    "repair_cap",
    "tech_attack",
    "save_target",
    "sensor_range",
    "size",
    # Mounts
    "limited_bonus",  # +N to Limited weapon uses
    # System budget
    "system_points",
]

# Conditions for conditional effects
ConditionType = Literal[
    # Target conditions
    "target_prone",
    "target_immobilized",
    "target_below_half_hp",
    "target_has_lock_on",
    "target_larger",
    "target_smaller",
    "target_same_size",
    "target_hidden",
    "target_jammed",
    "target_shredded",
    "target_impaired",
    "target_slowed",
    "target_stunned",
    "target_exposed",
    "target_invisible",
    "target_engaged",
    # Attacker conditions
    "after_boost",
    "after_move_8_plus",
    "in_danger_zone",
    "hidden",
    "engaged",
    "exposed",
    "overheated",
    # Attack type conditions
    "melee_attack",
    "ranged_attack",
    "tech_attack",
    # Save conditions
    "save_vs_knockback_or_prone",
]


class StatModifier(BaseModel):
    """
    Numeric modifier to a stat.
    
    Examples:
        StatModifier(stat="hp", value=5)  # +5 HP
        StatModifier(stat="size", value=1)  # +1 Size
    """
    stat: StatType
    value: int
    
    model_config = {"frozen": True}


class DamageModifier(BaseModel):
    """
    Bonus damage under specific conditions.
    
    Examples:
        DamageModifier(flat=1, condition="melee_attack")  # +1 melee damage
        DamageModifier(dice=DiceExpression.parse("1d6"), damage_type="kinetic")  # +1d6 kinetic
    """
    dice: DiceExpression | None = Field(default=None, description="Bonus dice (e.g., DiceExpression.parse('1d6'))")
    flat: int = Field(default=0, description="Flat bonus damage")
    damage_type: DamageType | None = Field(default=None)
    condition: ConditionType | None = Field(default=None)
    
    model_config = {"frozen": True}


class RangeModifier(BaseModel):
    """
    Modifier to range or threat.
    
    Examples:
        RangeModifier(range_type="threat", value=1)  # +1 Threat
        RangeModifier(range_type="range", value=5)  # +5 Range
    """
    range_type: Literal["range", "threat", "sensors"]
    value: int
    
    model_config = {"frozen": True}


class ActionGrant(BaseModel):
    """
    Grants a new action or ability.
    
    Examples:
        ActionGrant(action_type="quick", name="Afterburner", trigger="after_boost")
        ActionGrant(action_type="reaction", name="Juke", trigger="on_successful_agility_save")
    """
    action_type: ActionType
    name: str
    trigger: str | None = Field(default=None, description="When this action can be used")
    uses_per: Literal["unlimited", "round", "scene", "mission"] = "unlimited"
    
    model_config = {"frozen": True}


class Immunity(BaseModel):
    """
    Immunity to a condition, damage type, or effect.
    
    Examples:
        Immunity(target="burn")  # Immune to Burn
        Immunity(target="knockback", condition="from_smaller")  # Immune to knockback from smaller
    """
    target: str = Field(..., description="What you're immune to (condition, damage type, or effect)")
    condition: str | None = Field(default=None, description="Conditional immunity")
    
    model_config = {"frozen": True}


class Resistance(BaseModel):
    """
    Resistance (half damage) to a damage type.
    
    Examples:
        Resistance(damage_type="energy")
        Resistance(damage_type="all", condition="can_see_bondmate")
    """
    damage_type: DamageType | Literal["all"]
    condition: str | None = Field(default=None)
    
    model_config = {"frozen": True}


class AccuracyModifier(BaseModel):
    """
    Modifier to accuracy/difficulty on rolls.
    
    Examples:
        AccuracyModifier(value=1, condition="target_has_lock_on")  # +1 Accuracy vs Lock On
        AccuracyModifier(value=-1)  # -1 Accuracy (difficulty)
    """
    value: int = Field(..., description="Positive = accuracy, negative = difficulty")
    condition: ConditionType | str | None = Field(default=None)
    applies_to: Literal["all", "melee", "ranged", "tech"] = "all"
    
    model_config = {"frozen": True}


class MovementGrant(BaseModel):
    """
    Grants movement or teleportation.
    
    Examples:
        MovementGrant(spaces=2, movement_type="fly", trigger="on_successful_save")
        MovementGrant(spaces=3, movement_type="teleport", trigger="after_boost")
    """
    spaces: int
    movement_type: Literal["walk", "fly", "teleport"] = "walk"
    trigger: str | None = Field(default=None)
    
    model_config = {"frozen": True}


class StatusGrant(BaseModel):
    """
    Grants or inflicts a status condition.
    
    Examples:
        StatusGrant(status="invisible", target="self", trigger="after_move_8_plus")
        StatusGrant(status="prone", target="enemy", trigger="on_hit")
    """
    status: StatusType | Literal["invisible", "flying"]
    target: Literal["self", "enemy", "ally", "adjacent"]
    trigger: str | None = Field(default=None)
    duration: Literal["end_of_turn", "start_of_next_turn", "until_attack", "scene"] = "end_of_turn"
    
    model_config = {"frozen": True}


class MechanicalEffect(BaseModel):
    """
    Composable mechanical effect for talents, core bonuses, systems, etc.
    
    This is the primary building block for encoding game mechanics
    without relying on natural language descriptions.
    
    Example:
        MechanicalEffect(
            stat_mods=[StatModifier(stat="hp", value=5)],
            immunities=[Immunity(target="burn")],
        )
    """
    # Stat modifications
    stat_mods: list[StatModifier] = Field(default_factory=list)
    accuracy_mods: list[AccuracyModifier] = Field(default_factory=list)
    damage_mods: list[DamageModifier] = Field(default_factory=list)
    range_mods: list[RangeModifier] = Field(default_factory=list)
    
    # Grants and immunities
    action_grants: list[ActionGrant] = Field(default_factory=list)
    movement_grants: list[MovementGrant] = Field(default_factory=list)
    status_grants: list[StatusGrant] = Field(default_factory=list)
    immunities: list[Immunity] = Field(default_factory=list)
    resistances: list[Resistance] = Field(default_factory=list)
    
    # For complex effects not yet fully modeled
    # This should be mechanical shorthand, not flavor text
    special: str | None = Field(
        default=None, 
        description="Brief mechanical note for effects not yet modeled (e.g., 'grapple_pull_5')"
    )
    
    model_config = {"frozen": True}
    
    def is_empty(self) -> bool:
        """Check if this effect has no components."""
        return (
            not self.stat_mods
            and not self.accuracy_mods
            and not self.damage_mods
            and not self.range_mods
            and not self.action_grants
            and not self.movement_grants
            and not self.status_grants
            and not self.immunities
            and not self.resistances
            and not self.special
        )


# Convenience constructors for common patterns

def stat_bonus(stat: StatType, value: int) -> MechanicalEffect:
    """Create a simple stat bonus effect."""
    return MechanicalEffect(stat_mods=[StatModifier(stat=stat, value=value)])


def damage_bonus(
    flat: int = 0,
    dice: DiceExpression | str | None = None,
    condition: ConditionType | None = None,
) -> MechanicalEffect:
    """Create a damage bonus effect."""
    bonus_dice = DiceExpression.parse(dice) if isinstance(dice, str) else dice
    return MechanicalEffect(damage_mods=[DamageModifier(flat=flat, dice=bonus_dice, condition=condition)])


def immunity_to(target: str, condition: str | None = None) -> MechanicalEffect:
    """Create an immunity effect."""
    return MechanicalEffect(immunities=[Immunity(target=target, condition=condition)])
