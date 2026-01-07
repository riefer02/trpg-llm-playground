"""Roll mechanics models for Lancer TTRPG."""

from typing import Literal
from pydantic import Field, computed_field
from core.shared.models import FrozenModel


RollType = Literal["skill_check", "attack", "save"]
FlatBonusSource = Literal["trigger", "mech_skill", "grit"]


class AccuracyDifficulty(FrozenModel):
    """Accuracy and difficulty dice pool for a roll."""

    accuracy: int = Field(default=0, ge=0)
    difficulty: int = Field(default=0, ge=0)

    @computed_field
    @property
    def net(self) -> int:
        """Net accuracy (positive) or difficulty (negative)."""
        return self.accuracy - self.difficulty

    @computed_field
    @property
    def dice_count(self) -> int:
        """Number of d6 rolled after canceling."""
        return abs(self.net)

    @computed_field
    @property
    def direction(self) -> Literal["accuracy", "difficulty", "none"]:
        if self.net > 0:
            return "accuracy"
        if self.net < 0:
            return "difficulty"
        return "none"


class FlatBonus(FrozenModel):
    """Flat bonus applied to a roll (one source at a time)."""

    source: FlatBonusSource
    value: int = Field(default=0, ge=0, le=6)


class RollModifiers(FrozenModel):
    """Combined modifiers for a roll."""

    accuracy_difficulty: AccuracyDifficulty = Field(default_factory=AccuracyDifficulty)
    flat_bonus: FlatBonus | None = None


class DifficultyModifier(FrozenModel):
    """Difficulty modifier for a roll (typically +1, extendable for special cases)."""

    value: int = Field(
        default=1,
        ge=1,
        description="Difficulty value (+1 standard, higher for extreme cases)",
    )
    reason: str = Field(default="", description="Why the check is difficult")


class SkillCheck(FrozenModel):
    """A narrative skill check (target 10 by default)."""

    roll_type: Literal["skill_check"] = "skill_check"
    target: int = Field(default=10, ge=0)
    modifiers: RollModifiers = Field(default_factory=RollModifiers)
    is_difficult: bool = Field(
        default=False, description="Adds +1 difficulty per PR2 rules"
    )


class AttackRoll(FrozenModel):
    """An attack roll against a defense value."""

    roll_type: Literal["attack"] = "attack"
    target: int = Field(..., ge=0, description="Target defense value")
    modifiers: RollModifiers = Field(default_factory=RollModifiers)


class SaveRoll(FrozenModel):
    """A save roll against an attacker's save target."""

    roll_type: Literal["save"] = "save"
    target: int = Field(..., ge=0, description="Attacker save target")
    modifiers: RollModifiers = Field(default_factory=RollModifiers)


class ContestedCheck(FrozenModel):
    """Two opposed skill checks; ties go to the attacker."""

    attacker: SkillCheck
    defender: SkillCheck
    tie_breaker: Literal["attacker"] = "attacker"


class AttackResolutionResult(FrozenModel):
    """Complete result of attack resolution.

    Provides detailed breakdown of hit detection for attack actions.
    """

    roll: int = Field(..., ge=1, le=20, description="1d20 roll result")
    attack_bonus: int = Field(
        default=0, description="Flat bonus added to roll (grit/skill)"
    )
    accuracy_dice_rolls: list[int] = Field(
        default_factory=list, description="Accuracy d6 rolls (keep highest)"
    )
    difficulty_dice_rolls: list[int] = Field(
        default_factory=list, description="Difficulty d6 rolls (keep lowest)"
    )
    net_accuracy: int = Field(
        default=0, description="Accuracy dice minus difficulty dice after cancellation"
    )
    total_accuracy: int = Field(
        ..., description="Total accuracy (roll + attack_bonus + net_accuracy)"
    )
    target_defense: int = Field(
        ..., ge=0, description="Target defense value (evasion/E-defense)"
    )
    hit: bool = Field(..., description="Whether attack hit the target")
    is_critical: bool = Field(
        default=False, description="Whether attack was a critical (natural 20)"
    )
    miss_by: int = Field(
        default=0, ge=0, description="How much the attack missed by (0 if hit)"
    )


def _roll_d6(count: int, forced_rolls: list[int] | None = None) -> list[int]:
    """Roll d6 dice, using forced rolls if available.

    Args:
        count: Number of d6 to roll
        forced_rolls: Optional list of forced roll values for deterministic testing

    Returns:
        List of roll results
    """
    if count <= 0:
        return []
    if forced_rolls:
        return list(forced_rolls[:count])
    from core.shared.dice import DiceExpression

    return DiceExpression.parse(f"{count}d6").roll()


def resolve_attack(
    attack_bonus: int,
    target_defense: int,
    accuracy_bonus: int = 0,
    difficulty_bonus: int = 0,
    forced_roll: int | None = None,
    forced_accuracy_rolls: list[int] | None = None,
    forced_difficulty_rolls: list[int] | None = None,
) -> AttackResolutionResult:
    """Resolve an attack roll per Lancer rules.

    Hit detection (PR2):
    - Roll 1d20 + attack_bonus
    - Roll accuracy_bonus d6 (keep highest) and difficulty_bonus d6 (keep lowest)
    - Accuracy and difficulty dice cancel out (1 acc + 1 diff = 0)
    - Net accuracy = highest remaining acc - lowest remaining diff
    - Hit when: total_accuracy > target_defense
    - OR (total_accuracy == target_defense AND 1d20 roll >= 10)
    - Critical on natural 20

    Args:
        attack_bonus: Flat bonus added to the 1d20 roll (e.g., grit, skill bonus)
        target_defense: Defense value to beat (evasion or e-defense)
        accuracy_bonus: Number of accuracy dice to roll (each adds 1d6, keep highest)
        difficulty_bonus: Number of difficulty dice to roll (each adds 1d6, keep lowest)
        forced_roll: Optional forced 1d20 roll value for deterministic testing
        forced_accuracy_rolls: Optional forced accuracy die rolls for testing
        forced_difficulty_rolls: Optional forced difficulty die rolls for testing

    Returns:
        AttackResolutionResult with full breakdown of the attack
    """
    roll = forced_roll if forced_roll is not None else _roll_d6(1)[0]

    accuracy_dice = _roll_d6(accuracy_bonus, forced_accuracy_rolls)
    difficulty_dice = _roll_d6(difficulty_bonus, forced_difficulty_rolls)

    cancel_count = min(len(accuracy_dice), len(difficulty_dice))
    remaining_acc = (
        accuracy_dice[cancel_count:] if cancel_count < len(accuracy_dice) else []
    )
    remaining_diff = (
        difficulty_dice[cancel_count:] if cancel_count < len(difficulty_dice) else []
    )

    accuracy_kept = max(remaining_acc) if remaining_acc else 0
    difficulty_kept = min(remaining_diff) if remaining_diff else 0

    net_accuracy = accuracy_kept - difficulty_kept

    total_accuracy = roll + attack_bonus + net_accuracy

    is_critical = roll == 20

    if is_critical:
        hit = True
    elif roll >= 10:
        hit = total_accuracy > target_defense or (
            total_accuracy == target_defense and roll >= 10
        )
    else:
        hit = total_accuracy > target_defense

    miss_by = max(0, target_defense - total_accuracy) if not hit else 0

    return AttackResolutionResult(
        roll=roll,
        attack_bonus=attack_bonus,
        accuracy_dice_rolls=accuracy_dice,
        difficulty_dice_rolls=difficulty_dice,
        net_accuracy=net_accuracy,
        total_accuracy=total_accuracy,
        target_defense=target_defense,
        hit=hit,
        is_critical=is_critical,
        miss_by=miss_by,
    )
