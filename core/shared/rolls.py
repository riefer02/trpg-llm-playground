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
