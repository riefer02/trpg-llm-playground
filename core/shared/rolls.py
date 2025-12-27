"""Roll mechanics models for Lancer TTRPG."""

from typing import Literal
from pydantic import BaseModel, Field, computed_field


RollType = Literal["skill_check", "attack", "save"]
FlatBonusSource = Literal["trigger", "mech_skill", "grit"]


class AccuracyDifficulty(BaseModel):
    """Accuracy and difficulty dice pool for a roll."""

    accuracy: int = Field(default=0, ge=0)
    difficulty: int = Field(default=0, ge=0)

    model_config = {"frozen": True}

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


class FlatBonus(BaseModel):
    """Flat bonus applied to a roll (one source at a time)."""

    source: FlatBonusSource
    value: int = Field(default=0, ge=0, le=6)

    model_config = {"frozen": True}


class RollModifiers(BaseModel):
    """Combined modifiers for a roll."""

    accuracy_difficulty: AccuracyDifficulty = Field(default_factory=AccuracyDifficulty)
    flat_bonus: FlatBonus | None = None

    model_config = {"frozen": True}


class SkillCheck(BaseModel):
    """A narrative skill check (target 10 by default)."""

    roll_type: Literal["skill_check"] = "skill_check"
    target: int = Field(default=10, ge=0)
    modifiers: RollModifiers = Field(default_factory=RollModifiers)

    model_config = {"frozen": True}


class AttackRoll(BaseModel):
    """An attack roll against a defense value."""

    roll_type: Literal["attack"] = "attack"
    target: int = Field(..., ge=0, description="Target defense value")
    modifiers: RollModifiers = Field(default_factory=RollModifiers)

    model_config = {"frozen": True}


class SaveRoll(BaseModel):
    """A save roll against an attacker's save target."""

    roll_type: Literal["save"] = "save"
    target: int = Field(..., ge=0, description="Attacker save target")
    modifiers: RollModifiers = Field(default_factory=RollModifiers)

    model_config = {"frozen": True}


class ContestedCheck(BaseModel):
    """Two opposed skill checks; ties go to the attacker."""

    attacker: SkillCheck
    defender: SkillCheck
    tie_breaker: Literal["attacker"] = "attacker"

    model_config = {"frozen": True}
