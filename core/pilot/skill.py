"""Pilot skill types and models for Lancer TTRPG.

Skills in Lancer:
- Pilots have 4 skills, each with a trigger list
- Skill ranks range from +0 to +6
- Triggers describe situations where the skill applies
"""

from typing import Literal
from pydantic import BaseModel, Field

# The four pilot skills in Lancer
SkillType = Literal["hull", "agility", "systems", "engineering"]

# All skills with their abbreviations
SKILLS: dict[SkillType, str] = {
    "hull": "HULL",
    "agility": "AGI",
    "systems": "SYS",
    "engineering": "ENG",
}

# Skill triggers - situations where each skill applies
SKILL_TRIGGERS: dict[SkillType, list[str]] = {
    "hull": [
        "Assault",
        "Threaten", 
        "Apply Fists to Faces",
        "Survive",
        "Take Control",
    ],
    "agility": [
        "Act Unseen or Unheard",
        "Get Somewhere Quickly",
        "Perform a Feat of Dexterity",
        "Stay Cool Under Fire",
        "Take Someone Out",
    ],
    "systems": [
        "Hack or Fix",
        "Invent or Create",
        "Read a Situation",
        "Spot",
        "Investigate",
    ],
    "engineering": [
        "Blow Something Up",
        "Charm",
        "Get a Hold of Something",
        "Lead or Inspire",
        "Pull Rank",
        "Word on the Street",
    ],
}


class Skill(BaseModel):
    """
    A pilot skill with a rank.
    
    In Lancer, pilots have 4 mech skills (HULL, AGI, SYS, ENG)
    that determine their mech's base stats, and also serve as
    the basis for pilot skill checks using triggers.
    """
    
    skill_type: SkillType
    rank: int = Field(default=0, ge=0, le=6)
    
    model_config = {"frozen": True}
    
    @property
    def abbreviation(self) -> str:
        """Get the abbreviated form (HULL, AGI, SYS, ENG)."""
        return SKILLS[self.skill_type]
    
    @property
    def triggers(self) -> list[str]:
        """Get the triggers associated with this skill."""
        return SKILL_TRIGGERS[self.skill_type]
    
    def __str__(self) -> str:
        return f"{self.abbreviation} +{self.rank}"


class SkillSet(BaseModel):
    """
    A complete set of pilot skills.
    
    At LL0, pilots have +2 to distribute among their 4 skills.
    They gain +2 more at each level up.
    """
    
    hull: int = Field(default=0, ge=0, le=6)
    agility: int = Field(default=0, ge=0, le=6)
    systems: int = Field(default=0, ge=0, le=6)
    engineering: int = Field(default=0, ge=0, le=6)
    
    model_config = {"frozen": True}
    
    def total_points(self) -> int:
        """Total skill points allocated."""
        return self.hull + self.agility + self.systems + self.engineering
    
    def get(self, skill_type: SkillType) -> int:
        """Get rank for a specific skill type."""
        return getattr(self, skill_type)
    
    def as_dict(self) -> dict[SkillType, int]:
        """Return skills as a dictionary."""
        return {
            "hull": self.hull,
            "agility": self.agility,
            "systems": self.systems,
            "engineering": self.engineering,
        }

