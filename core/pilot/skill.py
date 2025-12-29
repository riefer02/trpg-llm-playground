"""Mech skills and pilot triggers for Lancer TTRPG.

Skills in Lancer:
- Pilots have 4 mech skills (HULL, AGI, SYS, ENG)
- Skill ranks range from +0 to +6
- Pilot triggers provide flat bonuses on pilot skill checks
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

# The four pilot skills in Lancer
SkillType = Literal["hull", "agility", "systems", "engineering"]

# All skills with their abbreviations
SKILLS: dict[SkillType, str] = {
    "hull": "HULL",
    "agility": "AGI",
    "systems": "SYS",
    "engineering": "ENG",
}

class TriggerDefinition(FrozenModel):
    """A trigger definition usable for pilot skill checks."""

    id: str = Field(..., description="Unique trigger identifier")
    name: str = Field(..., description="Display name")



class PilotTrigger(FrozenModel):
    """A pilot's trigger rank (+2 to +6)."""

    trigger_id: str = Field(..., description="ID of the trigger definition")
    rank: int = Field(
        default=2,
        ge=2,
        le=6,
        multiple_of=2,
        description="Trigger bonus (+2 to +6, in +2 increments)",
    )



# Trigger definitions (no mech skill linkage in the core rules)
TRIGGER_DEFINITIONS: list[TriggerDefinition] = [
    TriggerDefinition(id="apply_fists_to_faces", name="Apply Fists to Faces"),
    TriggerDefinition(id="assault", name="Assault"),
    TriggerDefinition(id="blow_something_up", name="Blow Something Up"),
    TriggerDefinition(id="threaten", name="Threaten"),
    TriggerDefinition(id="take_control", name="Take Control"),
    TriggerDefinition(id="survive", name="Survive"),
    TriggerDefinition(id="stay_cool", name="Stay Cool"),
    TriggerDefinition(id="take_someone_out", name="Take Someone Out"),
    TriggerDefinition(id="show_off", name="Show Off"),
    TriggerDefinition(id="get_somewhere_quickly", name="Get Somewhere Quickly"),
    TriggerDefinition(id="act_unseen_or_unheard", name="Act Unseen or Unheard"),
    TriggerDefinition(id="hack_or_fix", name="Hack or Fix"),
    TriggerDefinition(id="patch", name="Patch"),
    TriggerDefinition(id="invent_or_create", name="Invent or Create"),
    TriggerDefinition(id="read_a_situation", name="Read A Situation"),
    TriggerDefinition(id="spot", name="Spot"),
    TriggerDefinition(id="investigate", name="Investigate"),
    TriggerDefinition(id="charm", name="Charm"),
    TriggerDefinition(id="pull_rank", name="Pull Rank"),
    TriggerDefinition(id="word_on_the_streets", name="Word On the Streets"),
    TriggerDefinition(id="get_a_hold_of_something", name="Get a Hold of Something"),
    TriggerDefinition(id="lead_or_inspire", name="Lead or Inspire"),
]


def get_trigger_definition(trigger_id: str) -> TriggerDefinition | None:
    """Look up a trigger definition by ID."""
    for trigger in TRIGGER_DEFINITIONS:
        if trigger.id == trigger_id:
            return trigger
    return None


class Skill(FrozenModel):
    """
    A pilot skill with a rank.
    
    In Lancer, pilots have 4 mech skills (HULL, AGI, SYS, ENG)
    that determine their mech's base stats. Pilot skill checks
    instead use triggers for flat bonuses.
    """
    
    skill_type: SkillType
    rank: int = Field(default=0, ge=0, le=6)
    
    
    @property
    def abbreviation(self) -> str:
        """Get the abbreviated form (HULL, AGI, SYS, ENG)."""
        return SKILLS[self.skill_type]
    
    def __str__(self) -> str:
        return f"{self.abbreviation} +{self.rank}"


class SkillSet(FrozenModel):
    """
    A complete set of pilot skills.
    
    At LL0, pilots have +2 to distribute among their 4 skills.
    They gain +1 more at each level up.
    """
    
    hull: int = Field(default=0, ge=0, le=6)
    agility: int = Field(default=0, ge=0, le=6)
    systems: int = Field(default=0, ge=0, le=6)
    engineering: int = Field(default=0, ge=0, le=6)
    
    
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
