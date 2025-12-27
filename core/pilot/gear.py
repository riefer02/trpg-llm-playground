"""Pilot gear models for Lancer TTRPG."""

from typing import Literal
from pydantic import BaseModel, Field

from core.shared.effects import MechanicalEffect


PilotGearCategory = Literal["clothing", "armor", "weapon", "gear"]


class PilotGearItemDefinition(BaseModel):
    """Definition for a pilot gear item."""

    id: str = Field(..., description="Unique gear identifier")
    name: str = Field(..., description="Display name")
    category: PilotGearCategory
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)

    model_config = {"frozen": True}


class PilotGearRules(BaseModel):
    """Loadout limits for pilot gear."""

    clothing_required: bool = True
    armor_optional: bool = True
    max_weapons: int = 2
    max_gear: int = 3

    model_config = {"frozen": True}


DEFAULT_PILOT_GEAR_RULES = PilotGearRules()


class PilotLoadout(BaseModel):
    """Pilot gear selection for a mission."""

    clothing: str | None = Field(default=None, description="Clothing item ID")
    armor: str | None = Field(default=None, description="Armor item ID")
    weapons: list[str] = Field(default_factory=list, max_length=2, description="Weapon item IDs")
    gear: list[str] = Field(default_factory=list, max_length=3, description="Other gear item IDs")

    model_config = {"frozen": True}

    def total_items(self) -> int:
        """Total number of selected items."""
        count = len(self.weapons) + len(self.gear)
        if self.clothing:
            count += 1
        if self.armor:
            count += 1
        return count
