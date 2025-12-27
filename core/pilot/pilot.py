"""Main Pilot model for Lancer TTRPG.

The Pilot is the central entity representing a player character.
Pilots have skills, backgrounds, talents, licenses, and stats
that define their capabilities both in and out of the mech.
"""

from pydantic import BaseModel, Field, computed_field
from uuid import uuid4

from core.pilot.skill import SkillSet
from core.pilot.background import Background
from core.pilot.talent import Talent
from core.pilot.license import License
from core.pilot.core_bonus import CoreBonus


class Pilot(BaseModel):
    """
    A Lancer pilot character.
    
    Pilots progress from License Level 0 (LL0) to LL12.
    At each level, they gain:
    - +2 skill points to distribute
    - +1 license level (except LL0)
    - +1 talent rank (except LL0)
    
    Every 3 license levels with a manufacturer unlocks a core bonus.
    """
    
    # Identity
    id: str = Field(default_factory=lambda: str(uuid4()))
    callsign: str = Field(..., min_length=1, description="The pilot's callsign")
    name: str = Field(default="", description="The pilot's real name (optional)")
    
    # Background & Flavor
    background: Background | None = Field(default=None)
    notes: str = Field(default="", description="Player notes about the character")
    
    # Progression
    level: int = Field(default=0, ge=0, le=12, description="License Level (LL0-LL12)")
    
    # Skills (mech stats + pilot triggers)
    skills: SkillSet = Field(default_factory=SkillSet)
    
    # Abilities
    talents: list[Talent] = Field(default_factory=list)
    licenses: list[License] = Field(default_factory=list)
    core_bonuses: list[CoreBonus] = Field(default_factory=list)
    
    # Stats (derived, but can be overridden)
    hp_bonus: int = Field(default=0, description="Bonus HP from talents/gear")
    armor_bonus: int = Field(default=0, description="Bonus armor from gear")
    
    model_config = {"validate_assignment": True}
    
    @computed_field
    @property
    def grit(self) -> int:
        """
        Grit is half of level (rounded up), minimum 0.
        Used for various bonuses and HP calculation.
        """
        return (self.level + 1) // 2
    
    @computed_field
    @property
    def hp(self) -> int:
        """
        Pilot HP = 6 + Grit + any bonuses.
        This is for when the pilot is out of the mech.
        """
        return 6 + self.grit + self.hp_bonus
    
    @computed_field
    @property
    def armor(self) -> int:
        """Pilot armor (usually 0 unless wearing hardsuit)."""
        return self.armor_bonus
    
    @computed_field
    @property
    def evasion(self) -> int:
        """Pilot evasion = 10 by default."""
        return 10
    
    @computed_field
    @property
    def e_defense(self) -> int:
        """Pilot e-defense = 10 by default."""
        return 10
    
    @computed_field
    @property
    def speed(self) -> int:
        """Pilot speed = 4 by default."""
        return 4
    
    # Progression Helpers
    
    def total_talent_ranks(self) -> int:
        """Total talent ranks the pilot has."""
        return sum(t.rank for t in self.talents)
    
    def total_license_levels(self) -> int:
        """Total license levels the pilot has."""
        return sum(lic.rank for lic in self.licenses)
    
    def max_talent_ranks(self) -> int:
        """Maximum talent ranks for this level."""
        return self.level  # 1 talent rank per level
    
    def max_license_levels(self) -> int:
        """Maximum license levels for this level."""
        return self.level  # 1 license level per level
    
    def max_skill_points(self) -> int:
        """Maximum skill points for this level."""
        return 2 + (self.level * 2)  # 2 base + 2 per level
    
    def max_core_bonuses(self) -> int:
        """
        Maximum core bonuses for this level.
        Earned every 3 license levels with any manufacturer.
        """
        return self.level // 3
    
    def get_license(self, license_id: str) -> License | None:
        """Get a specific license by ID."""
        for lic in self.licenses:
            if lic.license_id == license_id:
                return lic
        return None
    
    def get_talent(self, talent_id: str) -> Talent | None:
        """Get a specific talent by ID."""
        for t in self.talents:
            if t.talent_id == talent_id:
                return t
        return None
    
    def has_core_bonus(self, core_bonus_id: str) -> bool:
        """Check if pilot has a specific core bonus."""
        return any(cb.core_bonus_id == core_bonus_id for cb in self.core_bonuses)


# Factory functions for creating pilots

def create_ll0_pilot(
    callsign: str,
    name: str = "",
    background: Background | None = None,
    skills: SkillSet | None = None,
) -> Pilot:
    """
    Create a new License Level 0 pilot.
    
    LL0 pilots start with:
    - 2 skill points to distribute
    - No talents
    - No licenses
    - No core bonuses
    """
    return Pilot(
        callsign=callsign,
        name=name,
        background=background,
        level=0,
        skills=skills or SkillSet(),
    )

