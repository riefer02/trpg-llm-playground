"""Main Pilot model for Lancer TTRPG.

The Pilot is the central entity representing a player character.
Pilots have skills, backgrounds, talents, licenses, and stats
that define their capabilities both in and out of the mech.
"""

from pydantic import BaseModel, Field, computed_field, model_validator
from uuid import uuid4

from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.background import Background
from core.pilot.gear import PilotLoadout, get_pilot_gear_stat_mods, validate_pilot_loadout
from core.pilot.combat import DEFAULT_PILOT_COMBAT_STATS
from core.shared.enums import SizeClass
from core.pilot.talent import Talent
from core.pilot.license import License
from core.pilot.core_bonus import CoreBonus
from core.pilot.progression import LEVEL_CAP, get_level_progression
from core.pilot.validation import ProgressionValidation, validate_pilot_progression


class Pilot(BaseModel):
    """
    A Lancer pilot character.
    
    Pilots progress from License Level 0 (LL0) to LL12.
    At each level, they gain:
    - +1 mech skill point
    - +1 license point
    - +1 talent point
    - +2 trigger points
    
    Every 3 license levels with a manufacturer unlocks a core bonus.
    """
    
    # Identity
    id: str = Field(default_factory=lambda: str(uuid4()))
    callsign: str = Field(..., min_length=1, description="The pilot's callsign")
    name: str = Field(default="", description="The pilot's real name (optional)")
    
    # Background & Flavor
    background: Background | None = Field(default=None)
    notes: str = Field(default="", description="Player notes about the character")
    pilot_gear: PilotLoadout | None = Field(default=None, description="Pilot gear loadout")
    
    # Progression
    level: int = Field(default=0, ge=0, le=LEVEL_CAP, description="License Level (LL0-LL12)")
    
    # Skills (mech stats + pilot triggers)
    skills: SkillSet = Field(default_factory=SkillSet)
    
    # Narrative triggers (pilot skill checks)
    triggers: list[PilotTrigger] = Field(default_factory=list)

    # Abilities
    talents: list[Talent] = Field(default_factory=list)
    licenses: list[License] = Field(default_factory=list)
    core_bonuses: list[CoreBonus] = Field(default_factory=list)
    
    # Stats (derived, but can be overridden)
    hp_bonus: int = Field(default=0, description="Bonus HP from talents/gear")
    armor_bonus: int = Field(default=0, description="Bonus armor from gear")
    
    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def _validate_pilot_gear(self) -> "Pilot":
        if self.pilot_gear is None:
            return self
        validation = validate_pilot_loadout(self.pilot_gear)
        if not validation.valid:
            messages = "; ".join(
                issue.message for issue in validation.issues if issue.severity == "error"
            )
            raise ValueError(f"Invalid pilot gear loadout: {messages}")
        return self

    def _gear_stat_mods(self) -> dict[str, int]:
        if not self.pilot_gear:
            return {}
        return get_pilot_gear_stat_mods(self.pilot_gear)
    
    @computed_field
    @property
    def grit(self) -> int:
        """
        Grit is half of level (rounded up), minimum 0.
        Used for various bonuses and HP calculation.
        """
        progression = get_level_progression(self.level)
        return progression.grit
    
    @computed_field
    @property
    def hp(self) -> int:
        """
        Pilot HP = 6 + Grit + any bonuses.
        This is for when the pilot is out of the mech.
        """
        mods = self._gear_stat_mods()
        return DEFAULT_PILOT_COMBAT_STATS.hp + self.grit + self.hp_bonus + mods.get("hp", 0)
    
    @computed_field
    @property
    def armor(self) -> int:
        """Pilot armor (usually 0 unless wearing hardsuit)."""
        mods = self._gear_stat_mods()
        return self.armor_bonus + mods.get("armor", 0)
    
    @computed_field
    @property
    def evasion(self) -> int:
        """Pilot evasion = 10 by default."""
        mods = self._gear_stat_mods()
        return DEFAULT_PILOT_COMBAT_STATS.evasion + mods.get("evasion", 0)
    
    @computed_field
    @property
    def e_defense(self) -> int:
        """Pilot e-defense = 10 by default."""
        mods = self._gear_stat_mods()
        return DEFAULT_PILOT_COMBAT_STATS.e_defense + mods.get("e_defense", 0)
    
    @computed_field
    @property
    def speed(self) -> int:
        """Pilot speed = 4 by default."""
        mods = self._gear_stat_mods()
        return DEFAULT_PILOT_COMBAT_STATS.speed + mods.get("speed", 0)

    @computed_field
    @property
    def size(self) -> SizeClass:
        """Pilot size category."""
        return DEFAULT_PILOT_COMBAT_STATS.size

    @computed_field
    @property
    def save_target(self) -> int:
        """Pilot save target = 10 + grit."""
        return 10 + self.grit

    @computed_field
    @property
    def attack_bonus(self) -> int:
        """Base attack bonus from grit."""
        return self.grit
    
    # Progression Helpers
    
    def total_talent_ranks(self) -> int:
        """Total talent ranks the pilot has."""
        return sum(t.rank for t in self.talents)
    
    def total_license_levels(self) -> int:
        """Total license levels the pilot has."""
        return sum(lic.rank for lic in self.licenses)

    def total_trigger_points(self) -> int:
        """Total trigger points allocated."""
        return sum(trigger.rank for trigger in self.triggers)
    
    def max_talent_ranks(self) -> int:
        """Maximum talent ranks for this level."""
        progression = get_level_progression(self.level)
        return progression.total_talent_points
    
    def max_license_levels(self) -> int:
        """Maximum license levels for this level."""
        progression = get_level_progression(self.level)
        return progression.license_points
    
    def max_skill_points(self) -> int:
        """Maximum skill points for this level."""
        progression = get_level_progression(self.level)
        return progression.total_mech_skill_points
    
    def max_core_bonuses(self) -> int:
        """
        Maximum core bonuses for this level.
        Earned every 3 license levels with any manufacturer.
        """
        progression = get_level_progression(self.level)
        return progression.core_bonuses

    def max_trigger_points(self) -> int:
        """Maximum trigger points for this level."""
        progression = get_level_progression(self.level)
        return progression.pilot_trigger_points
    
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

    def validate_progression(self) -> ProgressionValidation:
        """Validate this pilot against the progression table."""
        return validate_pilot_progression(self)


# Factory functions for creating pilots

def create_ll0_pilot(
    callsign: str,
    name: str = "",
    background: Background | None = None,
    skills: SkillSet | None = None,
    triggers: list[PilotTrigger] | None = None,
    talents: list[Talent] | None = None,
    pilot_gear: PilotLoadout | None = None,
) -> Pilot:
    """
    Create a new License Level 0 pilot.
    
    LL0 pilots start with:
    - 2 skill points to distribute
    - 4 triggers at +2 each
    - Three rank I talents
    - No licenses
    - No core bonuses
    """
    resolved_skills = skills or SkillSet()
    resolved_triggers = triggers or []
    resolved_talents = talents or []

    if resolved_skills.total_points() != 2:
        raise ValueError("LL0 pilots must allocate exactly 2 mech skill points.")
    if len(resolved_triggers) != 4 or any(trigger.rank != 2 for trigger in resolved_triggers):
        raise ValueError("LL0 pilots must have exactly 4 triggers at +2 each.")
    if sum(talent.rank for talent in resolved_talents) != 3 or any(talent.rank != 1 for talent in resolved_talents):
        raise ValueError("LL0 pilots must have exactly three rank I talents.")

    return Pilot(
        callsign=callsign,
        name=name,
        background=background,
        level=0,
        skills=resolved_skills,
        triggers=resolved_triggers,
        talents=resolved_talents,
        pilot_gear=pilot_gear,
    )
