"""Main Pilot model for Lancer TTRPG.

The Pilot is the central entity representing a player character.
Pilots have skills, backgrounds, talents, licenses, and stats
that define their capabilities both in and out of the mech.
"""

from pydantic import BaseModel, Field, computed_field, model_validator
from uuid import uuid4
from typing import TYPE_CHECKING, Literal

from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.background import Background
from core.pilot.gear import (
    PilotLoadout,
    get_pilot_gear_stat_mods,
    validate_pilot_loadout,
)
from core.pilot.combat import DEFAULT_PILOT_COMBAT_STATS
from core.shared.enums import SizeClass
from core.pilot.talent import Talent
from core.pilot.license import License
from core.pilot.core_bonus import CoreBonus
from core.pilot.progression import LEVEL_CAP, get_level_progression
from core.pilot.validation import ProgressionValidation, validate_pilot_progression
from core.shared.id_helpers import (
    PilotIdField,
    LicenseIdField,
    TalentIdField,
    CoreBonusIdField,
)

if TYPE_CHECKING:
    from core.pilot.clone_state import CloneState, Quirk, QuirkSource
    from core.pilot.down_and_out import DownAndOutResult


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
    id: PilotIdField = Field(default="")
    callsign: str = Field(..., min_length=1, description="The pilot's callsign")
    name: str = Field(default="", description="The pilot's real name (optional)")

    # Background & Flavor
    background: Background | None = Field(default=None)
    notes: str = Field(default="", description="Player notes about the character")
    pilot_gear: PilotLoadout | None = Field(
        default=None, description="Pilot gear loadout"
    )

    # Progression
    level: int = Field(
        default=0, ge=0, le=LEVEL_CAP, description="License Level (LL0-LL12)"
    )

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

    # Cloning and Death State (Priority 55)
    clone_state: "CloneState | None" = Field(
        default=None,
        description="Cloning state for campaign tracking (None if cloning not in use)",
    )
    is_dead: bool = Field(default=False, description="Whether pilot is currently dead")
    has_down_and_out_trauma: bool = Field(
        default=False,
        description="Whether pilot has a quirk from surviving Down and Out",
    )

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def _validate_pilot_gear(self) -> "Pilot":
        if self.pilot_gear is None:
            return self
        validation = validate_pilot_loadout(self.pilot_gear)
        if not validation.valid:
            messages = "; ".join(
                issue.message
                for issue in validation.issues
                if issue.severity == "error"
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
        return (
            DEFAULT_PILOT_COMBAT_STATS.hp
            + self.grit
            + self.hp_bonus
            + mods.get("hp", 0)
        )

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

    def get_license(self, license_id: LicenseIdField) -> License | None:
        """Get a specific license by ID."""
        for lic in self.licenses:
            if lic.license_id == license_id:
                return lic
        return None

    def get_talent(self, talent_id: TalentIdField) -> Talent | None:
        """Get a specific talent by ID."""
        for t in self.talents:
            if t.talent_id == talent_id:
                return t
        return None

    def has_core_bonus(self, core_bonus_id: CoreBonusIdField) -> bool:
        """Check if pilot has a specific core bonus."""
        return any(cb.core_bonus_id == core_bonus_id for cb in self.core_bonuses)

    def validate_progression(self) -> ProgressionValidation:
        """Validate this pilot against the progression table."""
        return validate_pilot_progression(self)

        from core.pilot.down_and_out import resolve_down_and_out

        result = resolve_down_and_out(down_and_out_input)

        return {
            "damage_dealt": damage,
            "hp_remaining": result.hp_after,
            "is_down_and_out": result.outcome == "down_and_out",
            "is_dead": result.outcome == "died",
            "requires_check": True,
            "roll_result": result.roll_result,
            "outcome": result.outcome,
            "narrative_notes": result.narrative_notes,
        }

    def roll_down_and_out(self) -> "DownAndOutResult":
        """Roll a Down and Out check for this pilot.

        Returns:
            DownAndOutResult from the check
        """
        from core.pilot.down_and_out import resolve_down_and_out, DownAndOutInput

        input_data = DownAndOutInput(
            pilot_id=self.id, current_hp=0, max_hp=self.hp, base_evasion=self.evasion
        )
        return resolve_down_and_out(input_data)

    def apply_clone_result(
        self, new_hp: int, new_evasion: int, quirk: "Quirk"
    ) -> "Pilot":
        """Apply clone result to this pilot (for the cloned version).

        Creates a new Pilot with clone state set.

        Args:
            new_hp: HP after cloning (session start value)
            new_evasion: Evasion after cloning (session start value)
            quirk: Quirk assigned to this clone

        Returns:
            New Pilot with clone state updated
        """
        from core.pilot.clone_state import CloneState, CloneStatus, QuirkSource

        new_clone_state = CloneState(
            status=CloneStatus(times_cloned=1, is_dead=False),
            assigned_quirk=quirk,
            quirk_source="clone",
            session_start_hp=new_hp,
            session_start_evasion=new_evasion,
            clone_applicable=True,
        )

        return Pilot(
            id=self.id,
            callsign=self.callsign,
            name=self.name,
            background=self.background,
            notes=self.notes,
            pilot_gear=self.pilot_gear,
            level=self.level,
            skills=self.skills,
            triggers=self.triggers,
            talents=self.talents,
            licenses=self.licenses,
            core_bonuses=self.core_bonuses,
            hp_bonus=self.hp_bonus,
            armor_bonus=self.armor_bonus,
            clone_state=new_clone_state,
            is_dead=False,
            has_down_and_out_trauma=False,
        )

    def add_quirk(self, quirk: "Quirk", source: "QuirkSource") -> "Pilot":
        """Add a quirk to this pilot.

        Args:
            quirk: Quirk to add
            source: Source of the quirk ("clone" or "down_and_out_trauma")

        Returns:
            New Pilot with quirk added
        """
        from core.pilot.clone_state import CloneState, QuirkSource as QS

        new_quirk_source: QS = "clone" if source == "clone" else "down_and_out_trauma"

        if self.clone_state is None:
            new_clone_state = CloneState(
                status=CloneStatus(times_cloned=0, is_dead=self.is_dead),
                assigned_quirk=quirk,
                quirk_source=new_quirk_source,
                session_start_hp=self.hp,
                session_start_evasion=self.evasion,
                clone_applicable=True,
            )
        else:
            new_clone_state = self.clone_state.with_quirk(quirk, new_quirk_source)

        is_trauma = source == "down_and_out_trauma"

        return Pilot(
            id=self.id,
            callsign=self.callsign,
            name=self.name,
            background=self.background,
            notes=self.notes,
            pilot_gear=self.pilot_gear,
            level=self.level,
            skills=self.skills,
            triggers=self.triggers,
            talents=self.talents,
            licenses=self.licenses,
            core_bonuses=self.core_bonuses,
            hp_bonus=self.hp_bonus,
            armor_bonus=self.armor_bonus,
            clone_state=new_clone_state,
            is_dead=self.is_dead,
            has_down_and_out_trauma=is_trauma or self.has_down_and_out_trauma,
        )

    def record_session_start(self) -> "Pilot":
        """Record session start state for potential clone rewind.

        Returns:
            New Pilot with session snapshot recorded
        """
        from core.pilot.clone_state import CloneState

        if self.clone_state is None:
            new_clone_state = CloneState(
                status=CloneStatus(times_cloned=0, is_dead=False),
                session_start_hp=self.hp,
                session_start_evasion=self.evasion,
                clone_applicable=True,
            )
        else:
            new_clone_state = self.clone_state.with_session_snapshot(
                self.hp, self.evasion
            )

        return Pilot(
            id=self.id,
            callsign=self.callsign,
            name=self.name,
            background=self.background,
            notes=self.notes,
            pilot_gear=self.pilot_gear,
            level=self.level,
            skills=self.skills,
            triggers=self.triggers,
            talents=self.talents,
            licenses=self.licenses,
            core_bonuses=self.core_bonuses,
            hp_bonus=self.hp_bonus,
            armor_bonus=self.armor_bonus,
            clone_state=new_clone_state,
            is_dead=self.is_dead,
            has_down_and_out_trauma=self.has_down_and_out_trauma,
        )


# Factory functions for creating pilots


def create_ll0_pilot(
    callsign: str,
    name: str = "",
    background: Background | None = None,
    skills: SkillSet | None = None,
    triggers: list[PilotTrigger] | None = None,
    talents: list[Talent] | None = None,
    pilot_gear: PilotLoadout | None = None,
    id: PilotIdField | None = None,
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
    if len(resolved_triggers) != 4 or any(
        trigger.rank != 2 for trigger in resolved_triggers
    ):
        raise ValueError("LL0 pilots must have exactly 4 triggers at +2 each.")
    if sum(talent.rank for talent in resolved_talents) != 3 or any(
        talent.rank != 1 for talent in resolved_talents
    ):
        raise ValueError("LL0 pilots must have exactly three rank I talents.")

    return Pilot(
        id=id if id is not None else "",
        callsign=callsign,
        name=name,
        background=background,
        level=0,
        skills=resolved_skills,
        triggers=resolved_triggers,
        talents=resolved_talents,
        pilot_gear=pilot_gear,
    )
