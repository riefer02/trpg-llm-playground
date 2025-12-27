"""Pilot domain models for Lancer TTRPG."""

from core.pilot.skill import Skill, SkillSet, SkillType, SKILLS, SKILL_TRIGGERS
from core.pilot.background import Background, STANDARD_BACKGROUNDS, get_background
from core.pilot.talent import (
    Talent,
    TalentRank,
    TalentDefinition,
    EXAMPLE_TALENTS,
    get_talent_definition,
)
from core.pilot.license import (
    License,
    LicenseDefinition,
    Manufacturer,
    MANUFACTURER_NAMES,
    EXAMPLE_LICENSES,
    get_license_definition,
    get_licenses_by_manufacturer,
)
from core.pilot.core_bonus import (
    CoreBonus,
    CoreBonusDefinition,
    ALL_CORE_BONUSES,
    get_core_bonus_definition,
    get_core_bonuses_by_manufacturer,
)
from core.pilot.pilot import Pilot, create_ll0_pilot

__all__ = [
    # Main model
    "Pilot",
    "create_ll0_pilot",
    # Skills
    "Skill",
    "SkillSet",
    "SkillType",
    "SKILLS",
    "SKILL_TRIGGERS",
    # Backgrounds
    "Background",
    "STANDARD_BACKGROUNDS",
    "get_background",
    # Talents
    "Talent",
    "TalentRank",
    "TalentDefinition",
    "EXAMPLE_TALENTS",
    "get_talent_definition",
    # Licenses
    "License",
    "LicenseDefinition",
    "Manufacturer",
    "MANUFACTURER_NAMES",
    "EXAMPLE_LICENSES",
    "get_license_definition",
    "get_licenses_by_manufacturer",
    # Core Bonuses
    "CoreBonus",
    "CoreBonusDefinition",
    "ALL_CORE_BONUSES",
    "get_core_bonus_definition",
    "get_core_bonuses_by_manufacturer",
]

