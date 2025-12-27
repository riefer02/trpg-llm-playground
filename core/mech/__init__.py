"""Mech domain models for Lancer TTRPG."""

from core.mech.weapon import (
    WeaponSize,
    WeaponType,
    WeaponDamageType,
    MechWeaponDefinition,
)
from core.mech.mounts import MountSlotType, MountSlot, allowed_weapon_sizes
from core.mech.frame import CoreSystemDefinition, MechFrameBaseStats, MechFrameDefinition
from core.mech.rules import (
    MechPilotingRules,
    CorePowerRules,
    SystemPointRules,
    DEFAULT_MECH_PILOTING_RULES,
    DEFAULT_CORE_POWER_RULES,
    DEFAULT_SYSTEM_POINT_RULES,
)
from core.mech.build import (
    MountedWeapon,
    InstalledSystem,
    MechBuild,
    MechDerivedStats,
    compute_mech_stats,
)
from core.mech.validation import MechBuildIssue, MechBuildValidation, validate_mech_build
from core.mech.examples import (
    build_example_everest_frame,
    build_example_raleigh_frame,
    build_oda_ll0_mech_example,
    build_oda_ll3_mech_example,
    evaluate_oda_ll0_mech_example,
    evaluate_oda_ll3_mech_example,
    compute_oda_ll0_stats,
    compute_oda_ll3_stats,
)

__all__ = [
    "WeaponSize",
    "WeaponType",
    "WeaponDamageType",
    "MechWeaponDefinition",
    "MountSlotType",
    "MountSlot",
    "allowed_weapon_sizes",
    "CoreSystemDefinition",
    "MechFrameBaseStats",
    "MechFrameDefinition",
    "MechPilotingRules",
    "CorePowerRules",
    "SystemPointRules",
    "DEFAULT_MECH_PILOTING_RULES",
    "DEFAULT_CORE_POWER_RULES",
    "DEFAULT_SYSTEM_POINT_RULES",
    "MountedWeapon",
    "InstalledSystem",
    "MechBuild",
    "MechDerivedStats",
    "compute_mech_stats",
    "MechBuildIssue",
    "MechBuildValidation",
    "validate_mech_build",
    "build_example_everest_frame",
    "build_example_raleigh_frame",
    "build_oda_ll0_mech_example",
    "build_oda_ll3_mech_example",
    "evaluate_oda_ll0_mech_example",
    "evaluate_oda_ll3_mech_example",
    "compute_oda_ll0_stats",
    "compute_oda_ll3_stats",
]
