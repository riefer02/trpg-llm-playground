"""Example mech builds for validation and reference."""

from core.mech.build import MechBuild, MountedWeapon, InstalledSystem, compute_mech_stats
from core.mech.frame import CoreSystemDefinition, MechFrameBaseStats, MechFrameDefinition
from core.mech.mounts import MountSlot
from core.mech.validation import MechBuildValidation, validate_mech_build
from core.pilot.skill import SkillSet
from core.shared.effects import MechanicalEffect, StatModifier


def build_example_everest_frame() -> MechFrameDefinition:
    """Stubbed GMS Everest frame matching the LL0 example stats."""
    return MechFrameDefinition(
        id="gms_everest",
        name="GMS Everest",
        manufacturer="GMS",
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=0,
            hp=10,
            evasion=8,
            e_defense=8,
            speed=4,
            sensor_range=10,
            tech_attack=0,
            heat_cap=6,
            repair_cap=5,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="heavy"),
            MountSlot(slot_type="main"),
            MountSlot(slot_type="aux_aux"),
        ],
        system_points=6,
        core_system=CoreSystemDefinition(
            id="gms_hyperspec_fuel_injector",
            name="GMS Hyperspec Fuel Injector",
        ),
    )


def build_example_raleigh_frame() -> MechFrameDefinition:
    """Stubbed IPS-N Raleigh frame matching the LL3 example stats."""
    return MechFrameDefinition(
        id="ipsn_raleigh",
        name="IPS-N Raleigh",
        manufacturer="IPS-N",
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=1,
            hp=9,
            evasion=8,
            e_defense=8,
            speed=4,
            sensor_range=10,
            tech_attack=0,
            heat_cap=5,
            repair_cap=4,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="heavy"),
            MountSlot(slot_type="main"),
            MountSlot(slot_type="aux_aux"),
        ],
        system_points=5,
        core_system=CoreSystemDefinition(
            id="ipsn_m35_mjolnir",
            name="IPS-N M35 Mjolnir Cannon",
        ),
    )


def build_oda_ll0_mech_example() -> tuple[MechFrameDefinition, MechBuild, SkillSet, int, list[MechanicalEffect]]:
    """Build Oda's LL0 Everest mech loadout."""
    frame = build_example_everest_frame()
    build = MechBuild(
        frame_id=frame.id,
        weapons=[
            MountedWeapon(mount_index=0, weapon_id="anti_material_rifle", weapon_size="heavy"),
            MountedWeapon(mount_index=1, weapon_id="assault_rifle", weapon_size="main"),
            MountedWeapon(mount_index=2, weapon_id="tactical_knife", weapon_size="aux"),
            MountedWeapon(mount_index=2, weapon_id="tactical_knife", weapon_size="aux"),
        ],
        systems=[
            InstalledSystem(system_id="gms_hex_charges", sp_cost=2),
            InstalledSystem(system_id="gms_jump_jet_burst", sp_cost=1),
            InstalledSystem(system_id="personalizations", sp_cost=1),
            InstalledSystem(system_id="gms_shield_type_1", sp_cost=2),
        ],
    )
    skills = SkillSet(hull=2, agility=0, systems=0, engineering=0)
    grit = 0
    bonus_effects = [MechanicalEffect(stat_mods=[StatModifier(stat="hp", value=2)])]
    return frame, build, skills, grit, bonus_effects


def build_oda_ll3_mech_example() -> tuple[MechFrameDefinition, MechBuild, SkillSet, int, list[MechanicalEffect]]:
    """Build Oda's LL3 Raleigh mech loadout."""
    frame = build_example_raleigh_frame()
    build = MechBuild(
        frame_id=frame.id,
        weapons=[
            MountedWeapon(mount_index=0, weapon_id="anti_material_rifle", weapon_size="heavy"),
            MountedWeapon(mount_index=1, weapon_id="assault_rifle", weapon_size="main"),
            MountedWeapon(mount_index=2, weapon_id="hand_cannon", weapon_size="aux"),
            MountedWeapon(mount_index=2, weapon_id="hand_cannon", weapon_size="aux"),
        ],
        systems=[
            InstalledSystem(system_id="gms_hex_charges", sp_cost=2),
            InstalledSystem(system_id="gms_shield_type_1", sp_cost=2),
            InstalledSystem(system_id="ipsn_breaching_charges", sp_cost=2),
            InstalledSystem(system_id="gms_jump_jet_burst", sp_cost=1),
        ],
    )
    skills = SkillSet(hull=5, agility=0, systems=0, engineering=0)
    grit = 2
    bonus_effects = [MechanicalEffect(stat_mods=[StatModifier(stat="hp", value=5)])]
    return frame, build, skills, grit, bonus_effects


def evaluate_oda_ll0_mech_example() -> MechBuildValidation:
    """Validate the LL0 Everest build."""
    frame, build, skills, grit, _ = build_oda_ll0_mech_example()
    return validate_mech_build(frame, build, skills, grit)


def evaluate_oda_ll3_mech_example() -> MechBuildValidation:
    """Validate the LL3 Raleigh build."""
    frame, build, skills, grit, _ = build_oda_ll3_mech_example()
    return validate_mech_build(frame, build, skills, grit)


def compute_oda_ll0_stats() -> dict[str, int | str]:
    """Compute LL0 mech stats for the example."""
    frame, _, skills, grit, effects = build_oda_ll0_mech_example()
    stats = compute_mech_stats(frame, skills, grit, bonus_effects=effects)
    return stats.model_dump()


def compute_oda_ll3_stats() -> dict[str, int | str]:
    """Compute LL3 mech stats for the example."""
    frame, _, skills, grit, effects = build_oda_ll3_mech_example()
    stats = compute_mech_stats(frame, skills, grit, bonus_effects=effects)
    return stats.model_dump()
