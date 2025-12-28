from core.mech.build import build_mech_from_compendium
from core.mech.compendium import (
    SYSTEM_DEFINITIONS_BY_ID,
    WEAPON_DEFINITIONS_BY_ID,
    get_frame_definition,
)
from core.mech.validation import validate_mech_build
from core.pilot.skill import SkillSet


def _get_errors(validation) -> set[str]:
    return {issue.code for issue in validation.issues if issue.severity == "error"}


def test_integrated_weapon_requires_native_frame() -> None:
    frame = get_frame_definition("gms_everest")
    assert frame is not None
    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[(0, "ipsn_m35_mjolnir")],
        system_ids=[],
    )
    skills = SkillSet(hull=0, agility=0, systems=0, engineering=0)
    validation = validate_mech_build(
        frame,
        build,
        skills,
        grit=0,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        license_ranks={"raleigh": 2},
    )
    error_codes = _get_errors(validation)
    assert "integrated_weapon_frame_mismatch" in error_codes


def test_integrated_weapon_valid_on_integrated_mount() -> None:
    frame = get_frame_definition("ipsn_raleigh")
    assert frame is not None
    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[(3, "ipsn_m35_mjolnir")],
        system_ids=[],
    )
    skills = SkillSet(hull=0, agility=0, systems=0, engineering=0)
    validation = validate_mech_build(
        frame,
        build,
        skills,
        grit=0,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        license_ranks={"raleigh": 2},
    )
    error_codes = _get_errors(validation)
    assert "integrated_weapon_requires_integrated_mount" not in error_codes
    assert "integrated_weapon_frame_mismatch" not in error_codes


def test_latch_drone_integrated_weapon_restrictions() -> None:
    frame = get_frame_definition("ipsn_lancaster")
    assert frame is not None
    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[(1, "ipsn_latch_drone")],
        system_ids=[],
    )
    skills = SkillSet(hull=0, agility=0, systems=0, engineering=0)
    validation = validate_mech_build(
        frame,
        build,
        skills,
        grit=0,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        license_ranks={"lancaster": 2},
    )
    error_codes = _get_errors(validation)
    assert "integrated_weapon_requires_integrated_mount" not in error_codes
    assert "integrated_weapon_frame_mismatch" not in error_codes
