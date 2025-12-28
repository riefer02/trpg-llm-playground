from core.mech.build import build_mech_from_compendium
from core.mech.compendium import (
    SYSTEM_DEFINITIONS_BY_ID,
    WEAPON_DEFINITIONS_BY_ID,
    get_frame_definition,
)
from core.mech.validation import validate_mech_build
from core.pilot.skill import SkillSet


def test_weapon_license_rank_required() -> None:
    frame = get_frame_definition("gms_everest")
    assert frame is not None
    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[(0, "hand_cannon")],
        system_ids=[],
    )
    skills = SkillSet(hull=0, agility=0, systems=0, engineering=0)

    blocked = validate_mech_build(
        frame,
        build,
        skills,
        grit=0,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        license_ranks={"raleigh": 0},
    )
    error_codes = {issue.code for issue in blocked.issues if issue.severity == "error"}
    assert "license_requirement_not_met" in error_codes

    allowed = validate_mech_build(
        frame,
        build,
        skills,
        grit=0,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        license_ranks={"raleigh": 1},
    )
    allowed_errors = {issue.code for issue in allowed.issues if issue.severity == "error"}
    assert "license_requirement_not_met" not in allowed_errors
