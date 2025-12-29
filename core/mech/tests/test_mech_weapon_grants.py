from core.mech.build import build_mech_from_compendium
from core.mech.compendium import (
    SYSTEM_DEFINITIONS_BY_ID,
    WEAPON_DEFINITIONS_BY_ID,
    get_frame_definition,
)
from core.mech.validation import validate_mech_build
from core.pilot.skill import SkillSet
from core.shared.dice import DiceExpression
from core.shared.effects import MechanicalEffect, WeaponGrantEffect, WeaponRangeSpec
from core.shared.payloads import DamageSpec


def test_weapon_grant_integrated_mounts_validate() -> None:
    frame = get_frame_definition("gms_everest")
    assert frame is not None
    grant_effect = MechanicalEffect(
        weapon_grants=[
            WeaponGrantEffect(
                weapon_id="test_integrated_grant_weapon",
                name="Test Integrated Gun",
                size="main",
                weapon_type="cqb",
                ranges=[WeaponRangeSpec(range_type="range", value=5)],
                damage=[
                    DamageSpec(
                        damage_type="energy",
                        dice=DiceExpression.parse("1d3"),
                    )
                ],
                integrated_mount=True,
            )
        ]
    )
    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[(len(frame.mounts), "test_integrated_grant_weapon")],
        system_ids=[],
        bonus_effects=[grant_effect],
    )
    skills = SkillSet(hull=0, agility=0, systems=0, engineering=0)
    validation = validate_mech_build(
        frame,
        build,
        skills,
        grit=0,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        bonus_effects=[grant_effect],
    )
    error_codes = {issue.code for issue in validation.issues if issue.severity == "error"}
    assert "mount_index_out_of_range" not in error_codes
    assert "integrated_weapon_requires_integrated_mount" not in error_codes
