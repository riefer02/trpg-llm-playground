from core.mech.build import build_mech_from_compendium
from core.mech.compendium import (
    SYSTEM_DEFINITIONS_BY_ID,
    WEAPON_DEFINITIONS_BY_ID,
    get_frame_definition,
)
from core.mech.build_validation import validate_mech_build
from core.pilot.skill import SkillSet
from core.shared.effects import MechanicalEffect, DicePoolEffect, DicePoolSpendOption


def test_mech_build_effect_validation() -> None:
    frame = get_frame_definition("gms_everest")
    assert frame is not None
    build = build_mech_from_compendium(
        frame_id=frame.id,
        weapon_mounts=[(0, "assault_rifle")],
        system_ids=[],
    )
    skills = SkillSet(hull=0, agility=0, systems=0, engineering=0)
    invalid_effect = MechanicalEffect(
        dice_pools=[
            DicePoolEffect(
                pool_name="test_pool",
                die_size=6,
                max_dice=1,
                spend_options=[
                    DicePoolSpendOption(
                        name="Overcost",
                        dice_cost=2,
                        effect=MechanicalEffect(),
                    )
                ],
            )
        ]
    )
    validation = validate_mech_build(
        frame,
        build,
        skills,
        grit=0,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        bonus_effects=[invalid_effect],
    )
    error_codes = {
        issue.code for issue in validation.issues if issue.severity == "error"
    }
    assert "effect_validation_error" in error_codes
