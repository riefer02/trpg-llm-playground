
from core.mech.build import compute_mech_stats
from core.mech.compendium import SYSTEM_DEFINITIONS_BY_ID, WEAPON_DEFINITIONS_BY_ID
from core.mech.examples import build_oda_ll0_mech_example, build_oda_ll3_mech_example
from core.mech.build_validation import validate_mech_build
from core.pilot.core_bonus import get_core_bonus_definition
from core.pilot.examples import build_oda_ll0_pilot, build_oda_ll3_pilot


def _assert_no_errors(validation) -> None:
    errors = [issue for issue in validation.issues if issue.severity == "error"]
    assert not errors, f"Validation errors: {errors}"


# Fidelity note: expected stats mirror PR2 mechanical baselines for the referenced builds.
def test_character_creation_workflow_ll0() -> None:
    pilot = build_oda_ll0_pilot()
    _assert_no_errors(pilot.validate_progression())

    frame, build, skills, grit, effects = build_oda_ll0_mech_example()
    mech_validation = validate_mech_build(
        frame,
        build,
        skills,
        grit,
        licenses=pilot.licenses,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        bonus_effects=effects,
    )
    _assert_no_errors(mech_validation)

    stats = compute_mech_stats(frame, skills, grit, bonus_effects=effects)
    assert stats.hp == 16


def test_character_creation_workflow_ll3() -> None:
    pilot = build_oda_ll3_pilot()
    _assert_no_errors(pilot.validate_progression())

    frame, build, skills, grit, _ = build_oda_ll3_mech_example()
    core_bonus_effects = []
    for core_bonus in pilot.core_bonuses:
        definition = get_core_bonus_definition(core_bonus.core_bonus_id)
        assert definition is not None, f"Missing core bonus: {core_bonus.core_bonus_id}"
        core_bonus_effects.append(definition.effects)
    mech_validation = validate_mech_build(
        frame,
        build,
        skills,
        grit,
        licenses=pilot.licenses,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        bonus_effects=core_bonus_effects,
    )
    _assert_no_errors(mech_validation)

    stats = compute_mech_stats(frame, skills, grit, bonus_effects=core_bonus_effects)
    assert stats.hp == 27
