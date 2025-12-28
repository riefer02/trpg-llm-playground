import pytest

from core.mech.examples import (
    evaluate_oda_ll0_mech_example,
    evaluate_oda_ll3_mech_example,
    evaluate_example_combat_scenario,
    evaluate_example_combat_scenario_with_area,
    evaluate_example_combat_scenario_with_ai,
    evaluate_example_combat_scenario_with_flight,
    evaluate_example_combat_scenario_with_grapple,
    evaluate_example_combat_scenario_with_line,
    evaluate_example_combat_scenario_with_lock_on_consumption,
    evaluate_example_combat_scenario_with_search,
    evaluate_example_combat_scenario_with_seeking,
    evaluate_example_combat_scenario_with_stabilize,
    evaluate_example_combat_scenario_with_terrain,
    evaluate_structure_and_overheat_examples,
)


def _assert_no_errors(validation) -> None:
    errors = [issue for issue in validation.issues if issue.severity == "error"]
    assert not errors, f"Validation errors: {errors}"


@pytest.mark.parametrize(
    "evaluate_fn",
    [evaluate_oda_ll0_mech_example, evaluate_oda_ll3_mech_example],
)
def test_mech_build_examples_valid(evaluate_fn) -> None:
    validation = evaluate_fn()
    _assert_no_errors(validation)


@pytest.mark.parametrize(
    "evaluate_fn",
    [
        evaluate_example_combat_scenario,
        evaluate_example_combat_scenario_with_terrain,
        evaluate_example_combat_scenario_with_area,
        evaluate_example_combat_scenario_with_line,
        evaluate_example_combat_scenario_with_flight,
        evaluate_example_combat_scenario_with_grapple,
        evaluate_example_combat_scenario_with_stabilize,
        evaluate_example_combat_scenario_with_seeking,
        evaluate_example_combat_scenario_with_search,
        evaluate_example_combat_scenario_with_lock_on_consumption,
    ],
)
def test_combat_examples_valid(evaluate_fn) -> None:
    validation = evaluate_fn()
    _assert_no_errors(validation)


def test_combat_ai_blocks_pilot_actions() -> None:
    validation = evaluate_example_combat_scenario_with_ai()
    error_codes = {issue.code for issue in validation.issues if issue.severity == "error"}
    assert "ai_pilot_action_disallowed" in error_codes


def test_structure_and_overheat_examples_deterministic() -> None:
    results = evaluate_structure_and_overheat_examples()

    structure = results["structure"]
    assert structure["rolls"] == [1, 4]
    assert structure["chosen"] == [1]
    assert structure["outcome"] == "direct_hit"
    assert structure["direct_hit"] == "direct_hit"
    assert structure["spillover"] == 3

    system_trauma = results["system_trauma"]
    assert system_trauma["rolls"] == [3]
    assert system_trauma["chosen"] == [3]
    assert system_trauma["outcome"] == "direct_hit"
    assert system_trauma["direct_hit"] == "direct_hit"
    assert system_trauma["trauma_target"] == "direct_hit"
    assert system_trauma["fallback_reason"] == "none_available"

    overheat = results["overheat"]
    assert overheat["rolls"] == [1, 4]
    assert overheat["chosen"] == [1]
    assert overheat["outcome"] == "meltdown"
    assert overheat["meltdown"] == "meltdown"
