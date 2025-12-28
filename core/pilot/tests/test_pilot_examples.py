import pytest

from core.pilot.examples import (
    evaluate_example_downtime_plan,
    evaluate_example_pilot_ll0,
    evaluate_oda_ll0_example,
    evaluate_oda_ll3_example,
)


def _assert_no_errors(validation) -> None:
    errors = [issue for issue in validation.issues if issue.severity == "error"]
    assert not errors, f"Validation errors: {errors}"


def test_example_pilot_ll0_valid() -> None:
    assert evaluate_example_pilot_ll0() is True


def test_example_downtime_plan_valid() -> None:
    assert evaluate_example_downtime_plan() is True


@pytest.mark.parametrize(
    "evaluate_fn",
    [evaluate_oda_ll0_example, evaluate_oda_ll3_example],
)
def test_oda_examples_valid(evaluate_fn) -> None:
    pilot_validation, mech_validation, mismatches = evaluate_fn()
    _assert_no_errors(pilot_validation)
    _assert_no_errors(mech_validation)
    assert mismatches == []
