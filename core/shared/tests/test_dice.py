import pytest

from core.shared.dice import DiceExpression, round_up


def test_dice_expression_parse_and_stats() -> None:
    expr = DiceExpression.parse("2d6+3")
    assert expr.count == 2
    assert expr.size == 6
    assert expr.modifier == 3
    assert str(expr) == "2d6+3"
    assert expr.min_value() == 5
    assert expr.max_value() == 15
    assert expr.average() == 10.0


def test_dice_expression_negative_modifier() -> None:
    expr = DiceExpression.parse("1d20-1")
    assert expr.count == 1
    assert expr.size == 20
    assert expr.modifier == -1
    assert str(expr) == "1d20-1"
    assert expr.min_value() == 0
    assert expr.max_value() == 19


def test_dice_expression_parse_invalid() -> None:
    with pytest.raises(ValueError):
        DiceExpression.parse("not-a-roll")


def test_round_up_whole_numbers() -> None:
    assert round_up(5.0) == 5
    assert round_up(10.0) == 10
    assert round_up(1.0) == 1


def test_round_up_fractions() -> None:
    assert round_up(5.1) == 6
    assert round_up(5.5) == 6
    assert round_up(5.9) == 6


def test_round_up_negative() -> None:
    assert round_up(-1.0) == -1
    assert round_up(-0.5) == 0
    assert round_up(-1.1) == -1


def test_round_up_zero() -> None:
    assert round_up(0.0) == 0
    assert round_up(0.1) == 1
