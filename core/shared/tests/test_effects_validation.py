import pytest

from core.shared.dice import DiceExpression
from core.shared.effects import (
    CountdownDieEffect,
    CountdownDieTrigger,
    DicePoolEffect,
    DicePoolGain,
    DicePoolSpendOption,
    MechanicalEffect,
)
from core.shared.effects_validation import (
    merge_countdown_dice_by_name,
    merge_dice_pools_from_effects,
    merge_dice_pools_by_name,
    validate_mechanical_effect,
)


def test_merge_dice_pools_conflict() -> None:
    first = DicePoolEffect(pool_name="test", die_size=6)
    second = DicePoolEffect(pool_name="test", die_size=8)
    merged, issues = merge_dice_pools_by_name([first, second])
    assert len(merged) == 1
    assert any(issue.severity == "error" for issue in issues)


def test_merge_dice_pools_combines_triggers_and_spends() -> None:
    first = DicePoolEffect(
        pool_name="blade",
        die_size=6,
        gain_triggers=[DicePoolGain(trigger="on_hit")],
    )
    second = DicePoolEffect(
        pool_name="blade",
        die_size=6,
        spend_options=[
            DicePoolSpendOption(
                name="Parry",
                action_type="reaction",
                dice_cost=1,
                effect=MechanicalEffect(),
            )
        ],
    )
    merged, issues = merge_dice_pools_by_name([first, second])
    assert not issues
    assert merged[0].gain_triggers
    assert merged[0].spend_options


def test_merge_countdown_dice_conflict() -> None:
    first = CountdownDieEffect(die_name="test", die_size=6)
    second = CountdownDieEffect(die_name="test", die_size=8)
    merged, issues = merge_countdown_dice_by_name([first, second])
    assert len(merged) == 1
    assert any(issue.severity == "error" for issue in issues)


def test_validate_mechanical_effect_collects_issues() -> None:
    effect = MechanicalEffect(
        dice_pools=[
            DicePoolEffect(pool_name="test", die_size=6),
            DicePoolEffect(pool_name="test", die_size=8),
        ]
    )
    issues = validate_mechanical_effect(effect)
    assert any(issue.severity == "error" for issue in issues)


def test_merge_dice_pools_from_effects_resolves_unset_fields() -> None:
    effect_one = MechanicalEffect(
        dice_pools=[
            DicePoolEffect(
                pool_name="blade",
                die_size=6,
                max_dice=3,
                gain_triggers=[DicePoolGain(trigger="on_hit")],
            )
        ]
    )
    effect_two = MechanicalEffect(
        dice_pools=[
            DicePoolEffect(
                pool_name="blade",
                spend_options=[
                    DicePoolSpendOption(
                        name="Parry",
                        action_type="reaction",
                        dice_cost=1,
                        effect=MechanicalEffect(),
                    )
                ],
            )
        ]
    )
    merged, issues = merge_dice_pools_from_effects([effect_one, effect_two])
    assert not issues
    assert merged[0].max_dice == 3
    assert merged[0].gain_triggers
    assert merged[0].spend_options


def test_validate_mechanical_effect_warns_on_empty_spend_pool() -> None:
    effect = MechanicalEffect(
        dice_pools=[
            DicePoolEffect(
                pool_name="empty",
                die_size=6,
                spend_options=[
                    DicePoolSpendOption(
                        name="Spend",
                        action_type="reaction",
                        dice_cost=1,
                        effect=MechanicalEffect(),
                    )
                ],
            )
        ]
    )
    issues = validate_mechanical_effect(effect)
    assert any(issue.severity == "warning" for issue in issues)


def test_validate_mechanical_effect_errors_on_spend_over_max() -> None:
    effect = MechanicalEffect(
        dice_pools=[
            DicePoolEffect(
                pool_name="over",
                die_size=6,
                max_dice=1,
                spend_options=[
                    DicePoolSpendOption(
                        name="Spend",
                        action_type="reaction",
                        dice_cost=2,
                        effect=MechanicalEffect(),
                    )
                ],
            )
        ]
    )
    issues = validate_mechanical_effect(effect)
    assert any(issue.severity == "error" for issue in issues)


def test_validate_mechanical_effect_warns_on_countdown_without_decrement() -> None:
    effect = MechanicalEffect(
        countdown_dice=[
            CountdownDieEffect(
                die_name="storm",
                die_size=6,
                starting_value=6,
                minimum_value=1,
                spend_requires_value=1,
                reset_value=6,
                spend_options=[
                    DicePoolSpendOption(
                        name="Spend",
                        action_type="full",
                        dice_cost=1,
                        effect=MechanicalEffect(),
                    )
                ],
            )
        ]
    )
    issues = validate_mechanical_effect(effect)
    assert any(issue.severity == "warning" for issue in issues)


def test_spend_option_requires_cost_or_roll_pair() -> None:
    with pytest.raises(ValueError):
        DicePoolSpendOption(
            name="Deflect",
            action_type="reaction",
            dice_cost=1,
            spend_any_number=True,
            effect=MechanicalEffect(),
        )
    with pytest.raises(ValueError):
        DicePoolSpendOption(
            name="Deflect",
            action_type="reaction",
            dice_cost=None,
            spend_any_number=False,
            roll=DiceExpression.parse("1d6"),
            effect=MechanicalEffect(),
        )


def test_countdown_validation_range() -> None:
    with pytest.raises(ValueError):
        CountdownDieEffect(die_name="test", die_size=6, starting_value=7)
    with pytest.raises(ValueError):
        CountdownDieEffect(
            die_name="test",
            die_size=6,
            starting_value=6,
            minimum_value=2,
            reset_value=1,
        )


def test_countdown_roll_threshold_pairing() -> None:
    with pytest.raises(ValueError):
        DicePoolSpendOption(
            name="Roll",
            action_type="reaction",
            dice_cost=1,
            roll_threshold=5,
            effect=MechanicalEffect(),
        )
    with pytest.raises(ValueError):
        DicePoolSpendOption(
            name="Roll",
            action_type="reaction",
            dice_cost=1,
            roll=DiceExpression.parse("1d6"),
            effect=MechanicalEffect(),
        )
