"""Tests for per-target counter resolution."""

from core.shared.effects import PerTargetCounter
from core.mech.combat_resolution import (
    resolve_per_target_counter,
)


def test_resolve_per_target_counter_within_limit() -> None:
    """Counter increments successfully when within max_count."""
    counter = PerTargetCounter(
        effect_id="basilisk_stun",
        current_count=0,
        max_count=1,
        reset_on="scene_end",
        target_id="target_1",
    )

    result = resolve_per_target_counter(counter=counter)

    assert result.effect_id == "basilisk_stun"
    assert result.target_id == "target_1"
    assert result.previous_count == 0
    assert result.new_count == 1
    assert result.was_applied is True
    assert result.limit_exceeded is False


def test_resolve_per_target_counter_at_limit() -> None:
    """Counter at max_count prevents further application."""
    counter = PerTargetCounter(
        effect_id="basilisk_stun",
        current_count=1,
        max_count=1,
        reset_on="scene_end",
        target_id="target_1",
    )

    result = resolve_per_target_counter(counter=counter)

    assert result.effect_id == "basilisk_stun"
    assert result.previous_count == 1
    assert result.new_count == 1
    assert result.was_applied is False
    assert result.limit_exceeded is True


def test_resolve_per_target_counter_exceeds_limit() -> None:
    """Applying multiple counts that exceed max_count is prevented."""
    counter = PerTargetCounter(
        effect_id="h0r_os_invasion",
        current_count=0,
        max_count=2,
        reset_on="scene_end",
        target_id="target_2",
    )

    result = resolve_per_target_counter(counter=counter, applied_count=3)

    assert result.previous_count == 0
    assert result.new_count == 0
    assert result.was_applied is False
    assert result.limit_exceeded is True


def test_resolve_per_target_counter_multiple_applications() -> None:
    """Counter can be applied multiple times up to limit."""
    counter = PerTargetCounter(
        effect_id="h0r_os_invasion",
        current_count=0,
        max_count=3,
        reset_on="scene_end",
        target_id="target_3",
    )

    result1 = resolve_per_target_counter(counter=counter)
    assert result1.previous_count == 0
    assert result1.new_count == 1
    assert result1.was_applied is True
    assert result1.limit_exceeded is False

    counter = counter.model_copy(update={"current_count": result1.new_count})
    result2 = resolve_per_target_counter(counter=counter)
    assert result2.previous_count == 1
    assert result2.new_count == 2
    assert result2.was_applied is True

    counter = counter.model_copy(update={"current_count": result2.new_count})
    result3 = resolve_per_target_counter(counter=counter)
    assert result3.previous_count == 2
    assert result3.new_count == 3
    assert result3.was_applied is True

    counter = counter.model_copy(update={"current_count": result3.new_count})
    result4 = resolve_per_target_counter(counter=counter)
    assert result4.previous_count == 3
    assert result4.new_count == 3
    assert result4.was_applied is False
    assert result4.limit_exceeded is True


def test_resolve_per_target_counter_no_target_id() -> None:
    """Counter without target_id uses empty string."""
    counter = PerTargetCounter(
        effect_id="basilisk_stun_template",
        current_count=0,
        max_count=1,
        reset_on="scene_end",
        target_id=None,
    )

    result = resolve_per_target_counter(counter=counter)

    assert result.target_id == ""


def test_reset_per_round_reactions() -> None:
    """Test that per-round reactions are properly reset."""
    from core.mech.combat_state import CombatantState, CombatStats, CombatResources
    from core.mech.combat_resolution import reset_per_round_reactions

    combatant = CombatantState(
        id="mech_1",
        name="Test Mech",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=10,
            evasion=8,
            e_defense=10,
        ),
        resources=CombatResources(hp_current=10),
        per_round_reactions={"overwatch": 1, "brace": 1},
    )

    updated = reset_per_round_reactions(combatant)

    assert updated.per_round_reactions == {}
    assert updated.id == "mech_1"
    assert updated.stats.hp_max == 10


def test_reset_per_round_reactions_empty() -> None:
    """Test that reset works on combatant with no reactions used."""
    from core.mech.combat_state import CombatantState, CombatStats, CombatResources
    from core.mech.combat_resolution import reset_per_round_reactions

    combatant = CombatantState(
        id="mech_2",
        name="Fresh Mech",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=10,
            evasion=8,
            e_defense=10,
        ),
        resources=CombatResources(hp_current=10),
        per_round_reactions={},
    )

    updated = reset_per_round_reactions(combatant)

    assert updated.per_round_reactions == {}
