"""Tests for Hide/Search mechanics.

Tests cover:
- can_hide() requirements (hard cover, soft cover area, LOS, invisible, engaged)
- is_soft_cover_area() zone detection
- attempt_hide() action resolution
- check_hidden_broken() condition lifecycle
- can_search() sensor/range requirements
- attempt_search() contested checks
- resolve_disengage() engagement state
"""

import pytest
from core.shared.hide_search import (
    HideAttempt,
    HideResult,
    SearchAttempt,
    SearchResult,
    can_hide,
    is_soft_cover_area,
    attempt_hide,
    check_hidden_broken,
    can_search,
    attempt_search,
    resolve_disengage,
    is_hidden_target_revealable,
    get_cover_for_hiding,
)
from core.mech.grid import HexCoord
from core.mech.terrain import TerrainMap, TerrainHex
from core.shared.enums import StatusType


class TestCanHide:
    """Tests for can_hide() function."""

    def test_can_hide_with_hard_cover(self):
        """Hard cover allows hiding."""
        result, reason = can_hide(
            has_hard_cover=True,
            is_in_soft_cover_area=False,
            has_los=True,
            is_invisible=False,
            is_engaged=False,
        )
        assert result is True
        assert "hard cover" in reason

    def test_cannot_hide_when_engaged(self):
        """Cannot hide when engaged, even with cover."""
        result, reason = can_hide(
            has_hard_cover=True,
            is_in_soft_cover_area=True,
            has_los=False,
            is_invisible=False,
            is_engaged=True,
        )
        assert result is False
        assert "engaged" in reason.lower()

    def test_can_hide_invisible(self):
        """Invisible characters can always hide (even without cover)."""
        result, reason = can_hide(
            has_hard_cover=False,
            is_in_soft_cover_area=False,
            has_los=True,
            is_invisible=True,
            is_engaged=False,
        )
        assert result is True
        assert "invisible" in reason.lower()

    def test_can_hide_without_los(self):
        """Lack of line of sight is sufficient for hiding."""
        result, reason = can_hide(
            has_hard_cover=False,
            is_in_soft_cover_area=False,
            has_los=False,
            is_invisible=False,
            is_engaged=False,
        )
        assert result is True
        assert "line of sight" in reason.lower()

    def test_can_hide_in_soft_cover_area(self):
        """Soft cover area allows hiding."""
        result, reason = can_hide(
            has_hard_cover=False,
            is_in_soft_cover_area=True,
            has_los=True,
            is_invisible=False,
            is_engaged=False,
        )
        assert result is True
        assert "soft cover" in reason.lower()

    def test_cannot_hide_without_cover(self):
        """Cannot hide without cover or invisibility."""
        result, reason = can_hide(
            has_hard_cover=False,
            is_in_soft_cover_area=False,
            has_los=True,
            is_invisible=False,
            is_engaged=False,
        )
        assert result is False
        assert "need" in reason.lower()


class TestIsSoftCoverArea:
    """Tests for is_soft_cover_area() function."""

    def test_no_terrain(self):
        """No terrain means no soft cover area."""
        result = is_soft_cover_area(None, HexCoord(q=0, r=0))
        assert result is False

    def test_single_hex_soft_cover(self):
        """Single hex with soft cover is not an area."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True),
            ]
        )
        result = is_soft_cover_area(terrain, HexCoord(q=0, r=0))
        assert result is False

    def test_two_hexes_soft_cover(self):
        """Two adjacent hexes with soft cover is not an area (needs 3+)."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
            ]
        )
        result = is_soft_cover_area(terrain, HexCoord(q=0, r=0))
        assert result is False

    def test_three_adjacent_hexes(self):
        """Three adjacent hexes with soft cover is an area."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=1, r=-1), provides_soft_cover=True),
            ]
        )
        result = is_soft_cover_area(terrain, HexCoord(q=0, r=0))
        assert result is True

    def test_smoke_cloud_simulation(self):
        """Smoke cloud simulation (blast area) qualifies as soft cover area."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=-1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=0, r=1), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=0, r=-1), provides_soft_cover=True),
            ]
        )
        result = is_soft_cover_area(terrain, HexCoord(q=0, r=0))
        assert result is True

    def test_non_soft_cover_terrain(self):
        """Terrain without soft cover doesn't qualify."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=False),
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=1, r=-1), provides_soft_cover=True),
            ]
        )
        result = is_soft_cover_area(terrain, HexCoord(q=0, r=0))
        assert result is False

    def test_custom_min_adjacent(self):
        """Custom minimum adjacent hexes threshold."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=1, r=-1), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=0, r=-1), provides_soft_cover=True),
            ]
        )
        result = is_soft_cover_area(terrain, HexCoord(q=0, r=0), min_adjacent_hexes=5)
        assert result is False


class TestAttemptHide:
    """Tests for attempt_hide() function."""

    def test_hide_with_hard_cover(self):
        """Hide succeeds with hard cover."""
        attempt = HideAttempt(
            has_hard_cover=True,
            is_in_soft_cover_area=False,
            has_los=True,
            is_invisible=False,
            is_engaged=False,
        )
        result = attempt_hide(attempt)
        assert result.can_hide is True
        assert result.hidden_condition_applied is True

    def test_hide_in_soft_cover_area(self):
        """Hide succeeds in soft cover area."""
        attempt = HideAttempt(
            has_hard_cover=False,
            is_in_soft_cover_area=True,
            has_los=True,
            is_invisible=False,
            is_engaged=False,
        )
        result = attempt_hide(attempt)
        assert result.can_hide is True
        assert result.hidden_condition_applied is True

    def test_hide_when_engaged_fails(self):
        """Hide fails when engaged."""
        attempt = HideAttempt(
            has_hard_cover=True,
            is_in_soft_cover_area=False,
            has_los=False,
            is_invisible=False,
            is_engaged=True,
        )
        result = attempt_hide(attempt)
        assert result.can_hide is False
        assert result.hidden_condition_applied is False

    def test_hide_invisible_succeeds(self):
        """Hide succeeds when invisible and not engaged."""
        attempt = HideAttempt(
            has_hard_cover=False,
            is_in_soft_cover_area=False,
            has_los=True,
            is_invisible=True,
            is_engaged=False,
        )
        result = attempt_hide(attempt)
        assert result.can_hide is True
        assert result.hidden_condition_applied is True


class TestCheckHiddenBroken:
    """Tests for check_hidden_broken() function."""

    def test_attack_breaks_hidden(self):
        """Attacking breaks hidden condition."""
        result = check_hidden_broken(
            current_conditions=["hidden"],
            action_type="attack",
            has_los=True,
            has_cover=True,
            is_invisible=False,
        )
        assert result.hidden_broken is True
        assert result.break_type == "attack"

    def test_boost_breaks_hidden(self):
        """Boosting breaks hidden condition."""
        result = check_hidden_broken(
            current_conditions=["hidden"],
            action_type="boost",
            has_los=True,
            has_cover=True,
            is_invisible=False,
        )
        assert result.hidden_broken is True
        assert result.break_type == "boost"

    def test_reaction_breaks_hidden(self):
        """Taking a reaction breaks hidden condition."""
        result = check_hidden_broken(
            current_conditions=["hidden"],
            action_type="reaction",
            has_los=True,
            has_cover=True,
            is_invisible=False,
        )
        assert result.hidden_broken is True
        assert result.break_type == "reaction"

    def test_cover_lost_breaks_hidden(self):
        """Losing cover benefit due to LOS breaks hidden."""
        result = check_hidden_broken(
            current_conditions=["hidden"],
            action_type="none",
            has_los=True,
            has_cover=False,
            is_invisible=False,
        )
        assert result.hidden_broken is True
        assert result.break_type == "cover_lost"

    def test_cover_destroyed_breaks_hidden(self):
        """Cover being destroyed breaks hidden."""
        result = check_hidden_broken(
            current_conditions=["hidden"],
            action_type="none",
            has_los=True,
            has_cover=True,
            is_invisible=False,
            cover_destroyed=True,
        )
        assert result.hidden_broken is True
        assert result.break_type == "cover_destroyed"

    def test_invisibility_lost_breaks_hidden(self):
        """Losing invisibility breaks hidden if hiding while invisible."""
        result = check_hidden_broken(
            current_conditions=["hidden", "invisible"],
            action_type="none",
            has_los=True,
            has_cover=False,
            is_invisible=False,
        )
        assert result.hidden_broken is True
        assert result.break_type == "invisibility_lost"

    def test_hidden_maintained_with_cover(self):
        """Hidden condition maintained when cover is still valid."""
        result = check_hidden_broken(
            current_conditions=["hidden"],
            action_type="none",
            has_los=False,
            has_cover=True,
            is_invisible=False,
        )
        assert result.hidden_broken is False
        assert result.break_type == "none"

    def test_not_hidden(self):
        """Non-hidden character returns not broken."""
        result = check_hidden_broken(
            current_conditions=[],
            action_type="none",
            has_los=True,
            has_cover=True,
            is_invisible=False,
        )
        assert result.hidden_broken is False
        assert result.break_type == "none"


class TestCanSearch:
    """Tests for can_search() function."""

    def test_can_search_mech_in_sensor_range(self):
        """Can search for mech in sensor range."""
        result, reason = can_search(
            target_in_sensor_range=True,
            is_pilot_search=False,
            pilot_in_range_5=False,
        )
        assert result is True
        assert "sensor range" in reason.lower()

    def test_cannot_search_mech_out_of_sensor_range(self):
        """Cannot search for mech outside sensor range."""
        result, reason = can_search(
            target_in_sensor_range=False,
            is_pilot_search=False,
            pilot_in_range_5=False,
        )
        assert result is False
        assert "sensor range" in reason.lower()

    def test_can_search_pilot_in_range_5(self):
        """Pilot can search if in range 5."""
        result, reason = can_search(
            target_in_sensor_range=False,
            is_pilot_search=True,
            pilot_in_range_5=True,
        )
        assert result is True
        assert "pilot" in reason.lower()

    def test_cannot_search_pilot_out_of_range_5(self):
        """Pilot cannot search if not in range 5."""
        result, reason = can_search(
            target_in_sensor_range=False,
            is_pilot_search=True,
            pilot_in_range_5=False,
        )
        assert result is False
        assert "range 5" in reason.lower()


class TestAttemptSearch:
    """Tests for attempt_search() function."""

    def test_search_success_mech(self):
        """Mech search contested check mechanics."""
        attempt = SearchAttempt(
            searcher_systems_bonus=5,
            target_agility_bonus=0,
            target_in_sensor_range=True,
            is_pilot_search=False,
        )
        result = attempt_search(attempt)
        assert result.search_roll is not None
        assert result.target_roll is not None
        assert result.search_total is not None
        assert result.target_total is not None
        assert result.search_total == result.search_roll + 5
        assert result.target_total == result.target_roll

    def test_search_failure_mech(self):
        """Mech search outcome depends on contested rolls."""
        attempt = SearchAttempt(
            searcher_systems_bonus=0,
            target_agility_bonus=5,
            target_in_sensor_range=True,
            is_pilot_search=False,
        )
        result = attempt_search(attempt)
        assert result.search_roll is not None
        assert result.target_roll is not None
        assert result.search_total is not None
        assert result.target_total is not None
        assert result.search_total == result.search_roll
        assert result.target_total == result.target_roll + 5

    def test_search_roll_values(self):
        """Search rolls are in valid range (1-20)."""
        attempt = SearchAttempt(
            searcher_systems_bonus=0,
            target_agility_bonus=0,
            target_in_sensor_range=True,
            is_pilot_search=False,
        )
        result = attempt_search(attempt)
        assert result.search_roll is not None
        assert result.target_roll is not None
        assert 1 <= result.search_roll <= 20
        assert 1 <= result.target_roll <= 20


class TestResolveDisengage:
    """Tests for resolve_disengage() function."""

    def test_disengage_succeeds(self):
        """Disengage action always succeeds."""
        result = resolve_disengage()
        assert result.disengage_successful is True

    def test_disengage_ignores_engagement(self):
        """Disengage causes movement to ignore engagement."""
        result = resolve_disengage()
        assert result.ignores_engagement is True

    def test_disengage_prevents_reactions(self):
        """Disengage prevents reactions during movement."""
        result = resolve_disengage()
        assert result.prevents_reactions is True

    def test_disengage_duration(self):
        """Disengage effect lasts until end of turn."""
        result = resolve_disengage()
        assert result.duration_turn_end is True


class TestIsHiddenTargetRevealable:
    """Tests for is_hidden_target_revealable() function."""

    def test_revealable_after_successful_search(self):
        """Target is revealable after successful search."""
        result, reason = is_hidden_target_revealable(
            was_hidden=True,
            search_successful=True,
            is_in_sensor_range=True,
        )
        assert result is True

    def test_not_revealable_if_not_hidden(self):
        """Non-hidden target is not revealable."""
        result, reason = is_hidden_target_revealable(
            was_hidden=False,
            search_successful=True,
            is_in_sensor_range=True,
        )
        assert result is False

    def test_not_revealable_after_failed_search(self):
        """Target not revealable after failed search."""
        result, reason = is_hidden_target_revealable(
            was_hidden=True,
            search_successful=False,
            is_in_sensor_range=True,
        )
        assert result is False


class TestGetCoverForHiding:
    """Tests for get_cover_for_hiding() function."""

    def test_no_terrain(self):
        """No terrain means no cover for hiding."""
        result = get_cover_for_hiding(None, HexCoord(q=0, r=0))
        assert result == (False, False, "No terrain, no cover available")

    def test_adjacent_hard_cover(self):
        """Adjacent hard cover detected."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0)),
                TerrainHex(coord=HexCoord(q=1, r=0), provides_hard_cover=True),
            ]
        )
        result = get_cover_for_hiding(terrain, HexCoord(q=0, r=0))
        assert result[0] is True

    def test_soft_cover_area(self):
        """Soft cover area detected."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=1, r=-1), provides_soft_cover=True),
            ]
        )
        result = get_cover_for_hiding(terrain, HexCoord(q=0, r=0))
        assert result[1] is True

    def test_blocks_los_without_cover(self):
        """Blocks LOS but no cover for hiding."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0), blocks_line_of_sight=True),
            ]
        )
        result = get_cover_for_hiding(terrain, HexCoord(q=0, r=0))
        assert result[0] is False
        assert result[1] is False
        assert "line of sight" in result[2].lower()


class TestHideSearchModels:
    """Tests for Hide/Search model validation."""

    def test_hide_attempt_defaults(self):
        """HideAttempt has correct defaults."""
        attempt = HideAttempt()
        assert attempt.has_hard_cover is False
        assert attempt.is_in_soft_cover_area is False
        assert attempt.has_los is True
        assert attempt.is_invisible is False
        assert attempt.is_engaged is False
        assert attempt.current_conditions == []

    def test_search_attempt_defaults(self):
        """SearchAttempt has correct defaults."""
        attempt = SearchAttempt()
        assert attempt.searcher_systems_bonus == 0
        assert attempt.target_agility_bonus == 0
        assert attempt.target_in_sensor_range is False
        assert attempt.is_pilot_search is False
        assert attempt.pilot_in_range_5 is False
        assert attempt.pilot_skill_bonus == 0

    def test_hide_result_defaults(self):
        """HideResult has correct defaults."""
        result = HideResult(can_hide=False)
        assert result.hidden_condition_applied is False
        assert result.reason == ""

    def test_search_result_defaults(self):
        """SearchResult has correct defaults."""
        result = SearchResult(search_successful=False)
        assert result.target_revealed is False
        assert result.search_roll is None
        assert result.search_total is None
        assert result.target_roll is None
        assert result.target_total is None
        assert result.reason == ""
