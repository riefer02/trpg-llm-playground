"""Tests for cloning system (Priority 55).

Tests cover:
- CloneState and CloneStatus models
- Down and Out resolution
- Flash cloning process
- Quirk system
"""

import pytest
from core.pilot.clone_state import (
    CloneStatus,
    CloneState,
    Quirk,
    QuirkType,
    QuirkSource,
)
from core.pilot.down_and_out import (
    resolve_down_and_out,
    DownAndOutInput,
    apply_pilot_rest,
    PilotRestInput,
    resolve_pilot_death,
    PilotDeathResolutionInput,
)
from core.pilot.flash_clone import (
    can_be_cloned,
    create_flash_clone,
    FlashCloneInput,
    PilotSnapshot,
    check_second_clone_eligibility,
    SecondCloneCheckInput,
)
from core.pilot.quirks import (
    get_quirk_by_roll,
    get_physical_quirks,
    get_mental_quirks,
    roll_random_quirk,
    apply_quirk,
    QuirkApplicationInput,
    generate_narrative_prompts,
    get_all_quirks,
    count_quirks,
    QUIRK_TABLE,
)


class TestCloneStatus:
    """Tests for CloneStatus model."""

    def test_default_status(self):
        """Default CloneStatus should be alive and uncloned."""
        status = CloneStatus()
        assert status.times_cloned == 0
        assert status.is_dead is False
        assert status.clone_available is True

    def test_clone_available_when_dead(self):
        """Clone should be available even when dead if not cloned before."""
        status = CloneStatus(is_dead=True)
        assert status.clone_available is True

    def test_clone_not_available_after_clone(self):
        """Clone should not be available after being cloned."""
        status = CloneStatus(times_cloned=1, is_dead=False)
        assert status.clone_available is False

    def test_clone_not_available_when_dead_and_cloned(self):
        """Clone should not be available when dead and already cloned."""
        status = CloneStatus(times_cloned=1, is_dead=True)
        assert status.clone_available is False

    def test_can_be_revived(self):
        """Pilot can be revived if dead and not cloned before."""
        status = CloneStatus(is_dead=True)
        assert status.can_be_revived is True

    def test_cannot_be_revived_after_clone(self):
        """Pilot cannot be revived if already cloned."""
        status = CloneStatus(times_cloned=1, is_dead=True)
        assert status.can_be_revived is False

    def test_mark_cloned(self):
        """mark_cloned should increment clone count and set alive."""
        status = CloneStatus(times_cloned=0, is_dead=True)
        new_status = status.mark_cloned()
        assert new_status.times_cloned == 1
        assert new_status.is_dead is False

    def test_mark_dead(self):
        """mark_dead should set is_dead to True."""
        status = CloneStatus()
        new_status = status.mark_dead()
        assert new_status.is_dead is True
        assert new_status.times_cloned == 0

    def test_mark_alive(self):
        """mark_alive should set is_dead to False."""
        status = CloneStatus(is_dead=True)
        new_status = status.mark_alive()
        assert new_status.is_dead is False


class TestCloneState:
    """Tests for CloneState model."""

    def test_default_clone_state(self):
        """Default CloneState should have default values."""
        state = CloneState()
        assert state.status.times_cloned == 0
        assert state.status.is_dead is False
        assert state.assigned_quirk is None
        assert state.quirk_source is None
        assert state.clone_applicable is True

    def test_is_cloned_property(self):
        """is_cloned should return True if times_cloned > 0."""
        state = CloneState(status=CloneStatus(times_cloned=1))
        assert state.is_cloned is True

        state_uncloned = CloneState(status=CloneStatus(times_cloned=0))
        assert state_uncloned.is_cloned is False

    def test_can_be_cloned_property(self):
        """can_be_cloned should check clone availability."""
        state = CloneState(
            status=CloneStatus(times_cloned=0, is_dead=True), clone_applicable=True
        )
        assert state.can_be_cloned is True

        state_used = CloneState(
            status=CloneStatus(times_cloned=1, is_dead=True), clone_applicable=True
        )
        assert state_used.can_be_cloned is False

        state_disabled = CloneState(
            status=CloneStatus(times_cloned=0, is_dead=True), clone_applicable=False
        )
        assert state_disabled.can_be_cloned is False

    def test_with_quirk(self):
        """with_quirk should create new state with quirk assigned."""
        state = CloneState()
        quirk = get_quirk_by_roll(1)
        new_state = state.with_quirk(quirk, "clone")

        assert new_state.assigned_quirk == quirk
        assert new_state.quirk_source == "clone"
        assert new_state.has_quirk is True

    def test_with_increased_clone_count(self):
        """with_increased_clone_count should increment clone count."""
        state = CloneState(status=CloneStatus(times_cloned=0))
        new_state = state.with_increased_clone_count()
        assert new_state.status.times_cloned == 1

    def test_with_session_snapshot(self):
        """with_session_snapshot should record HP and evasion."""
        state = CloneState()
        new_state = state.with_session_snapshot(hp=12, evasion=10)
        assert new_state.session_start_hp == 12
        assert new_state.session_start_evasion == 10


class TestQuirks:
    """Tests for Quirk system."""

    def test_quirk_table_has_20_quirks(self):
        """QUIRK_TABLE should have exactly 20 quirks."""
        assert count_quirks() == 20

    def test_get_quirk_by_roll(self):
        """get_quirk_by_roll should return correct quirk."""
        quirk = get_quirk_by_roll(1)
        assert quirk is not None
        assert quirk.roll == 1
        assert quirk.name == "Cybernetic Replacement"

    def test_get_quirk_by_invalid_roll(self):
        """get_quirk_by_roll should return None for invalid roll."""
        assert get_quirk_by_roll(0) is None
        assert get_quirk_by_roll(21) is None

    def test_get_physical_quirks(self):
        """get_physical_quirks should return only physical quirks."""
        physical = get_physical_quirks()
        assert len(physical) > 0
        for quirk in physical:
            assert quirk.quirk_type == "physical"

    def test_get_mental_quirks(self):
        """get_mental_quirks should return only mental quirks."""
        mental = get_mental_quirks()
        assert len(mental) > 0
        for quirk in mental:
            assert quirk.quirk_type == "mental"

    def test_physical_and_mental_cover_all_quirks(self):
        """All quirks should be either physical or mental."""
        physical = get_physical_quirks()
        mental = get_mental_quirks()
        assert len(physical) + len(mental) == 20

    def test_roll_random_quirk(self):
        """roll_random_quirk should return a valid quirk."""
        quirk = roll_random_quirk()
        assert quirk is not None
        assert 1 <= quirk.roll <= 20

    def test_apply_quirk(self):
        """apply_quirk should create application result."""
        quirk = get_quirk_by_roll(1)
        input_data = QuirkApplicationInput(
            pilot_id="pilot_1", quirk=quirk, source="clone", existing_quirks=[]
        )
        result = apply_quirk(input_data)

        assert result.applied is True
        assert result.quirk == quirk
        assert result.source == "clone"
        assert result.total_quirks == 1
        assert len(result.narrative_prompts) > 0

    def test_generate_narrative_prompts(self):
        """generate_narrative_prompts should return non-empty list."""
        quirk = get_quirk_by_roll(1)
        prompts = generate_narrative_prompts(quirk)
        assert len(prompts) > 0

    def test_all_quirks_have_descriptions(self):
        """All quirks should have non-empty descriptions."""
        for quirk in get_all_quirks():
            assert quirk.description is not None
            assert len(quirk.description) > 0

    def test_all_quirks_have_names(self):
        """All quirks should have non-empty names."""
        for quirk in get_all_quirks():
            assert quirk.name is not None
            assert len(quirk.name) > 0


class TestDownAndOutResolution:
    """Tests for Down and Out resolution."""

    def test_recover_on_6(self):
        """Rolling 6 should recover to 1 HP."""
        input_data = DownAndOutInput(
            pilot_id="pilot_1", current_hp=0, max_hp=10, base_evasion=10
        )

        result = resolve_down_and_out(input_data, roll_result=6)

        assert result.outcome == "recovered"
        assert result.hp_after == 1
        assert result.evasion_after == 10
        assert result.quirk_eligible is False

    def test_down_and_out_on_2_5(self):
        """Rolling 2-5 should result in Down and Out."""
        input_data = DownAndOutInput(
            pilot_id="pilot_1", current_hp=0, max_hp=10, base_evasion=10
        )

        result = resolve_down_and_out(input_data, roll_result=3)

        assert result.outcome == "down_and_out"
        assert result.hp_after == 0
        assert result.evasion_after == 5
        assert result.quirk_eligible is True

    def test_death_on_1(self):
        """Rolling 1 should result in death."""
        input_data = DownAndOutInput(
            pilot_id="pilot_1", current_hp=0, max_hp=10, base_evasion=10
        )

        result = resolve_down_and_out(input_data, roll_result=1)

        assert result.outcome == "died"
        assert result.hp_after == 0
        assert result.evasion_after == 0

    def test_voluntary_death(self):
        """Voluntary death should return voluntary_death outcome."""
        input_data = DownAndOutInput(
            pilot_id="pilot_1",
            current_hp=0,
            max_hp=10,
            base_evasion=10,
            voluntary_death=True,
        )

        result = resolve_down_and_out(input_data)

        assert result.outcome == "voluntary_death"
        assert result.roll_result is None

    def test_short_rest_recovery(self):
        """1 hour rest should recover 1/2 max HP."""
        input_data = PilotRestInput(
            pilot_id="pilot_1",
            current_hp=3,
            max_hp=10,
            hours_rested=1,
            is_down_and_out=True,
        )

        result = apply_pilot_rest(input_data)

        assert result.hp_after == 8
        assert result.hp_recovered == 5
        assert result.down_and_out_cleared is True

    def test_full_rest_recovery(self):
        """10+ hour rest should recover to full HP."""
        input_data = PilotRestInput(
            pilot_id="pilot_1",
            current_hp=5,
            max_hp=10,
            hours_rested=10,
            is_down_and_out=True,
        )

        result = apply_pilot_rest(input_data)

        assert result.hp_after == 10
        assert result.hp_recovered == 5
        assert result.is_recovered is True
        assert result.down_and_out_cleared is True


class TestPilotDeathResolution:
    """Tests for pilot death resolution."""

    def test_clone_available_for_first_death(self):
        """Clone should be available for first death."""
        input_data = PilotDeathResolutionInput(
            pilot_id="pilot_1",
            death_circumstances="Killed in action",
            has_previous_clone=False,
            clone_allowed=True,
        )

        result = resolve_pilot_death(input_data)

        assert result.clone_available is True
        assert result.is_permanent_death is False

    def test_permanent_death_for_second_clone(self):
        """Second death should be permanent death."""
        input_data = PilotDeathResolutionInput(
            pilot_id="pilot_1",
            death_circumstances="Killed in action",
            has_previous_clone=True,
            clone_allowed=True,
        )

        result = resolve_pilot_death(input_data)

        assert result.clone_available is False
        assert result.is_permanent_death is True

    def test_permanent_death_when_clone_not_allowed(self):
        """Death should be permanent if clone not allowed."""
        input_data = PilotDeathResolutionInput(
            pilot_id="pilot_1",
            death_circumstances="Killed in action",
            has_previous_clone=False,
            clone_allowed=False,
        )

        result = resolve_pilot_death(input_data)

        assert result.clone_available is False
        assert result.is_permanent_death is True


class TestFlashCloning:
    """Tests for flash cloning process."""

    def test_can_be_cloned_with_none_state(self):
        """can_be_cloned should return True when state is None."""
        assert can_be_cloned(None) is True

    def test_can_be_cloned_with_available_state(self):
        """can_be_cloned should return True when clone available."""
        state = CloneState(status=CloneStatus(times_cloned=0, is_dead=True))
        assert can_be_cloned(state) is True

    def test_cannot_clone_if_already_cloned(self):
        """can_be_cloned should return False if already cloned."""
        state = CloneState(status=CloneStatus(times_cloned=1, is_dead=True))
        assert can_be_cloned(state) is False

    def test_create_flash_clone(self):
        """create_flash_clone should create valid clone result."""
        snapshot = PilotSnapshot(hp=10, evasion=10)
        input_data = FlashCloneInput(
            original_pilot_id="pilot_1",
            session_snapshot=snapshot,
            current_level=3,
            current_skills={},
            current_talents=[],
            current_licenses=[],
            current_core_bonuses=[],
            current_gear=[],
            clone_allowed=True,
            party_aware=False,
        )
        quirk = get_quirk_by_roll(1)

        result = create_flash_clone(input_data, quirk)

        assert result.success is True
        assert result.clone_count == 1
        assert result.quirk == quirk
        assert result.session_hp == 10
        assert result.level_preserved == 3

    def test_clone_not_allowed(self):
        """create_flash_clone should fail if clone not allowed."""
        snapshot = PilotSnapshot(hp=10, evasion=10)
        input_data = FlashCloneInput(
            original_pilot_id="pilot_1",
            session_snapshot=snapshot,
            current_level=3,
            current_skills={},
            current_talents=[],
            current_licenses=[],
            current_core_bonuses=[],
            current_gear=[],
            clone_allowed=False,
            party_aware=False,
        )
        quirk = get_quirk_by_roll(1)

        result = create_flash_clone(input_data, quirk)

        assert result.success is False
        assert result.failure_reason is not None

    def test_second_clone_not_allowed(self):
        """check_second_clone_eligibility should reject second clone."""
        input_data = SecondCloneCheckInput(
            pilot_id="pilot_1", has_previous_clone=True, current_clone_state=None
        )

        result = check_second_clone_eligibility(input_data)

        assert result.can_be_cloned is False
        assert result.is_permanent_death is True

    def test_first_clone_allowed(self):
        """check_second_clone_eligibility should allow first clone."""
        input_data = SecondCloneCheckInput(
            pilot_id="pilot_1", has_previous_clone=False, current_clone_state=None
        )

        result = check_second_clone_eligibility(input_data)

        assert result.can_be_cloned is True
        assert result.is_permanent_death is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
