"""Tests for Grapple and Ram resolution helpers."""

from core.shared.grapple import (
    GrappleAttempt,
    GrappleResult,
    RamAttempt,
    can_grapple,
    attempt_grapple,
    resolve_grapple_status,
    end_grapple,
    contest_grapple_check,
    calculate_group_grapple_size,
    can_ram,
    attempt_ram,
    resolve_disarm_on_grapple,
    get_knockback_direction,
    is_valid_grapple_size,
    is_larger_in_grapple,
)
from core.shared.enums import SizeClass
from core.mech.grid import HexCoord


class TestCanGrapple:
    """Tests for can_grapple function."""

    def test_grapple_allowed_with_mount(self):
        """Grapple should be allowed when attacker has mount."""
        can, reason = can_grapple("size_1", "size_1", attacker_has_mount=True)
        assert can is True
        assert reason == "Grapple can be initiated"

    def test_grapple_denied_without_mount(self):
        """Grapple should be denied when attacker lacks mount."""
        can, reason = can_grapple("size_1", "size_1", attacker_has_mount=False)
        assert can is False
        assert "lacks appropriate mount" in reason

    def test_grapple_allowed_any_size(self):
        """Grapple should be allowed for any valid size combination."""
        sizes: list[SizeClass] = [
            "size_half",
            "size_1",
            "size_2",
            "size_3",
            "size_4",
            "size_5",
        ]
        for attacker_size in sizes:
            for target_size in sizes:
                can, _ = can_grapple(attacker_size, target_size)
                assert can is True

    def test_grapple_denied_stunned_target(self):
        """Grapple should be denied when target is stunned."""
        can, reason = can_grapple("size_1", "size_1", target_conditions=["stunned"])
        assert can is False
        assert "stunned" in reason.lower()


class TestAttemptGrapple:
    """Tests for attempt_grapple function."""

    def test_grapple_miss(self):
        """Grapple should fail when attack misses."""
        attempt = GrappleAttempt(
            attacker_size="size_1",
            target_size="size_1",
            hit=False,
        )
        result = attempt_grapple(attempt)
        assert result.hit is False
        assert result.grapple_initiated is False
        assert result.smaller_party == "tie"

    def test_grapple_larger_attacker(self):
        """Larger attacker should not be immobilized."""
        attempt = GrappleAttempt(
            attacker_size="size_2",
            target_size="size_1",
            hit=True,
        )
        result = attempt_grapple(attempt)
        assert result.hit is True
        assert result.grapple_initiated is True
        assert result.attacker_engaged is True
        assert result.target_engaged is True
        assert result.smaller_party == "target"
        assert result.target_becomes_immobilized is True
        assert result.attacker_becomes_immobilized is False
        assert result.contested_check_required is False

    def test_grapple_smaller_attacker(self):
        """Smaller attacker should be immobilized."""
        attempt = GrappleAttempt(
            attacker_size="size_1",
            target_size="size_2",
            hit=True,
        )
        result = attempt_grapple(attempt)
        assert result.hit is True
        assert result.grapple_initiated is True
        assert result.smaller_party == "attacker"
        assert result.attacker_becomes_immobilized is True
        assert result.contested_check_required is False

    def test_grapple_same_size_requires_contested_check(self):
        """Same-size grapple should require contested HULL check."""
        attempt = GrappleAttempt(
            attacker_size="size_1",
            target_size="size_1",
            hit=True,
            attacker_hull_bonus=2,
            target_hull_bonus=1,
        )
        result = attempt_grapple(attempt)
        assert result.hit is True
        assert result.grapple_initiated is True
        assert result.smaller_party == "tie"
        assert result.contested_check_required is True
        assert result.contested_check_winner in ["attacker", "target", "tie"]

    def test_grapple_same_size_attacker_wins(self):
        """Attacker should win contested check with higher total."""
        attempt = GrappleAttempt(
            attacker_size="size_1",
            target_size="size_1",
            hit=True,
            attacker_hull_bonus=5,
            target_hull_bonus=1,
        )
        result = attempt_grapple(attempt)
        assert result.contested_check_winner == "attacker"


class TestResolveGrappleStatus:
    """Tests for resolve_grapple_status function."""

    def test_grapple_status_larger_attacker(self):
        """Status should reflect attacker as larger."""
        status = resolve_grapple_status(
            attacker_size="size_2",
            target_size="size_1",
        )
        assert status.active is True
        assert status.attacker_grappling is True
        assert status.target_grappled is True
        assert status.smaller_party == "target"
        assert status.immobilized_party == "target"
        assert status.can_boost is False
        assert status.can_take_reactions is False
        assert status.contested_check_pending is False

    def test_grapple_status_same_size(self):
        """Same-size grapple should have contested check pending."""
        status = resolve_grapple_status(
            attacker_size="size_1",
            target_size="size_1",
        )
        assert status.active is True
        assert status.smaller_party is None
        assert status.immobilized_party == "none"
        assert status.contested_check_pending is True


class TestEndGrapple:
    """Tests for end_grapple function."""

    def test_attacker_ends_free(self):
        """Attacker should be able to end grapple as free action."""
        result = end_grapple("attacker")
        assert result.ended is True
        assert result.by_initiator is True
        assert result.new_engagement_state == "disengaged"

    def test_defender_ends_quick_action(self):
        """Defender should end grapple with contested check as quick action."""
        result = end_grapple("target")
        assert result.ended is True
        assert result.by_initiator is False
        assert result.new_engagement_state == "disengaged"


class TestContestGrappleCheck:
    """Tests for contest_grapple_check function."""

    def test_contest_returns_winner(self):
        """Contested check should return a winner."""
        winner, roll = contest_grapple_check(2, 1)
        assert winner in ["attacker", "target", "tie"]
        assert 1 <= roll <= 20

    def test_contest_tie_possible(self):
        """Contested check should allow ties."""
        winner1, _ = contest_grapple_check(3, 3)
        winner2, _ = contest_grapple_check(3, 3)
        winner3, _ = contest_grapple_check(3, 3)
        assert winner1 == winner2 == winner3 == "tie"


class TestGroupGrappleSize:
    """Tests for group grapple size calculation."""

    def test_single_participant(self):
        """Single participant should have total = their size."""
        result = calculate_group_grapple_size(["size_2"])
        assert result.total_size == 2
        assert result.largest_size == "size_2"
        assert result.participant_count == 1

    def test_multiple_participants(self):
        """Multiple participants should sum sizes."""
        result = calculate_group_grapple_size(["size_1", "size_1"])
        assert result.total_size == 2
        assert result.participant_count == 2

    def test_mixed_sizes(self):
        """Mixed sizes should sum correctly."""
        result = calculate_group_grapple_size(["size_1", "size_2", "size_half"])
        assert result.total_size == 4
        assert result.largest_size == "size_2"

    def test_empty_group(self):
        """Empty group should have zero total."""
        result = calculate_group_grapple_size([])
        assert result.total_size == 0
        assert result.largest_size == "size_1"


class TestCanRam:
    """Tests for can_ram function."""

    def test_ram_allowed_with_mount(self):
        """Ram should be allowed with mount."""
        can, reason = can_ram("size_1", "size_1", attacker_has_mount=True)
        assert can is True

    def test_ram_denied_without_mount(self):
        """Ram should be denied without mount."""
        can, reason = can_ram("size_1", "size_1", attacker_has_mount=False)
        assert can is False


class TestAttemptRam:
    """Tests for attempt_ram function."""

    def test_ram_miss(self):
        """Ram should fail on miss."""
        attempt = RamAttempt(
            attacker_size="size_1",
            target_size="size_1",
            hit=False,
        )
        result = attempt_ram(attempt)
        assert result.hit is False
        assert result.target_becomes_prone is False

    def test_ram_hit_prone(self):
        """Ram hit should apply prone."""
        attempt = RamAttempt(
            attacker_size="size_1",
            target_size="size_1",
            hit=True,
        )
        result = attempt_ram(attempt)
        assert result.hit is True
        assert result.target_becomes_prone is True
        assert result.knockback_spaces == 1

    def test_ram_knockback_bonus(self):
        """Ram knockback bonus should add to base."""
        attempt = RamAttempt(
            attacker_size="size_1",
            target_size="size_1",
            hit=True,
            knockback_bonus=2,
        )
        result = attempt_ram(attempt)
        assert result.knockback_spaces == 3


class TestDisarmOnGrapple:
    """Tests for disarm on grapple."""

    def test_disarm_successful(self):
        """Disarm should succeed when target has mount."""
        grapple_result = GrappleResult(
            hit=True,
            grapple_initiated=True,
            smaller_party="target",
        )
        result = resolve_disarm_on_grapple(grapple_result, target_has_mount=True)
        assert result.attempted is True
        assert result.successful is True

    def test_disarm_no_mount(self):
        """Disarm should fail when target has no mount."""
        grapple_result = GrappleResult(
            hit=True,
            grapple_initiated=True,
            smaller_party="target",
        )
        result = resolve_disarm_on_grapple(grapple_result, target_has_mount=False)
        assert result.attempted is True
        assert result.successful is False


class TestKnockbackDirection:
    """Tests for knockback direction calculation."""

    def test_knockback_direction(self):
        """Should calculate correct direction vector."""
        from_coord = HexCoord(q=0, r=0)
        to_coord = HexCoord(q=1, r=0)
        direction = get_knockback_direction(from_coord, to_coord)
        assert direction.q == 1
        assert direction.r == 0


class TestSizeValidation:
    """Tests for size-related validation."""

    def test_all_sizes_valid(self):
        """All size classes should be valid for grappling."""
        sizes: list[SizeClass] = [
            "size_half",
            "size_1",
            "size_2",
            "size_3",
            "size_4",
            "size_5",
        ]
        for size in sizes:
            assert is_valid_grapple_size(size) is True

    def test_larger_comparison(self):
        """Should correctly identify larger size."""
        is_larger, result = is_larger_in_grapple("size_2", "size_1")
        assert is_larger is True
        assert result == "a"

    def test_smaller_comparison(self):
        """Should correctly identify smaller size."""
        is_larger, result = is_larger_in_grapple("size_1", "size_2")
        assert is_larger is False
        assert result == "b"
