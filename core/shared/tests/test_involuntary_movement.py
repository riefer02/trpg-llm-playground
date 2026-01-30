"""Tests for involuntary movement resolution helpers."""

from core.shared.involuntary_movement import (
    InvoluntaryMovementType,
    InvoluntaryMovementPath,
    PushResult,
    PullResult,
    KnockbackResult,
    DragResult,
    LiftResult,
    can_drag,
    can_lift,
    resolve_drag,
    resolve_lift,
    apply_drag_penalty,
    apply_lift_penalty,
    get_involuntary_movement_result,
    is_involuntary_movement,
    breaks_grapple,
    validate_straight_line_path,
    resolve_push,
    resolve_pull,
    resolve_knockback,
    resolve_shove,
)
from core.shared.enums import StatusType
from core.mech.grid import HexCoord


class TestValidateStraightLinePath:
    """Tests for straight-line path validation."""

    def test_clear_path(self):
        """Clear path should have path_clear=True."""
        start = HexCoord(q=0, r=0)
        direction = HexCoord(q=1, r=0)
        path = validate_straight_line_path(start, direction, 3)
        assert path.path_clear is True
        assert path.spaces_moved == 3
        assert path.end == HexCoord(q=3, r=0)

    def test_obstructed_path(self):
        """Path with obstruction should be blocked."""
        start = HexCoord(q=0, r=0)
        direction = HexCoord(q=1, r=0)
        occupied = {HexCoord(q=2, r=0): True}
        path = validate_straight_line_path(start, direction, 5, occupied_hexes=occupied)
        assert path.path_clear is False
        assert path.spaces_moved == 1
        assert path.obstructions == [HexCoord(q=2, r=0)]

    def test_zero_spaces(self):
        """Zero spaces should return immediately."""
        start = HexCoord(q=0, r=0)
        path = validate_straight_line_path(start, HexCoord(q=1, r=0), 0)
        assert path.path_clear is True
        assert path.spaces_moved == 0

    def test_diagonal_path(self):
        """Diagonal direction should work."""
        start = HexCoord(q=0, r=0)
        direction = HexCoord(q=1, r=-1)
        path = validate_straight_line_path(start, direction, 2)
        assert path.path_clear is True
        assert path.spaces_moved == 2


class TestResolvePush:
    """Tests for push resolution."""

    def test_push_successful(self):
        """Push should succeed when path is clear."""
        result = resolve_push(
            HexCoord(q=0, r=0),
            HexCoord(q=1, r=0),
            2,
        )
        assert result.spaces_pushed == 2
        assert result.end_position == HexCoord(q=2, r=0)
        assert result.obstructed is False

    def test_push_blocked(self):
        """Push should stop at obstruction."""
        occupied = {HexCoord(q=1, r=0): True}
        result = resolve_push(
            HexCoord(q=0, r=0),
            HexCoord(q=1, r=0),
            3,
            occupied_hexes=occupied,
        )
        assert result.spaces_pushed == 0
        assert result.obstructed is True
        assert result.obstruction_coord == HexCoord(q=1, r=0)


class TestResolvePull:
    """Tests for pull resolution."""

    def test_pull_successful(self):
        """Pull should move target toward source."""
        result = resolve_pull(
            source=HexCoord(q=0, r=0),
            target=HexCoord(q=5, r=0),
            spaces=2,
        )
        assert result.spaces_pulled == 2
        assert result.end_position == HexCoord(q=3, r=0)
        assert result.obstructed is False

    def test_pull_blocked(self):
        """Pull should stop at obstruction."""
        occupied = {HexCoord(q=4, r=0): True}
        result = resolve_pull(
            source=HexCoord(q=0, r=0),
            target=HexCoord(q=5, r=0),
            spaces=5,
            occupied_hexes=occupied,
        )
        assert result.spaces_pulled == 0
        assert result.obstructed is True


class TestResolveKnockback:
    """Tests for knockback resolution."""

    def test_knockback_successful(self):
        """Knockback should move target directly away."""
        result = resolve_knockback(
            source=HexCoord(q=0, r=0),
            target=HexCoord(q=2, r=0),
            spaces=2,
        )
        assert result.spaces_knocked == 2
        assert result.end_position == HexCoord(q=4, r=0)
        assert result.direction.q == 1
        assert result.direction.r == 0
        assert result.obstructed is False

    def test_knockback_blocked(self):
        """Knockback should stop at obstruction."""
        occupied = {HexCoord(q=3, r=0): True}
        result = resolve_knockback(
            source=HexCoord(q=0, r=0),
            target=HexCoord(q=2, r=0),
            spaces=5,
            occupied_hexes=occupied,
        )
        assert result.spaces_knocked == 0
        assert result.obstructed is True

    def test_knockback_direction_correct(self):
        """Knockback direction should be away from source."""
        result = resolve_knockback(
            source=HexCoord(q=5, r=5),
            target=HexCoord(q=5, r=3),
            spaces=2,
        )
        assert result.direction.q == 0
        assert result.direction.r == -1


class TestResolveShove:
    """Tests for shove resolution."""

    def test_shove_successful(self):
        """Shove should work like push."""
        result = resolve_shove(
            HexCoord(q=0, r=0),
            HexCoord(q=0, r=1),
            2,
        )
        assert result.spaces_shoved == 2
        assert result.end_position == HexCoord(q=0, r=2)


class TestCanDrag:
    """Tests for drag size checking."""

    def test_drag_within_limit(self):
        """Drag within size limit should succeed."""
        can, reason, max_size = can_drag("size_2", "size_2")
        assert can is True
        assert max_size == 4

    def test_drag_exceeds_limit(self):
        """Drag exceeding size limit should fail."""
        can, reason, max_size = can_drag("size_1", "size_3")
        assert can is False
        assert "Cannot drag" in reason

    def test_drag_pilot_limit(self):
        """Pilots have stricter drag limits."""
        can, reason, max_size = can_drag("size_1", "size_1", is_pilot=True)
        assert can is False
        assert "Pilots cannot drag" in reason

    def test_drag_pilot_allowed(self):
        """Pilots can drag size 1/2."""
        can, reason, max_size = can_drag("size_1", "size_half", is_pilot=True)
        assert can is True
        assert max_size == 1


class TestCanLift:
    """Tests for lift size checking."""

    def test_lift_within_limit(self):
        """Lift within own size should succeed."""
        can, reason = can_lift("size_2", "size_2")
        assert can is True

    def test_lift_exceeds_limit(self):
        """Lift exceeding own size should fail."""
        can, reason = can_lift("size_1", "size_2")
        assert can is False
        assert "Cannot lift" in reason

    def test_lift_pilot_limit(self):
        """Pilots have stricter lift limits."""
        can, reason = can_lift("size_1", "size_1", is_pilot=True)
        assert can is False

    def test_lift_pilot_allowed(self):
        """Pilots can lift size 1/2."""
        can, reason = can_lift("size_1", "size_half", is_pilot=True)
        assert can is True


class TestResolveDrag:
    """Tests for drag resolution."""

    def test_drag_success(self):
        """Drag should apply slowed condition."""
        result = resolve_drag(
            dragger_size="size_2",
            dragged_size="size_2",
            target_coord=HexCoord(q=3, r=0),
        )
        assert result.can_drag is True
        assert result.slowed_applied is True
        assert result.end_position == HexCoord(q=3, r=0)

    def test_drag_failure(self):
        """Drag exceeding limit should fail."""
        result = resolve_drag(
            dragger_size="size_1",
            dragged_size="size_3",
            target_coord=HexCoord(q=3, r=0),
        )
        assert result.can_drag is False


class TestResolveLift:
    """Tests for lift resolution."""

    def test_lift_success(self):
        """Lift should apply immobilized condition."""
        result = resolve_lift("size_2", "size_2")
        assert result.can_lift is True
        assert result.immobilized_applied is True
        assert result.lifted_overhead is True

    def test_lift_failure(self):
        """Lift exceeding limit should fail."""
        result = resolve_lift("size_1", "size_2")
        assert result.can_lift is False


class TestConditionApplication:
    """Tests for condition application during drag/lift."""

    def test_drag_penalty(self):
        """Drag should apply slowed."""
        conditions: list[StatusType] = []
        result = apply_drag_penalty(conditions)
        assert result.applied is True
        assert result.condition == "slowed"

    def test_lift_penalty(self):
        """Lift should apply immobilized."""
        conditions: list[StatusType] = []
        result = apply_lift_penalty(conditions)
        assert result.applied is True
        assert result.condition == "immobilized"


class TestInvoluntaryMovementHelpers:
    """Tests for helper functions."""

    def test_is_involuntary_movement(self):
        """Should identify involuntary movement types."""
        assert is_involuntary_movement("push") is True
        assert is_involuntary_movement("pull") is True
        assert is_involuntary_movement("knockback") is True
        assert is_involuntary_movement("shove") is True
        assert is_involuntary_movement("drag") is True
        assert is_involuntary_movement("lift") is True
        assert is_involuntary_movement("boost") is False
        assert is_involuntary_movement("skirmish") is False

    def test_breaks_grapple(self):
        """Should identify movements that break grapple."""
        assert breaks_grapple("knockback") is True
        assert breaks_grapple("shove") is True
        assert breaks_grapple("push") is True
        assert breaks_grapple("pull") is False
        assert breaks_grapple("drag") is False
        assert breaks_grapple("lift") is False

    def test_get_involuntary_movement_result(self):
        """Should create standardized result."""
        result = get_involuntary_movement_result(
            movement_type="push",
            start=HexCoord(q=0, r=0),
            end=HexCoord(q=2, r=0),
            spaces_moved=2,
            path_clear=True,
            obstructed=False,
        )
        assert result.movement_type == "push"
        assert result.spaces_moved == 2
        assert result.ignored_engagement is True
        assert result.provoked_reactions is False


class TestPushResult:
    """Tests for PushResult model."""

    def test_push_result_fields(self):
        """PushResult should have all expected fields."""
        result = PushResult(
            spaces_pushed=3,
            end_position=HexCoord(q=3, r=0),
            obstructed=False,
        )
        assert result.spaces_pushed == 3
        assert result.end_position == HexCoord(q=3, r=0)
        assert result.obstructed is False


class TestPullResult:
    """Tests for PullResult model."""

    def test_pull_result_fields(self):
        """PullResult should have all expected fields."""
        result = PullResult(
            spaces_pulled=2,
            end_position=HexCoord(q=3, r=0),
            obstructed=False,
        )
        assert result.spaces_pulled == 2
        assert result.end_position == HexCoord(q=3, r=0)


class TestKnockbackResult:
    """Tests for KnockbackResult model."""

    def test_knockback_result_fields(self):
        """KnockbackResult should have all expected fields."""
        result = KnockbackResult(
            spaces_knocked=2,
            end_position=HexCoord(q=4, r=0),
            obstructed=False,
            direction=HexCoord(q=1, r=0),
        )
        assert result.spaces_knocked == 2
        assert result.direction.q == 1


class TestDragResult:
    """Tests for DragResult model."""

    def test_drag_result_fields(self):
        """DragResult should have all expected fields."""
        result = DragResult(
            dragger_size="size_2",
            dragged_size="size_2",
            can_drag=True,
            max_drag_size=4,
            slowed_applied=True,
        )
        assert result.dragger_size == "size_2"
        assert result.max_drag_size == 4
        assert result.slowed_applied is True


class TestLiftResult:
    """Tests for LiftResult model."""

    def test_lift_result_fields(self):
        """LiftResult should have all expected fields."""
        result = LiftResult(
            lifter_size="size_2",
            lifted_size="size_2",
            can_lift=True,
            immobilized_applied=True,
            lifted_overhead=True,
        )
        assert result.lifted_overhead is True
        assert result.immobilized_applied is True


class TestMovementType:
    """Tests for InvoluntaryMovementType model."""

    def test_movement_type_fields(self):
        """InvoluntaryMovementType should have all expected fields."""
        movement = InvoluntaryMovementType(
            type="knockback",
            spaces=2,
            direction=HexCoord(q=1, r=0),
        )
        assert movement.type == "knockback"
        assert movement.spaces == 2


class TestMovementPath:
    """Tests for InvoluntaryMovementPath model."""

    def test_movement_path_fields(self):
        """InvoluntaryMovementPath should have all expected fields."""
        path = InvoluntaryMovementPath(
            start=HexCoord(q=0, r=0),
            end=HexCoord(q=3, r=0),
            spaces_moved=3,
            spaces_requested=5,
            path_clear=False,
            obstructions=[HexCoord(q=3, r=0)],
        )
        assert path.spaces_requested == 5
        assert len(path.obstructions) == 1
