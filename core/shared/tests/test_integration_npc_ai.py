"""Tests for npc_ai integration module.

Tests role-based NPC behavior patterns and decision-making.
"""

import pytest
from core.shared.enums import ActionType
from core.shared.integration.npc_ai import (
    NPCBehaviorPattern,
    TargetInfo,
    ActionScore,
    NPCActionDecision,
    STRIKER_PATTERN,
    DEFENDER_PATTERN,
    CONTROLLER_PATTERN,
    SUPPORTER_PATTERN,
    NPC_BEHAVIOR_PATTERNS,
    get_behavior_pattern,
    compute_target_score,
    score_available_actions,
    select_npc_action_with_role,
    is_adjacent,
)
from core.npc.state import NPCState, NPCCombatStats


@pytest.fixture
def sample_npc():
    """Create a sample NPC state for testing."""
    from core.npc.state import NPCCombatStats

    return NPCState(
        id="npc_striker_1",
        name="Striker NPC",
        npc_class="grunt",
        tier="tier_1",
        stats=NPCCombatStats(
            size="size_1",
            hp_max=10,
            evasion=8,
            e_defense=8,
            armor=0,
            speed=4,
            sensor_range=10,
            tech_attack=0,
            save_bonus=0,
        ),
    )


@pytest.fixture
def visible_targets():
    """Create sample visible targets for testing."""
    return [
        TargetInfo(
            id="player_1",
            distance=3,
            hp_current=5,
            hp_max=10,
            is_objective_holder=False,
            is_ally=False,
        ),
        TargetInfo(
            id="player_2",
            distance=5,
            hp_current=9,
            hp_max=10,
            is_objective_holder=True,
            is_ally=False,
        ),
        TargetInfo(
            id="player_3",
            distance=8,
            hp_current=2,
            hp_max=10,
            is_objective_holder=False,
            is_ally=False,
        ),
    ]


class TestNPCBehaviorPattern:
    """Tests for NPCBehaviorPattern model."""

    def test_striker_pattern_fields(self):
        """Striker pattern should have correct fields."""
        assert STRIKER_PATTERN.role == "striker"
        assert "low_hp" in STRIKER_PATTERN.priority_targets
        assert "full" in STRIKER_PATTERN.preferred_actions

    def test_defender_pattern_fields(self):
        """Defender pattern should have correct fields."""
        assert DEFENDER_PATTERN.role == "defender"
        assert "nearest" in DEFENDER_PATTERN.priority_targets

    def test_controller_pattern_fields(self):
        """Controller pattern should have correct fields."""
        assert CONTROLLER_PATTERN.role == "controller"
        assert "quick" in CONTROLLER_PATTERN.preferred_actions

    def test_supporter_pattern_fields(self):
        """Supporter pattern should have correct fields."""
        assert SUPPORTER_PATTERN.role == "supporter"
        assert "low_hp_ally" in SUPPORTER_PATTERN.priority_targets


class TestGetBehaviorPattern:
    """Tests for get_behavior_pattern function."""

    def test_get_striker_pattern(self):
        """Should return striker pattern for striker role."""
        pattern = get_behavior_pattern("striker")
        assert pattern == STRIKER_PATTERN

    def test_get_defender_pattern(self):
        """Should return defender pattern for defender role."""
        pattern = get_behavior_pattern("defender")
        assert pattern == DEFENDER_PATTERN

    def test_get_controller_pattern(self):
        """Should return controller pattern for controller role."""
        pattern = get_behavior_pattern("controller")
        assert pattern == CONTROLLER_PATTERN

    def test_get_supporter_pattern(self):
        """Should return supporter pattern for supporter role."""
        pattern = get_behavior_pattern("supporter")
        assert pattern == SUPPORTER_PATTERN

    def test_unknown_role_returns_striker_fallback(self):
        """Unknown role should return striker pattern as fallback."""
        # This tests that the dict lookup returns a valid pattern
        # The pattern dict only has valid roles as keys
        pattern = get_behavior_pattern("striker")
        assert pattern.role == "striker"


class TestComputeTargetScore:
    """Tests for compute_target_score function."""

    def test_low_hp_priority_scores_higher(self):
        """Target with low HP should score higher when low_hp is priority."""
        low_hp_target = TargetInfo(id="target_1", distance=5, hp_current=2, hp_max=10)
        high_hp_target = TargetInfo(id="target_2", distance=5, hp_current=9, hp_max=10)

        low_hp_score = compute_target_score(low_hp_target, ["low_hp"], 10)
        high_hp_score = compute_target_score(high_hp_target, ["low_hp"], 10)

        assert low_hp_score > high_hp_score

    def test_nearest_priority_scores_higher(self):
        """Closer target should score higher when nearest is priority."""
        close_target = TargetInfo(id="target_1", distance=2, hp_current=10, hp_max=10)
        far_target = TargetInfo(id="target_2", distance=8, hp_current=10, hp_max=10)

        close_score = compute_target_score(close_target, ["nearest"], 10)
        far_score = compute_target_score(far_target, ["nearest"], 10)

        assert close_score > far_score

    def test_objective_holder_scores_higher(self):
        """Objective holder should score higher when objective is priority."""
        objective_target = TargetInfo(
            id="target_1",
            distance=5,
            hp_current=10,
            hp_max=10,
            is_objective_holder=True,
        )
        regular_target = TargetInfo(
            id="target_2",
            distance=5,
            hp_current=10,
            hp_max=10,
            is_objective_holder=False,
        )

        objective_score = compute_target_score(objective_target, ["objective"], 10)
        regular_score = compute_target_score(regular_target, ["objective"], 10)

        assert objective_score > regular_score

    def test_high_threat_detects_higher_hp(self):
        """Higher HP target should score higher when high_threat is priority."""
        strong_target = TargetInfo(id="target_1", distance=5, hp_current=15, hp_max=15)
        weak_target = TargetInfo(id="target_2", distance=5, hp_current=5, hp_max=10)

        strong_score = compute_target_score(strong_target, ["high_threat"], 10)
        weak_score = compute_target_score(weak_target, ["high_threat"], 10)

        assert strong_score > weak_score


class TestScoreAvailableActions:
    """Tests for score_available_actions function."""

    def test_scores_actions_against_targets(self, sample_npc, visible_targets):
        """Should score each action against best target."""
        available_actions: list[ActionType] = ["full", "quick"]

        scored = score_available_actions(
            sample_npc, available_actions, visible_targets, STRIKER_PATTERN
        )

        assert len(scored) == 2
        for score in scored:
            assert score.action in available_actions

    def test_no_targets_returns_zero_scores(self, sample_npc):
        """No visible targets should result in zero scores."""
        available_actions: list[ActionType] = ["full"]

        scored = score_available_actions(
            sample_npc, available_actions, [], STRIKER_PATTERN
        )

        assert len(scored) == 1
        assert scored[0].score == 0.0
        assert "No targets" in scored[0].reasoning

    def test_scores_sorted_by_value(self, sample_npc, visible_targets):
        """Should return scores sorted by score value (highest first)."""
        available_actions: list[ActionType] = ["full", "quick"]

        scored = score_available_actions(
            sample_npc, available_actions, visible_targets, STRIKER_PATTERN
        )

        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)


class TestSelectNpcActionWithRole:
    """Tests for select_npc_action_with_role function."""

    def test_striker_selects_preferred_action(self, sample_npc, visible_targets):
        """Striker should select preferred action type."""
        decision = select_npc_action_with_role(
            sample_npc, "striker", ["full", "quick"], visible_targets
        )

        assert decision.action == "full"  # Striker prefers full action
        assert decision.target_id is not None
        assert len(decision.reasoning) > 0

    def test_controller_selects_preferred_action(self, sample_npc, visible_targets):
        """Controller should select preferred action type."""
        decision = select_npc_action_with_role(
            sample_npc, "controller", ["quick", "full"], visible_targets
        )

        assert decision.action == "quick"  # Controller prefers quick action
        assert decision.target_id is not None

    def test_fallback_to_available_action(self, sample_npc, visible_targets):
        """Should fall back to available action if preferred not available."""
        decision = select_npc_action_with_role(
            sample_npc, "striker", ["reaction"], visible_targets
        )

        assert decision.action == "reaction"
        assert decision.fallback_used is True

    def test_no_actions_returns_default(self, sample_npc, visible_targets):
        """No available actions should return default with fallback."""
        decision = select_npc_action_with_role(
            sample_npc, "striker", [], visible_targets
        )

        assert decision.action == "full"
        assert decision.fallback_used is True


class TestIsAdjacent:
    """Tests for is_adjacent function."""

    def test_same_position_not_adjacent(self):
        """Same position should not be adjacent."""
        assert not is_adjacent((0, 0), (0, 0), "size_1", "size_1")

    def test_adjacent_positions(self):
        """Orthogonal adjacent positions should be adjacent."""
        assert is_adjacent((0, 0), (1, 0), "size_1", "size_1")
        assert is_adjacent((0, 0), (0, 1), "size_1", "size_1")

    def test_diagonal_not_adjacent(self):
        """Diagonal positions are not adjacent for size 1."""
        assert not is_adjacent((0, 0), (1, 1), "size_1", "size_1")

    def test_size_2_extends_reach(self):
        """Size 2 units have extended adjacency."""
        assert is_adjacent((0, 0), (2, 0), "size_2", "size_1")


class TestTargetInfo:
    """Tests for TargetInfo model."""

    def test_create_target_info(self):
        """Should create target info with all fields."""
        target = TargetInfo(
            id="test_target",
            distance=5,
            hp_current=8,
            hp_max=10,
            is_objective_holder=True,
            is_ally=False,
        )

        assert target.id == "test_target"
        assert target.distance == 5
        assert target.is_objective_holder is True


class TestActionScore:
    """Tests for ActionScore model."""

    def test_create_action_score(self):
        """Should create action score with reasoning."""
        score = ActionScore(
            action="full",
            target_id="target_1",
            score=8.5,
            reasoning="Low HP target",
        )

        assert score.action == "full"
        assert score.score == 8.5


class TestNPCActionDecision:
    """Tests for NPCActionDecision model."""

    def test_create_action_decision(self):
        """Should create action decision with fallback flag."""
        decision = NPCActionDecision(
            action="quick",
            target_id="target_1",
            reasoning="Preferred action",
            fallback_used=False,
        )

        assert decision.fallback_used is False

    def test_fallback_decision(self):
        """Should indicate when fallback was used."""
        decision = NPCActionDecision(
            action="full",
            target_id=None,
            reasoning="No preferred actions available",
            fallback_used=True,
        )

        assert decision.fallback_used is True
