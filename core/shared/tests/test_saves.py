"""Tests for save resolution module."""

from core.shared.saves import (
    SaveRequest,
    SaveResult,
    SaveDifficultyModifier,
    resolve_save,
    compute_save_target,
    resolve_save_against_damage,
)


class TestSaveRequest:
    """Tests for SaveRequest model."""

    def test_basic_save_request(self):
        """Test creating a basic save request."""
        request = SaveRequest(
            save_type="hull",
            save_target=10,
        )
        assert request.save_type == "hull"
        assert request.save_target == 10
        assert request.save_bonus == 0
        assert request.target_conditions == []

    def test_save_request_with_bonus(self):
        """Test save request with skill bonus."""
        request = SaveRequest(
            save_type="agility",
            save_target=12,
            save_bonus=3,
        )
        assert request.save_bonus == 3

    def test_save_request_with_conditions(self):
        """Test save request with conditions."""
        request = SaveRequest(
            save_type="systems",
            save_target=10,
            target_conditions=["impaired", "shredded"],
        )
        assert "impaired" in request.target_conditions
        assert "shredded" in request.target_conditions

    def test_save_request_with_force_roll(self):
        """Test save request with forced roll for testing."""
        request = SaveRequest(
            save_type="engineering",
            save_target=8,
            force_roll=15,
        )
        assert request.force_roll == 15


class TestSaveResult:
    """Tests for SaveResult model."""

    def test_save_result_success(self):
        """Test save result for successful save."""
        result = SaveResult(
            save_type="hull",
            roll=12,
            roll_with_bonus=15,
            total=15,
            target=10,
            success=True,
            degree="success",
            difficulty_modifier=0,
            accuracy_modifier=0,
            flat_bonus=3,
        )
        assert result.success is True
        assert result.degree == "success"

    def test_save_result_failure(self):
        """Test save result for failed save."""
        result = SaveResult(
            save_type="agility",
            roll=8,
            roll_with_bonus=8,
            total=8,
            target=12,
            success=False,
            degree="failure",
            difficulty_modifier=2,
            accuracy_modifier=0,
            flat_bonus=0,
        )
        assert result.success is False
        assert result.degree == "failure"

    def test_save_result_critical_success(self):
        """Test save result for critical success."""
        result = SaveResult(
            save_type="systems",
            roll=20,
            roll_with_bonus=23,
            total=23,
            target=15,
            success=True,
            degree="critical_success",
            difficulty_modifier=0,
            accuracy_modifier=0,
            flat_bonus=3,
        )
        assert result.roll == 20
        assert result.degree == "critical_success"

    def test_save_result_critical_failure(self):
        """Test save result for critical failure."""
        result = SaveResult(
            save_type="engineering",
            roll=1,
            roll_with_bonus=1,
            total=1,
            target=8,
            success=False,
            degree="critical_failure",
            difficulty_modifier=0,
            accuracy_modifier=0,
            flat_bonus=0,
        )
        assert result.roll == 1
        assert result.degree == "critical_failure"


class TestResolveSave:
    """Tests for resolve_save function."""

    def test_basic_success(self):
        """Test basic successful save."""
        request = SaveRequest(
            save_type="hull",
            save_target=10,
            save_bonus=2,
            force_roll=12,
        )
        result = resolve_save(request)
        assert result.success is True
        assert result.total == 14
        assert result.degree == "success"

    def test_basic_failure(self):
        """Test basic failed save."""
        request = SaveRequest(
            save_type="agility",
            save_target=15,
            save_bonus=1,
            force_roll=8,
        )
        result = resolve_save(request)
        assert result.success is False
        assert result.total == 9
        assert result.degree == "failure"

    def test_tie_is_success(self):
        """Test that tie (total == target) is a success."""
        request = SaveRequest(
            save_type="systems",
            save_target=10,
            save_bonus=3,
            force_roll=7,
        )
        result = resolve_save(request)
        assert result.success is True
        assert result.total == 10
        assert result.target == 10

    def test_critical_success_natural_20(self):
        """Test critical success on natural 20."""
        request = SaveRequest(
            save_type="engineering",
            save_target=20,
            save_bonus=0,
            force_roll=20,
        )
        result = resolve_save(request)
        assert result.roll == 20
        assert result.success is True
        assert result.degree == "critical_success"

    def test_critical_failure_natural_1(self):
        """Test critical failure on natural 1."""
        request = SaveRequest(
            save_type="hull",
            save_target=5,
            save_bonus=5,
            force_roll=1,
        )
        result = resolve_save(request)
        assert result.roll == 1
        assert result.success is False
        assert result.degree == "critical_failure"

    def test_impaired_condition_adds_difficulty(self):
        """Test that Impaired condition adds +1 difficulty."""
        request = SaveRequest(
            save_type="agility",
            save_target=10,
            save_bonus=3,
            target_conditions=["impaired"],
            force_roll=8,
        )
        result = resolve_save(request)
        assert result.difficulty_modifier == 1
        assert result.total == 10
        assert result.success is True

    def test_shredded_condition_no_save_difficulty(self):
        """Test that Shredded condition does NOT add save difficulty (only ignores armor/resistance)."""
        request = SaveRequest(
            save_type="systems",
            save_target=10,
            save_bonus=2,
            target_conditions=["shredded"],
            force_roll=9,
        )
        result = resolve_save(request)
        assert result.difficulty_modifier == 0
        assert result.total == 11
        assert result.success is True

    def test_combined_conditions_impaired_only(self):
        """Test multiple conditions - only impaired adds save difficulty."""
        request = SaveRequest(
            save_type="engineering",
            save_target=10,
            save_bonus=2,
            target_conditions=["impaired", "shredded"],
            force_roll=9,
        )
        result = resolve_save(request)
        assert result.difficulty_modifier == 1
        assert result.total == 10
        assert result.success is True

    def test_all_save_types(self):
        """Test that all save types work correctly."""
        for save_type in ["hull", "agility", "systems", "engineering"]:
            request = SaveRequest(
                save_type=save_type,
                save_target=10,
                save_bonus=3,
                force_roll=12,
            )
            result = resolve_save(request)
            assert result.save_type == save_type
            assert result.success is True

    def test_reason_contains_save_type(self):
        """Test that reason string contains the save type."""
        request = SaveRequest(
            save_type="hull",
            save_target=10,
            save_bonus=2,
            force_roll=15,
        )
        result = resolve_save(request)
        assert "HULL" in result.reason

    def test_reason_indicates_success_or_failure(self):
        """Test that reason indicates success or failure."""
        success_request = SaveRequest(
            save_type="agility",
            save_target=10,
            save_bonus=5,
            force_roll=10,
        )
        success_result = resolve_save(success_request)
        assert "succeeds" in success_result.reason

        failure_request = SaveRequest(
            save_type="systems",
            save_target=15,
            save_bonus=2,
            force_roll=8,
        )
        failure_result = resolve_save(failure_request)
        assert "fails" in failure_result.reason


class TestComputeSaveTarget:
    """Tests for compute_save_target function."""

    def test_basic_save_target(self):
        """Test basic save target computation."""
        target = compute_save_target(base_save_target=10)
        assert target == 10

    def test_with_grit_bonus(self):
        """Test save target with grit bonus."""
        target = compute_save_target(base_save_target=10, grit_bonus=3)
        assert target == 13

    def test_with_condition_modifiers(self):
        """Test save target with condition modifiers."""
        target = compute_save_target(
            base_save_target=10,
            condition_modifiers=2,
        )
        assert target == 12

    def test_with_check_modifiers(self):
        """Test save target with check modifiers."""
        target = compute_save_target(
            base_save_target=10,
            check_modifiers=1,
        )
        assert target == 11

    def test_combined_modifiers(self):
        """Test save target with all modifiers combined."""
        target = compute_save_target(
            base_save_target=10,
            grit_bonus=2,
            condition_modifiers=1,
            check_modifiers=1,
        )
        assert target == 14


class TestResolveSaveAgainstDamage:
    """Tests for resolve_save_against_damage function."""

    def test_damage_half_on_success(self):
        """Test that damage is halved on successful save."""
        success, damage = resolve_save_against_damage(
            save_type="agility",
            damage_amount=10,
            save_target=10,
            save_bonus=5,
            half_on_save=True,
            force_roll=10,
        )
        assert success is True
        assert damage == 5

    def test_full_damage_on_failure(self):
        """Test that full damage is taken on failed save."""
        success, damage = resolve_save_against_damage(
            save_type="hull",
            damage_amount=10,
            save_target=15,
            save_bonus=2,
            half_on_save=True,
            force_roll=8,
        )
        assert success is False
        assert damage == 10

    def test_no_half_on_save(self):
        """Test with half_on_save=False."""
        success, damage = resolve_save_against_damage(
            save_type="systems",
            damage_amount=8,
            save_target=10,
            save_bonus=5,
            half_on_save=False,
            force_roll=10,
        )
        assert success is True
        assert damage == 8

    def test_conditions_affect_damage_save(self):
        """Test that conditions affect damage save resolution."""
        success, damage = resolve_save_against_damage(
            save_type="engineering",
            damage_amount=10,
            save_target=13,
            save_bonus=3,
            target_conditions=["impaired"],
            half_on_save=True,
            force_roll=10,
        )
        assert success is False
        assert damage == 10

    def test_all_save_types_damage(self):
        """Test all save types with damage resolution."""
        for save_type in ["hull", "agility", "systems", "engineering"]:
            success, damage = resolve_save_against_damage(
                save_type=save_type,
                damage_amount=6,
                save_target=8,
                save_bonus=4,
                half_on_save=True,
                force_roll=8,
            )
            assert success is True
            assert damage == 3


class TestSaveDifficultyModifier:
    """Tests for SaveDifficultyModifier model."""

    def test_basic_modifier(self):
        """Test creating a basic difficulty modifier."""
        modifier = SaveDifficultyModifier(
            source="impaired",
            value=1,
        )
        assert modifier.source == "impaired"
        assert modifier.value == 1

    def test_modifier_with_save_types(self):
        """Test modifier that applies to specific save types."""
        modifier = SaveDifficultyModifier(
            source="electronic_interference",
            value=2,
            applies_to=["systems", "engineering"],
        )
        assert modifier.applies_to == ["systems", "engineering"]

    def test_accuracy_modifier(self):
        """Test negative value as accuracy modifier."""
        modifier = SaveDifficultyModifier(
            source="acrobatics",
            value=-1,
        )
        assert modifier.value == -1


class TestSaveEdgeCases:
    """Edge case tests for save resolution."""

    def test_zero_save_target(self):
        """Test save against zero target with natural 20 succeeds, natural 1 fails."""
        request_success = SaveRequest(
            save_type="hull",
            save_target=0,
            force_roll=20,
        )
        result_success = resolve_save(request_success)
        assert result_success.success is True
        assert result_success.degree == "critical_success"

        request_failure = SaveRequest(
            save_type="hull",
            save_target=0,
            force_roll=1,
        )
        result_failure = resolve_save(request_failure)
        assert result_failure.success is False
        assert result_failure.degree == "critical_failure"

    def test_high_save_target_failure(self):
        """Test save against very high target."""
        request = SaveRequest(
            save_type="agility",
            save_target=25,
            save_bonus=5,
            force_roll=19,
        )
        result = resolve_save(request)
        assert result.success is False

    def test_negative_save_bonus(self):
        """Test save with negative bonus (penalty)."""
        request = SaveRequest(
            save_type="systems",
            save_target=10,
            save_bonus=-2,
            force_roll=15,
        )
        result = resolve_save(request)
        assert result.flat_bonus == -2
        assert result.success is True

    def test_modifier_breakdown(self):
        """Test that modifier breakdown is populated."""
        request = SaveRequest(
            save_type="hull",
            save_target=10,
            save_bonus=3,
            target_conditions=["impaired"],
            force_roll=12,
        )
        result = resolve_save(request)
        assert "condition:impaired" in result.modifier_breakdown
        assert result.modifier_breakdown["condition:impaired"] == 1

    def test_multiple_same_type_conditions(self):
        """Test multiple instances of same condition type."""
        request = SaveRequest(
            save_type="engineering",
            save_target=10,
            save_bonus=2,
            target_conditions=["impaired", "impaired"],
            force_roll=11,
        )
        result = resolve_save(request)
        assert result.difficulty_modifier == 2
        assert result.total == 11
        assert result.success is True
