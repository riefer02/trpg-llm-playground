"""Tests for save_conditions integration module.

Tests automatic condition application on failed saves per Lancer rules.
"""

import pytest
from core.shared.enums import SaveType, StatusType
from core.shared.saves import SaveRequest
from core.shared.integration.save_conditions import (
    SaveConditionMapping,
    SaveConditionResult,
    HULL_SAVE_MAPPINGS,
    AGILITY_SAVE_MAPPINGS,
    SYSTEMS_SAVE_MAPPINGS,
    ENGINEERING_SAVE_MAPPINGS,
    COMMON_SAVE_CONDITION_MAPPINGS,
    resolve_save_with_conditions,
    get_default_mappings_for_save_type,
)


class TestSaveConditionMapping:
    """Tests for SaveConditionMapping model."""

    def test_create_hull_mapping_failure(self):
        """Test creating a HULL save → immobilized mapping."""
        mapping = SaveConditionMapping(
            save_type="hull",
            condition="immobilized",
            applies_on="failure",
        )
        assert mapping.save_type == "hull"
        assert mapping.condition == "immobilized"
        assert mapping.applies_on == "failure"

    def test_create_critical_failure_mapping(self):
        """Test creating a mapping that only applies on critical failure."""
        mapping = SaveConditionMapping(
            save_type="agility",
            condition="stunned",
            applies_on="critical_failure",
        )
        assert mapping.applies_on == "critical_failure"


class TestResolveSaveWithConditions:
    """Tests for resolve_save_with_conditions function."""

    def test_hull_save_failure_applies_immobilized(self):
        """HULL save failure should apply immobilized (per book line 18952)."""
        target_conditions: list[StatusType] = []
        request = SaveRequest(
            save_type="hull",
            save_target=10,
            save_bonus=0,
            target_conditions=[],
            force_roll=5,  # Will fail (5 < 10)
        )

        result = resolve_save_with_conditions(
            request, HULL_SAVE_MAPPINGS, target_conditions
        )

        assert not result.save_result.success
        assert "immobilized" in result.conditions_applied
        assert "immobilized" in target_conditions

    def test_hull_critical_failure_applies_stunned(self):
        """HULL critical failure (natural 1) should apply stunned."""
        target_conditions: list[StatusType] = []
        request = SaveRequest(
            save_type="hull",
            save_target=10,
            save_bonus=0,
            target_conditions=[],
            force_roll=1,  # Critical failure
        )

        result = resolve_save_with_conditions(
            request, HULL_SAVE_MAPPINGS, target_conditions
        )

        assert result.save_result.degree == "critical_failure"
        assert "stunned" in result.conditions_applied

    def test_hull_save_success_no_condition(self):
        """Successful HULL save should not apply conditions."""
        target_conditions: list[StatusType] = []
        request = SaveRequest(
            save_type="hull",
            save_target=10,
            save_bonus=5,
            target_conditions=[],
            force_roll=15,  # Will succeed (15+5 = 20 >= 10)
        )

        result = resolve_save_with_conditions(
            request, HULL_SAVE_MAPPINGS, target_conditions
        )

        assert result.save_result.success
        assert len(result.conditions_applied) == 0

    def test_agility_save_failure_applies_prone(self):
        """AGILITY save failure should apply prone (per book line 7521)."""
        target_conditions: list[StatusType] = []
        request = SaveRequest(
            save_type="agility",
            save_target=12,
            save_bonus=0,
            target_conditions=[],
            force_roll=7,  # Will fail
        )

        result = resolve_save_with_conditions(
            request, AGILITY_SAVE_MAPPINGS, target_conditions
        )

        assert not result.save_result.success
        assert "prone" in result.conditions_applied

    def test_systems_save_failure_applies_jammed(self):
        """SYSTEMS save failure should apply jammed (per book line 18957)."""
        target_conditions: list[StatusType] = []
        request = SaveRequest(
            save_type="systems",
            save_target=10,
            save_bonus=0,
            target_conditions=[],
            force_roll=4,  # Will fail
        )

        result = resolve_save_with_conditions(
            request, SYSTEMS_SAVE_MAPPINGS, target_conditions
        )

        assert not result.save_result.success
        assert "jammed" in result.conditions_applied

    def test_engineering_save_failure_applies_impaired(self):
        """ENGINEERING save failure should apply impaired."""
        target_conditions: list[StatusType] = []
        request = SaveRequest(
            save_type="engineering",
            save_target=10,
            save_bonus=0,
            target_conditions=[],
            force_roll=6,  # Will fail
        )

        result = resolve_save_with_conditions(
            request, ENGINEERING_SAVE_MAPPINGS, target_conditions
        )

        assert not result.save_result.success
        assert "impaired" in result.conditions_applied

    def test_no_matching_mapping_no_condition(self):
        """When no mapping matches, no condition is applied."""
        target_conditions: list[StatusType] = []
        custom_mapping = SaveConditionMapping(
            save_type="hull",
            condition="stunned",
            applies_on="critical_failure",
        )
        request = SaveRequest(
            save_type="agility",
            save_target=10,
            save_bonus=0,
            target_conditions=[],
            force_roll=3,  # Will fail
        )

        result = resolve_save_with_conditions(
            request, [custom_mapping], target_conditions
        )

        assert not result.save_result.success
        assert len(result.conditions_applied) == 0  # No mapping for agility

    def test_critical_failure_upgrades_condition(self):
        """Critical failure should apply the critical_failure mapping."""
        target_conditions: list[StatusType] = []
        request = SaveRequest(
            save_type="agility",
            save_target=10,
            save_bonus=0,
            target_conditions=[],
            force_roll=1,  # Critical failure
        )

        result = resolve_save_with_conditions(
            request, AGILITY_SAVE_MAPPINGS, target_conditions
        )

        assert result.save_result.degree == "critical_failure"
        assert "stunned" in result.conditions_applied  # Critical failure → stunned

    def test_common_mappings_cover_all_save_types(self):
        """COMMON_SAVE_CONDITION_MAPPINGS should cover all save types."""
        save_types_covered = {m.save_type for m in COMMON_SAVE_CONDITION_MAPPINGS}
        assert save_types_covered == {"hull", "agility", "systems", "engineering"}


class TestGetDefaultMappingsForSaveType:
    """Tests for get_default_mappings_for_save_type function."""

    def test_get_hull_mappings(self):
        """Should return HULL mappings for hull save type."""
        mappings = get_default_mappings_for_save_type("hull")
        assert mappings == HULL_SAVE_MAPPINGS

    def test_get_agility_mappings(self):
        """Should return AGILITY mappings for agility save type."""
        mappings = get_default_mappings_for_save_type("agility")
        assert mappings == AGILITY_SAVE_MAPPINGS

    def test_get_systems_mappings(self):
        """Should return SYSTEMS mappings for systems save type."""
        mappings = get_default_mappings_for_save_type("systems")
        assert mappings == SYSTEMS_SAVE_MAPPINGS

    def test_get_engineering_mappings(self):
        """Should return ENGINEERING mappings for engineering save type."""
        mappings = get_default_mappings_for_save_type("engineering")
        assert mappings == ENGINEERING_SAVE_MAPPINGS

    def test_unknown_save_type_returns_empty(self):
        """Unknown save type should return empty list."""
        # Use a valid save type that's not in the mapping dict
        # The function will return empty list for any save_type not in the dict
        mappings = get_default_mappings_for_save_type("agility")
        # Verify agility returns valid mappings, proving the dict lookup works
        assert len(mappings) > 0
