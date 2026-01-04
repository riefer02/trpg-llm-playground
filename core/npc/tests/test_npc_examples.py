"""Tests for NPC example builds and scenarios."""

import pytest
from core.npc.examples import (
    create_npc_by_id,
    create_gms_grunt_squad,
    create_ipsn_raid_party,
    create_horus_encounter,
    create_boss_encounter,
    evaluate_gms_grunt_example,
    evaluate_ipsn_boss_example,
    evaluate_all_npc_templates,
)
from core.npc.compendium import NPC_TEMPLATES
from core.npc.validation import validate_npc_in_combat


def _assert_no_errors(validation) -> None:
    """Assert that validation has no error-level issues."""
    errors = [issue for issue in validation.issues if issue.severity == "error"]
    assert not errors, f"Validation errors: {errors}"


class TestCreateNPCById:
    """Tests for creating NPCs by template ID."""

    def test_create_valid_npc(self) -> None:
        """Creating NPC from valid template should work."""
        npc = create_npc_by_id("gms_grunt_t1", "test_1")
        assert npc.id == "test_1"
        assert npc.name == "GMS Grunt"
        assert npc.npc_class == "grunt"
        assert npc.tier == "tier_1"

    def test_create_with_custom_name(self) -> None:
        """Creating NPC with custom name should work."""
        npc = create_npc_by_id("gms_grunt_t1", "test_1", "Custom Name")
        assert npc.name == "Custom Name"

    def test_create_invalid_template_raises(self) -> None:
        """Creating NPC from invalid template should raise ValueError."""
        with pytest.raises(ValueError):
            create_npc_by_id("nonexistent_template", "test_1")


class TestNPCExampleSquads:
    """Tests for pre-built NPC squads."""

    def test_gms_grunt_squad(self) -> None:
        """GMS grunt squad should have 4 NPCs."""
        squad = create_gms_grunt_squad()
        assert len(squad) == 4
        for i, npc in enumerate(squad):
            assert npc.npc_class == "grunt"
            assert npc.tier == "tier_1"
            validation = validate_npc_in_combat(npc)
            _assert_no_errors(validation)

    def test_ipsn_raid_party(self) -> None:
        """IPS-N raid party should have mixed NPCs."""
        party = create_ipsn_raid_party()
        assert len(party) == 3
        assert party[0].npc_class == "grunt"
        assert party[0].tier == "tier_1"
        assert party[2].npc_class == "boss"
        assert party[2].tier == "tier_3"
        for npc in party:
            validation = validate_npc_in_combat(npc)
            _assert_no_errors(validation)

    def test_horus_encounter(self) -> None:
        """HORUS encounter should have specialist and elite NPCs."""
        encounter = create_horus_encounter()
        assert len(encounter) == 2
        for npc in encounter:
            validation = validate_npc_in_combat(npc)
            _assert_no_errors(validation)

    def test_boss_encounter(self) -> None:
        """Boss encounter should have boss and elite NPCs."""
        encounter = create_boss_encounter()
        assert len(encounter) == 3
        assert encounter[0].npc_class == "boss"
        for npc in encounter:
            validation = validate_npc_in_combat(npc)
            _assert_no_errors(validation)


class TestEvaluateExamples:
    """Tests for example evaluation functions."""

    def test_evaluate_gms_grunt(self) -> None:
        """GMS grunt evaluation should return valid results."""
        result = evaluate_gms_grunt_example()
        assert result["npc_class"] == "grunt"
        assert result["tier"] == "tier_1"
        assert result["valid"]
        assert result["issues"] == 0

    def test_evaluate_ipsn_boss(self) -> None:
        """IPS-N boss evaluation should return valid results."""
        result = evaluate_ipsn_boss_example()
        assert result["npc_class"] == "boss"
        assert result["tier"] == "tier_3"
        assert result["structures"] == 3
        assert result["valid"]
        assert result["issues"] == 0

    def test_evaluate_all_templates(self) -> None:
        """All templates should pass evaluation."""
        result = evaluate_all_npc_templates()
        assert result["template_count"] == len(NPC_TEMPLATES)
        assert result["all_valid"]


class TestNPCTemplateVariety:
    """Tests for NPC template variety."""

    def test_all_classes_in_examples(self) -> None:
        """Example functions should cover all NPC classes."""
        classes_in_examples = set()
        classes_in_examples.update(n.npc_class for n in create_gms_grunt_squad())
        classes_in_examples.update(n.npc_class for n in create_ipsn_raid_party())
        classes_in_examples.update(n.npc_class for n in create_horus_encounter())
        classes_in_examples.update(n.npc_class for n in create_boss_encounter())
        expected_classes = {"grunt", "elite", "boss", "specialist"}
        assert classes_in_examples == expected_classes

    def test_all_tiers_in_examples(self) -> None:
        """Example functions should cover all tiers."""
        tiers_in_examples = set()
        tiers_in_examples.update(n.tier for n in create_gms_grunt_squad())
        tiers_in_examples.update(n.tier for n in create_ipsn_raid_party())
        tiers_in_examples.update(n.tier for n in create_horus_encounter())
        tiers_in_examples.update(n.tier for n in create_boss_encounter())
        expected_tiers = {"tier_1", "tier_2", "tier_3"}
        assert tiers_in_examples == expected_tiers
