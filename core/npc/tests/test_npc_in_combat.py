"""Tests for NPC integration with combat state."""

import pytest
from core.npc.compendium import (
    get_npc_template,
    NPC_TEMPLATES,
)
from core.npc.state import (
    NPCState,
    NPCCombatStats,
    scale_npc_stats,
)
from core.npc.models import NPCTemplate, NPCStats, NPCStatsBase, NPCTierScaling
from core.npc.validation import validate_npc_template, validate_npc_in_combat
from core.npc.enums import NPCTier, NPCClass
from core.shared.enums import SizeClass


def _assert_no_errors(validation) -> None:
    """Assert that validation has no error-level issues."""
    errors = [issue for issue in validation.issues if issue.severity == "error"]
    assert not errors, f"Validation errors: {errors}"


class TestNPCInCombatIntegration:
    """Tests for NPC integration with combat system."""

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_npc_template_valid_in_combat(self, template) -> None:
        """All templates should create valid combat NPCs."""
        npc = NPCState.from_template(template, f"test_{template.id}")
        validation = validate_npc_in_combat(npc)
        _assert_no_errors(validation)

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_npc_tier_structure_matches(self, template) -> None:
        """NPC structure should match tier expectations."""
        npc = NPCState.from_template(template, f"test_{template.id}")
        expected = {"tier_1": 1, "tier_2": 2, "tier_3": 3}
        assert npc.structure_current == expected.get(template.tier, 1)

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_npc_hp_at_least_1(self, template) -> None:
        """NPC should have at least 1 HP."""
        npc = NPCState.from_template(template, f"test_{template.id}")
        assert npc.stats.hp_max >= 1
        assert npc.hp_current >= 1

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_npc_evasion_nonnegative(self, template) -> None:
        """NPC evasion should be non-negative."""
        npc = NPCState.from_template(template, f"test_{template.id}")
        assert npc.stats.evasion >= 0

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_npc_e_defense_nonnegative(self, template) -> None:
        """NPC e-defense should be non-negative."""
        npc = NPCState.from_template(template, f"test_{template.id}")
        assert npc.stats.e_defense >= 0

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_npc_size_valid(self, template) -> None:
        """NPC size should be a valid SizeClass."""
        npc = NPCState.from_template(template, f"test_{template.id}")
        assert npc.stats.size in (
            "size_half",
            "size_1",
            "size_2",
            "size_3",
            "size_4",
            "size_5",
        )

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_npc_abilities_tracked(self, template) -> None:
        """NPC should track abilities used."""
        npc = NPCState.from_template(template, f"test_{template.id}")
        assert npc.abilities_used is not None
        assert isinstance(npc.abilities_used, set)

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_npc_has_all_class_attributes(self, template) -> None:
        """NPC should have all class attributes."""
        npc = NPCState.from_template(template, f"test_{template.id}")
        assert hasattr(npc, "id")
        assert hasattr(npc, "name")
        assert hasattr(npc, "npc_class")
        assert hasattr(npc, "tier")
        assert hasattr(npc, "stats")
        assert hasattr(npc, "abilities_used")


class TestNPCTemplateToState:
    """Tests for converting templates to combat state."""

    def test_unique_instance_ids(self) -> None:
        """Multiple NPCs from same template should have unique IDs."""
        template = get_npc_template("gms_grunt_t1")
        assert template is not None
        npc1 = NPCState.from_template(template, "npc_1")
        npc2 = NPCState.from_template(template, "npc_2")
        npc3 = NPCState.from_template(template, "npc_3")
        assert npc1.id == "npc_1"
        assert npc2.id == "npc_2"
        assert npc3.id == "npc_3"

    def test_custom_name_override(self) -> None:
        """Custom name should override template name."""
        template = get_npc_template("gms_grunt_t1")
        assert template is not None
        npc = NPCState.from_template(template, "npc_1", name="Custom Name")
        assert npc.name == "Custom Name"

    def test_default_name_from_template(self) -> None:
        """Default name should come from template."""
        template = get_npc_template("gms_grunt_t1")
        assert template is not None
        npc = NPCState.from_template(template, "npc_1")
        assert npc.name == template.name


class TestNPCTierScalingInCombat:
    """Tests for tier scaling in combat context."""

    def test_tier_1_combat_stats(self) -> None:
        """Tier 1 should have baseline stats."""
        template = NPCTemplate(
            id="test_t1",
            name="Test T1",
            npc_class="grunt",
            tier="tier_1",
            stats=NPCStats(
                base=NPCStatsBase(
                    hp_base=10,
                    evasion_base=8,
                    e_defense_base=8,
                ),
            ),
        )
        npc = NPCState.from_template(template, "npc_1")
        assert npc.stats.hp_max == 10
        assert npc.stats.evasion == 8
        assert npc.stats.e_defense == 8

    def test_tier_2_combat_stats_higher(self) -> None:
        """Tier 2 should have higher stats."""
        template = NPCTemplate(
            id="test_t2",
            name="Test T2",
            npc_class="grunt",
            tier="tier_2",
            stats=NPCStats(
                base=NPCStatsBase(
                    hp_base=10,
                    evasion_base=8,
                    e_defense_base=8,
                ),
            ),
        )
        npc = NPCState.from_template(template, "npc_1")
        assert npc.stats.hp_max >= 10
        assert npc.stats.evasion >= 8
        assert npc.stats.e_defense >= 8

    def test_tier_3_combat_stats_highest(self) -> None:
        """Tier 3 should have highest stats with custom scaling."""
        scaling = NPCTierScaling(
            hp_multiplier=1.5,
            hp_adder_tier_3=20,
        )
        template = NPCTemplate(
            id="test_t3",
            name="Test T3",
            npc_class="boss",
            tier="tier_3",
            stats=NPCStats(
                base=NPCStatsBase(
                    hp_base=10,
                    evasion_base=8,
                    e_defense_base=8,
                ),
                scaling=scaling,
            ),
        )
        npc = NPCState.from_template(template, "npc_1")
        assert npc.stats.hp_max == 35  # 10 * 1.5 + 20
        assert npc.stats.evasion >= 10
        assert npc.stats.e_defense >= 10


class TestCombatReadiness:
    """Tests for NPC combat readiness."""

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_all_npcs_combat_ready(self, template) -> None:
        """All NPCs from compendium should be combat ready."""
        npc = NPCState.from_template(template, f"combat_{template.id}")
        template_validation = validate_npc_template(template)
        combat_validation = validate_npc_in_combat(npc)
        assert template_validation.valid, (
            f"Template {template.id} invalid: {template_validation.issues}"
        )
        assert combat_validation.valid, (
            f"NPC {npc.id} invalid: {combat_validation.issues}"
        )
