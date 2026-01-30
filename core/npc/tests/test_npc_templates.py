"""Tests for NPC template validation."""

import pytest
from core.npc.compendium import (
    NPC_TEMPLATES,
    NPC_TEMPLATES_BY_ID,
    get_npc_template,
    get_templates_by_class,
    get_templates_by_tier,
)
from core.npc.validation import validate_npc_template, batch_validate_templates


def _assert_no_errors(validation) -> None:
    """Assert that validation has no error-level issues."""
    errors = [issue for issue in validation.issues if issue.severity == "error"]
    assert not errors, f"Validation errors: {errors}"


class TestNPCTemplateCompendium:
    """Tests for NPC template compendium."""

    def test_all_templates_loadable(self) -> None:
        """All templates in compendium should load without errors."""
        assert len(NPC_TEMPLATES) > 0
        assert len(NPC_TEMPLATES_BY_ID) == len(NPC_TEMPLATES)

    def test_template_ids_unique(self) -> None:
        """All template IDs should be unique."""
        ids = [t.id for t in NPC_TEMPLATES]
        assert len(ids) == len(set(ids))

    def test_get_template_by_id(self) -> None:
        """get_npc_template should return correct template."""
        template = get_npc_template("gms_grunt_t1")
        assert template is not None
        assert template.id == "gms_grunt_t1"
        assert template.name == "GMS Grunt"

    def test_get_template_not_found(self) -> None:
        """get_npc_template should return None for unknown ID."""
        template = get_npc_template("nonexistent_npc")
        assert template is None

    def test_filter_by_class(self) -> None:
        """get_templates_by_class should filter correctly."""
        grunts = get_templates_by_class("grunt")
        assert len(grunts) > 0
        for g in grunts:
            assert g.npc_class == "grunt"

    def test_filter_by_tier(self) -> None:
        """get_templates_by_tier should filter correctly."""
        tier1 = get_templates_by_tier("tier_1")
        assert len(tier1) > 0
        for t in tier1:
            assert t.tier == "tier_1"

    def test_all_classes_represented(self) -> None:
        """All four NPC classes should have at least one template."""
        classes_present = set(t.npc_class for t in NPC_TEMPLATES)
        expected_classes = {"grunt", "elite", "boss", "specialist"}
        assert classes_present == expected_classes, (
            f"Missing classes: {expected_classes - classes_present}"
        )


class TestNPCTemplateValidation:
    """Tests for NPC template validation."""

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_template_valid(self, template) -> None:
        """All compendium templates should pass validation."""
        validation = validate_npc_template(template)
        _assert_no_errors(validation)

    def test_batch_validation_all_pass(self) -> None:
        """All templates should pass batch validation."""
        results = batch_validate_templates(NPC_TEMPLATES)
        for template_id, validation in results.items():
            _assert_no_errors(validation)

    def test_duplicate_ability_ids_flagged(self) -> None:
        """Template with duplicate ability IDs should fail validation."""
        from core.npc.models import NPCTemplate, NPCStats, NPCStatsBase, NPCAbility

        bad_template = NPCTemplate(
            id="test_duplicate",
            name="Test NPC",
            npc_class="grunt",
            role="striker",
            stats=NPCStats(base=NPCStatsBase()),
            abilities=[
                NPCAbility(id="same_id", name="Ability 1", trigger="on_hit"),
                NPCAbility(id="same_id", name="Ability 2", trigger="on_hit"),
            ],
        )
        validation = validate_npc_template(bad_template)
        error_codes = {
            issue.code for issue in validation.issues if issue.severity == "error"
        }
        assert "duplicate_ability_ids" in error_codes

    def test_invalid_ability_uses_flagged(self) -> None:
        """Template with invalid ability uses should fail validation."""
        from core.npc.models import NPCTemplate, NPCStats, NPCStatsBase, NPCAbility

        bad_template = NPCTemplate(
            id="test_invalid_uses",
            name="Test NPC",
            npc_class="grunt",
            role="striker",
            stats=NPCStats(base=NPCStatsBase()),
            abilities=[
                NPCAbility(
                    id="bad_ability",
                    name="Bad Ability",
                    trigger="on_hit",
                    uses_per_combat=-1,
                ),
            ],
        )
        validation = validate_npc_template(bad_template)
        error_codes = {
            issue.code for issue in validation.issues if issue.severity == "error"
        }
        assert "invalid_ability_uses" in error_codes

    def test_too_much_gear_warns(self) -> None:
        """Template with too much gear should warn."""
        from core.npc.models import NPCTemplate, NPCStats, NPCStatsBase, NPCGear

        gear_template = NPCTemplate(
            id="test_gear",
            name="Test NPC",
            npc_class="elite",
            role="defender",
            stats=NPCStats(base=NPCStatsBase()),
            gear=[NPCGear(weapon_id=f"weapon_{i}") for i in range(7)],
        )
        validation = validate_npc_template(gear_template)
        warning_codes = {
            issue.code for issue in validation.issues if issue.severity == "warning"
        }
        assert "too_much_gear" in warning_codes


class TestNPCTemplateStructure:
    """Tests for NPC template structure and completeness."""

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_template_has_required_fields(self, template) -> None:
        """All templates should have required fields populated."""
        assert template.id
        assert template.name
        assert template.npc_class in ("grunt", "elite", "boss", "specialist")
        assert template.tier in ("tier_1", "tier_2", "tier_3")
        assert template.role in ("striker", "defender", "controller", "supporter")
        assert template.stats is not None
        assert template.stats.base is not None

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_template_base_stats_valid(self, template) -> None:
        """All templates should have valid base stats."""
        base = template.stats.base
        assert base.hp_base >= 1
        assert base.evasion_base >= 0
        assert base.e_defense_base >= 0
        assert base.speed_base >= 0

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_template_has_gear_or_abilities(self, template) -> None:
        """Templates should have gear or abilities (except bosses)."""
        has_gear = len(template.gear) > 0
        has_abilities = len(template.abilities) > 0
        if template.npc_class != "boss":
            assert has_gear or has_abilities, (
                f"Template {template.id} has no gear or abilities"
            )


class TestNPCTemplateRoles:
    """Tests for NPC template role assignments."""

    def test_striker_templates_exist(self) -> None:
        """Should have striker templates."""
        from core.npc.templates import get_striker_templates

        strikers = get_striker_templates()
        assert len(strikers) > 0

    def test_defender_templates_exist(self) -> None:
        """Should have defender templates."""
        from core.npc.templates import get_defender_templates

        defenders = get_defender_templates()
        assert len(defenders) > 0

    def test_controller_templates_exist(self) -> None:
        """Should have controller templates."""
        from core.npc.templates import get_controller_templates

        controllers = get_controller_templates()
        assert len(controllers) > 0

    def test_supporter_templates_exist(self) -> None:
        """Should have supporter templates."""
        from core.npc.templates import get_supporter_templates

        supporters = get_supporter_templates()
        assert len(supporters) > 0

    @pytest.mark.parametrize("template", NPC_TEMPLATES)
    def test_all_templates_have_role(self, template) -> None:
        """All templates should have a role assigned."""
        assert template.role in ("striker", "defender", "controller", "supporter")
