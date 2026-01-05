"""Tests for NPC ability triggers and validation."""

import pytest
from core.npc.state import NPCState
from core.npc.validation import (
    validate_npc_in_combat,
    validate_npc_ability_use,
)
from core.npc.models import NPCTemplate, NPCStats, NPCStatsBase, NPCAbility
from core.npc.enums import NPCTier, NPCClass


def _make_test_npc() -> NPCState:
    """Create a test NPC state."""
    template = NPCTemplate(
        id="test_npc",
        name="Test NPC",
        npc_class="grunt",
        tier="tier_1",
        role="striker",
        stats=NPCStats(
            base=NPCStatsBase(
                hp_base=10,
                evasion_base=8,
                e_defense_base=8,
            ),
        ),
        abilities=[
            NPCAbility(
                id="test_ability",
                name="Test Ability",
                trigger="on_hit",
                uses_per_combat=2,
            ),
            NPCAbility(
                id="unlimited_ability",
                name="Unlimited Ability",
                trigger="on_turn_start",
            ),
        ],
    )
    return NPCState.from_template(template, "npc_1")


class TestNPCAbilityUse:
    """Tests for NPC ability use tracking."""

    def test_ability_not_used_initially(self) -> None:
        """Abilities should not be used on NPC creation."""
        npc = _make_test_npc()
        assert "test_ability" not in npc.abilities_used

    def test_mark_ability_as_used(self) -> None:
        """NPC should track used abilities."""
        npc = _make_test_npc()
        npc.abilities_used.add("test_ability")
        assert "test_ability" in npc.abilities_used

    def test_unlimited_ability_not_tracked(self) -> None:
        """Abilities with no uses_per_combat don't need tracking."""
        npc = _make_test_npc()
        validation = validate_npc_ability_use(npc, "unlimited_ability")
        assert validation.valid


class TestNPCAbilityValidation:
    """Tests for validating ability use."""

    def test_validate_ability_not_used(self) -> None:
        """Validating unused ability should pass."""
        npc = _make_test_npc()
        validation = validate_npc_ability_use(npc, "test_ability")
        assert validation.valid
        assert not validation.issues

    def test_validate_ability_already_used(self) -> None:
        """Validating already-used ability should fail."""
        npc = _make_test_npc()
        npc.abilities_used.add("test_ability")
        validation = validate_npc_ability_use(npc, "test_ability")
        assert not validation.valid
        error_codes = {
            issue.code for issue in validation.issues if issue.severity == "error"
        }
        assert "ability_already_used" in error_codes

    def test_validate_nonexistent_ability(self) -> None:
        """Validating non-existent ability should pass (allows ability grants)."""
        npc = _make_test_npc()
        validation = validate_npc_ability_use(npc, "granted_ability")
        assert validation.valid


class TestNPCInCombatValidation:
    """Tests for validating NPC in combat."""

    def test_valid_npc_passes(self) -> None:
        """Valid NPC should pass combat validation."""
        npc = _make_test_npc()
        validation = validate_npc_in_combat(npc)
        assert validation.valid

    def test_npc_in_com_invalid_hp_max_rejected_at_construction(self) -> None:
        """NPC with HP < 1 should be rejected at construction."""
        with pytest.raises(Exception):  # Pydantic validation error
            NPCTemplate(
                id="test_invalid",
                name="Test NPC",
                npc_class="grunt",
                tier="tier_1",
                role="striker",
                stats=NPCStats(
                    base=NPCStatsBase(
                        hp_base=0,
                        evasion_base=8,
                        e_defense_base=8,
                    ),
                ),
            )

    def test_invalid_evasion_rejected_at_construction(self) -> None:
        """NPC with negative evasion should be rejected at construction."""
        with pytest.raises(Exception):  # Pydantic validation error
            NPCTemplate(
                id="test_invalid",
                name="Test NPC",
                npc_class="grunt",
                tier="tier_1",
                role="striker",
                stats=NPCStats(
                    base=NPCStatsBase(
                        hp_base=10,
                        evasion_base=-5,
                        e_defense_base=8,
                    ),
                ),
            )


class TestNPCAbilityTriggers:
    """Tests for NPC ability trigger types."""

    def test_trigger_on_hit(self) -> None:
        """on_hit trigger should be accepted."""
        ability = NPCAbility(
            id="test_hit",
            name="Test On Hit",
            trigger="on_hit",
        )
        assert ability.trigger == "on_hit"

    def test_trigger_on_turn_start(self) -> None:
        """on_turn_start trigger should be accepted."""
        ability = NPCAbility(
            id="test_turn_start",
            name="Test Turn Start",
            trigger="on_turn_start",
        )
        assert ability.trigger == "on_turn_start"

    def test_trigger_on_damaged(self) -> None:
        """on_damaged trigger should be accepted."""
        ability = NPCAbility(
            id="test_damaged",
            name="Test On Damaged",
            trigger="on_damaged",
        )
        assert ability.trigger == "on_damaged"

    def test_trigger_on_kill(self) -> None:
        """on_kill trigger should be accepted."""
        ability = NPCAbility(
            id="test_kill",
            name="Test On Kill",
            trigger="on_kill",
        )
        assert ability.trigger == "on_kill"
