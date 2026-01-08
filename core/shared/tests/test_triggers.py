"""Comprehensive tests for the unified trigger system.

Tests TriggerType definitions, trigger resolution, and NPC trigger integration.
"""

import pytest
from core.shared.effects import TriggerType
from core.shared.triggers import (
    TriggerContext,
    TriggerResolution,
    check_trigger_condition,
    resolve_trigger,
    is_valid_for_player,
    is_valid_for_npc,
    NPC_ONLY_TRIGGERS,
    check_hp_below_half,
    check_adjacent,
    check_damage_dealt,
)
from core.npc.models import NPCAbility
from core.shared.effects import MechanicalEffect


class TestTriggerTypeDefinitions:
    """Tests that all expected triggers are defined in TriggerType."""

    def test_universal_triggers_defined(self) -> None:
        """Universal triggers that work for all combatants."""
        universal_triggers: list[TriggerType] = [
            "on_hit",
            "on_miss",
            "on_crit",
            "on_kill",
            "on_turn_start",
            "on_turn_end",
            "on_take_damage",
            "on_attack_roll",
            "on_activation",
            "on_overheat",
            "on_structure_loss",
            "on_heat_gain",
            "on_move",
            "on_reaction",
            "on_action",
            "on_overcharge",
            "on_core_power_spent",
            "on_tech_attack_hit",
            "on_target_failed_save",
            "on_stabilize",
            "on_skirmish",
            "on_inflict",
        ]
        for trigger in universal_triggers:
            assert trigger in TriggerType.__args__, f"Missing trigger: {trigger}"

    def test_ally_triggers_defined(self) -> None:
        """Ally-related triggers."""
        ally_triggers: list[TriggerType] = [
            "on_ally_hit",
            "on_ally_miss",
            "on_ally_damaged",
            "on_ally_targeted",
            "on_ally_turn_start",
            "on_ally_hit_target_within_range",
            "on_ally_killed",
        ]
        for trigger in ally_triggers:
            assert trigger in TriggerType.__args__, f"Missing ally trigger: {trigger}"

    def test_npc_only_triggers_defined(self) -> None:
        """NPC-specific triggers that are now in unified TriggerType."""
        npc_triggers: list[TriggerType] = [
            "on_deploy",
            "on_destroyed",
            "on_adjacent",
            "on_attacked",
            "on_ally_killed",
            "on_hp_below_half",
            "on_damage_dealt",
        ]
        for trigger in npc_triggers:
            assert trigger in TriggerType.__args__, f"Missing NPC trigger: {trigger}"

    def test_complex_triggers_defined(self) -> None:
        """Complex triggers requiring additional context."""
        complex_triggers: list[TriggerType] = [
            "on_first_adjacent_turn",
            "on_any_damage",
            "on_first_attack",
            "on_overkill",
            "on_detonate",
            "on_brace",
            "on_move_or_boost",
            "on_slipstream_jump",
            "on_extra_overwatch",
            "on_lock_on_consumed",
            "on_hide",
            "on_enter",
            "on_reload",
        ]
        for trigger in complex_triggers:
            assert trigger in TriggerType.__args__, (
                f"Missing complex trigger: {trigger}"
            )

    def test_trigger_count(self) -> None:
        """Verify trigger count is reasonable."""
        trigger_count = len(TriggerType.__args__)
        assert trigger_count >= 45, (
            f"Expected at least 45 triggers, got {trigger_count}"
        )


class TestTriggerContext:
    """Tests for TriggerContext creation and properties."""

    def test_create_basic_context(self) -> None:
        """Test basic context creation."""
        ctx = TriggerContext(
            trigger_type="on_hit",
            source_id="npc_1",
            target_id="player_1",
        )
        assert ctx.trigger_type == "on_hit"
        assert ctx.source_id == "npc_1"
        assert ctx.target_id == "player_1"

    def test_create_full_context(self) -> None:
        """Test context with all fields."""
        ctx = TriggerContext(
            trigger_type="on_hp_below_half",
            source_id="npc_1",
            target_id=None,
            damage_dealt=5,
            hp_before=20,
            hp_after=8,
            is_critical=False,
            adjacent_allies=2,
            round_number=3,
        )
        assert ctx.trigger_type == "on_hp_below_half"
        assert ctx.hp_before == 20
        assert ctx.hp_after == 8
        assert ctx.adjacent_allies == 2

    def test_context_defaults(self) -> None:
        """Test context default values."""
        ctx = TriggerContext(trigger_type="on_turn_start")
        assert ctx.source_id is None
        assert ctx.target_id is None
        assert ctx.damage_dealt is None
        assert ctx.hp_before is None
        assert ctx.hp_after is None
        assert ctx.is_critical is False
        assert ctx.adjacent_allies == 0
        assert ctx.round_number is None


class TestTriggerConditionChecks:
    """Tests for trigger condition checking functions."""

    def test_hp_below_half_true(self) -> None:
        """HP below half should trigger."""
        assert check_hp_below_half(20, 9) is True
        assert check_hp_below_half(10, 4) is True
        assert check_hp_below_half(100, 49) is True

    def test_hp_below_half_false(self) -> None:
        """HP at or above half should not trigger."""
        assert check_hp_below_half(20, 10) is False
        assert check_hp_below_half(20, 15) is False
        assert check_hp_below_half(10, 5) is False

    def test_hp_below_half_edge_cases(self) -> None:
        """Edge cases for HP check."""
        assert check_hp_below_half(None, 5) is True  # Default threshold
        assert check_hp_below_half(20, None) is False
        assert check_hp_below_half(None, None) is False

    def test_adjacent_true(self) -> None:
        """Having adjacent allies should trigger."""
        assert check_adjacent(1) is True
        assert check_adjacent(3) is True
        assert check_adjacent(5) is True

    def test_adjacent_false(self) -> None:
        """No adjacent allies should not trigger."""
        assert check_adjacent(0) is False

    def test_damage_dealt_true(self) -> None:
        """Dealing damage should trigger."""
        assert check_damage_dealt(1) is True
        assert check_damage_dealt(10) is True
        assert check_damage_dealt(100) is True

    def test_damage_dealt_false(self) -> None:
        """No damage dealt should not trigger."""
        assert check_damage_dealt(0) is False
        assert check_damage_dealt(None) is False
        assert check_damage_dealt(-1) is False


class TestCheckTriggerCondition:
    """Tests for the main check_trigger_condition function."""

    def test_hp_below_half_condition(self) -> None:
        """Test on_hp_below_half condition check."""
        ctx = TriggerContext(
            trigger_type="on_hp_below_half",
            hp_before=20,
            hp_after=8,
        )
        assert check_trigger_condition("on_hp_below_half", ctx) is True

    def test_adjacent_condition(self) -> None:
        """Test on_adjacent condition check."""
        ctx = TriggerContext(
            trigger_type="on_adjacent",
            adjacent_allies=2,
        )
        assert check_trigger_condition("on_adjacent", ctx) is True

    def test_damage_dealt_condition(self) -> None:
        """Test on_damage_dealt condition check."""
        ctx = TriggerContext(
            trigger_type="on_damage_dealt",
            damage_dealt=5,
        )
        assert check_trigger_condition("on_damage_dealt", ctx) is True

    def test_no_condition_required(self) -> None:
        """Triggers without special conditions should return True."""
        ctx = TriggerContext(trigger_type="on_hit")
        assert check_trigger_condition("on_hit", ctx) is True
        assert check_trigger_condition("on_turn_start", ctx) is True
        # on_kill requires damage_dealt context, test it separately
        ctx_kill = TriggerContext(trigger_type="on_kill", damage_dealt=5)
        assert check_trigger_condition("on_kill", ctx_kill) is True


class TestIsValidForPlayer:
    """Tests for is_valid_for_player function."""

    def test_universal_triggers_valid(self) -> None:
        """Universal triggers should be valid for players."""
        for trigger in [
            "on_hit",
            "on_miss",
            "on_crit",
            "on_turn_start",
            "on_take_damage",
        ]:
            assert is_valid_for_player(trigger) is True, (
                f"{trigger} should be valid for players"
            )

    def test_npc_only_triggers_invalid(self) -> None:
        """NPC-only triggers should not be valid for players."""
        for trigger in NPC_ONLY_TRIGGERS:
            assert is_valid_for_player(trigger) is False, (
                f"{trigger} should not be valid for players"
            )

    def test_all_npc_only_triggers_defined(self) -> None:
        """Verify NPC_ONLY_TRIGGERS set is properly defined."""
        expected_npc_only = {
            "on_deploy",
            "on_destroyed",
            "on_adjacent",
            "on_ally_killed",
            "on_hp_below_half",
            "on_first_adjacent_turn",
        }
        assert NPC_ONLY_TRIGGERS == expected_npc_only


class TestIsValidForNPC:
    """Tests for is_valid_for_npc function."""

    def test_all_triggers_valid_for_npc(self) -> None:
        """All triggers should be valid for NPCs."""
        for trigger in TriggerType.__args__:
            assert is_valid_for_npc(trigger) is True, (
                f"{trigger} should be valid for NPCs"
            )


class TestResolveTrigger:
    """Tests for the main resolve_trigger function."""

    def test_resolve_hit_for_player(self) -> None:
        """Player hit trigger should resolve successfully."""
        ctx = TriggerContext(trigger_type="on_hit")
        result = resolve_trigger("on_hit", ctx, is_npc=False)
        assert result.triggered is True

    def test_resolve_npc_only_for_player_fails(self) -> None:
        """NPC-only trigger for player should not trigger."""
        ctx = TriggerContext(trigger_type="on_deploy")
        result = resolve_trigger("on_deploy", ctx, is_npc=False)
        assert result.triggered is False
        assert "not valid for player" in result.message

    def test_resolve_npc_only_for_npc_succeeds(self) -> None:
        """NPC-only trigger for NPC should resolve."""
        ctx = TriggerContext(trigger_type="on_deploy")
        result = resolve_trigger("on_deploy", ctx, is_npc=True)
        assert result.triggered is True

    def test_resolve_hp_below_half_condition_met(self) -> None:
        """HP below half trigger should fire when condition is met."""
        ctx = TriggerContext(
            trigger_type="on_hp_below_half",
            hp_before=20,
            hp_after=8,
        )
        result = resolve_trigger("on_hp_below_half", ctx, is_npc=True)
        assert result.triggered is True

    def test_resolve_hp_below_half_condition_not_met(self) -> None:
        """HP below half trigger should not fire when condition is not met."""
        ctx = TriggerContext(
            trigger_type="on_hp_below_half",
            hp_before=20,
            hp_after=15,
        )
        result = resolve_trigger("on_hp_below_half", ctx, is_npc=True)
        assert result.triggered is False
        assert "condition not met" in result.message


class TestNPCAbilityWithTriggerType:
    """Tests that NPC abilities can use the unified TriggerType."""

    def test_npc_ability_with_universal_trigger(self) -> None:
        """NPC ability with universal trigger should work."""
        ability = NPCAbility(
            id="test_ability",
            name="Test Ability",
            trigger="on_hit",
        )
        assert ability.trigger == "on_hit"

    def test_npc_ability_with_npc_trigger(self) -> None:
        """NPC ability with NPC-specific trigger should work."""
        ability = NPCAbility(
            id="test_ability",
            name="Test Ability",
            trigger="on_deploy",
        )
        assert ability.trigger == "on_deploy"

    def test_npc_ability_with_hp_trigger(self) -> None:
        """NPC ability with HP trigger should work."""
        ability = NPCAbility(
            id="test_ability",
            name="Test Ability",
            trigger="on_hp_below_half",
        )
        assert ability.trigger == "on_hp_below_half"

    def test_npc_ability_with_damage_trigger(self) -> None:
        """NPC ability with damage trigger should work."""
        ability = NPCAbility(
            id="test_ability",
            name="Test Ability",
            trigger="on_damage_dealt",
        )
        assert ability.trigger == "on_damage_dealt"

    def test_npc_ability_with_effect(self) -> None:
        """NPC ability with mechanical effect should work."""
        ability = NPCAbility(
            id="test_ability",
            name="Test Ability",
            trigger="on_hit",
            effect=MechanicalEffect(
                direct_damages=[
                    {
                        "damage_type": "kinetic",
                        "flat": 2,
                    }
                ]
            ),
        )
        assert ability.trigger == "on_hit"
        assert ability.effect.direct_damages is not None


class TestTriggerResolutionClass:
    """Tests for TriggerResolution class."""

    def test_resolution_defaults(self) -> None:
        """Test TriggerResolution default values."""
        resolution = TriggerResolution(triggered=True)
        assert resolution.triggered is True
        assert resolution.effects == []
        assert resolution.message == ""

    def test_resolution_with_message(self) -> None:
        """Test TriggerResolution with custom message."""
        resolution = TriggerResolution(
            triggered=True,
            message="Trigger resolved successfully",
        )
        assert resolution.message == "Trigger resolved successfully"

    def test_resolution_with_effects(self) -> None:
        """Test TriggerResolution with effects."""
        from core.shared.effects import StatusGrant

        effect = MechanicalEffect(
            status_grants=[StatusGrant(status="impaired", target="enemy")]
        )
        resolution = TriggerResolution(
            triggered=True,
            effects=[effect],
        )
        assert len(resolution.effects) == 1


class TestTriggerTypeExhaustive:
    """Exhaustive tests to ensure all TriggerType values work."""

    @pytest.mark.parametrize("trigger", TriggerType.__args__)
    def test_all_triggers_resolve_for_npc(self, trigger: TriggerType) -> None:
        """All triggers should be valid for NPCs."""
        ctx = TriggerContext(trigger_type=trigger)
        result = resolve_trigger(trigger, ctx, is_npc=True)
        # Complex triggers might fail condition check, but should be valid
        assert result.triggered or "condition" in result.message.lower()

    @pytest.mark.parametrize("trigger", TriggerType.__args__)
    def test_universal_triggers_resolve_for_player(self, trigger: TriggerType) -> None:
        """Universal triggers should work for players."""
        if trigger in NPC_ONLY_TRIGGERS:
            return  # Skip NPC-only triggers
        ctx = TriggerContext(trigger_type=trigger)
        result = resolve_trigger(trigger, ctx, is_npc=False)
        assert result.triggered is True or "condition" in result.message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
