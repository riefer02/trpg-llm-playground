"""Tests for turn end resolution system."""

from __future__ import annotations

from typing import Literal
import pytest
from core.shared.models import FrozenModel
from core.shared.turn_end import (
    TurnEndEffectState,
    TurnEndTrigger,
    TurnEndInput,
    TurnEndResult,
    TurnEndTriggerResult,
    TurnEndTriggerSource,
    resolve_turn_end_triggers,
    expire_end_of_turn_effects,
    advance_end_of_next_turn_effects,
    resolve_turn_end,
    create_end_of_turn_effect,
    create_end_of_turn_trigger,
    has_active_turn_end_effect,
    get_active_effects_by_type,
    apply_turn_end_effect_to_state,
)
from core.shared.effects import EffectDuration


class TestTurnEndEffectState:
    """Tests for TurnEndEffectState model."""

    def test_create_buff_effect(self):
        """Test creating a buff type turn-end effect."""
        effect = TurnEndEffectState(
            effect_id="test_buff",
            effect_type="buff",
            target_id="mech1",
            duration_type="end_of_next_turn",
            applied_by="mech1",
            effect_data={"buff_type": "accuracy", "value": 1},
        )

        assert effect.effect_id == "test_buff"
        assert effect.effect_type == "buff"
        assert effect.target_id == "mech1"
        assert effect.duration_type == "end_of_next_turn"

    def test_create_disengage_effect(self):
        """Test creating a disengage type turn-end effect."""
        effect = TurnEndEffectState(
            effect_id="disengage_1",
            effect_type="disengage",
            target_id="mech1",
            duration_type="end_of_turn",
            applied_by="mech1",
            effect_data={},
        )

        assert effect.effect_type == "disengage"
        assert effect.duration_type == "end_of_turn"

    def test_create_cover_grant_effect(self):
        """Test creating a cover grant type turn-end effect."""
        effect = TurnEndEffectState(
            effect_id="cover_1",
            effect_type="cover_grant",
            target_id="ally1",
            duration_type="end_of_next_turn",
            applied_by="mech1",
            effect_data={"cover_type": "soft"},
        )

        assert effect.effect_type == "cover_grant"
        assert effect.effect_data["cover_type"] == "soft"


class TestTurnEndTrigger:
    """Tests for TurnEndTrigger model."""

    def test_create_talent_trigger(self):
        """Test creating a talent-based turn end trigger."""
        trigger = TurnEndTrigger(
            trigger_id="skirmisher_1",
            trigger_name="Skirmisher: Move & Shoot",
            source_type="talent",
            actor_id="mech1",
            effect_data={"type": "buff", "buff_type": "accuracy", "value": 1},
            order_priority=10,
        )

        assert trigger.trigger_id == "skirmisher_1"
        assert trigger.source_type == "talent"
        assert trigger.actor_id == "mech1"

    def test_create_system_trigger(self):
        """Test creating a system-based turn end trigger."""
        trigger = TurnEndTrigger(
            trigger_id="displacer_1",
            trigger_name="Displacer Field",
            source_type="system",
            actor_id="mech1",
            effect_data={"type": "cover_grant", "cover_type": "soft"},
        )

        assert trigger.source_type == "system"


class TestResolveTurnEndTriggers:
    """Tests for resolve_turn_end_triggers function."""

    def test_single_buff_trigger(self):
        """Test resolving a single buff trigger."""
        input_data = TurnEndInput(
            actor_id="mech1",
            triggers=[
                TurnEndTrigger(
                    trigger_id="skirmisher",
                    trigger_name="Skirmisher",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "accuracy", "value": 1},
                )
            ],
        )

        results, new_effects = resolve_turn_end_triggers(input_data)

        assert len(results) == 1
        assert results[0].triggered
        assert "accuracy" in results[0].effect_summary
        assert len(new_effects) == 1
        assert "skirmisher:buff" in new_effects

    def test_multiple_triggers_default_order(self):
        """Test multiple triggers resolve in default priority order."""
        input_data = TurnEndInput(
            actor_id="mech1",
            triggers=[
                TurnEndTrigger(
                    trigger_id="talent_b",
                    trigger_name="Talent B",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "evasion", "value": 2},
                    order_priority=20,
                ),
                TurnEndTrigger(
                    trigger_id="talent_a",
                    trigger_name="Talent A",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "accuracy", "value": 1},
                    order_priority=10,
                ),
            ],
        )

        results, _ = resolve_turn_end_triggers(input_data)

        assert len(results) == 2
        assert results[0].trigger_id == "talent_a"
        assert results[1].trigger_id == "talent_b"

    def test_multiple_triggers_actor_specified_order(self):
        """Test multiple triggers resolve in actor-specified order."""
        input_data = TurnEndInput(
            actor_id="mech1",
            triggers=[
                TurnEndTrigger(
                    trigger_id="talent_a",
                    trigger_name="Talent A",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "accuracy", "value": 1},
                    order_priority=10,
                ),
                TurnEndTrigger(
                    trigger_id="talent_b",
                    trigger_name="Talent B",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "evasion", "value": 2},
                    order_priority=20,
                ),
            ],
            specified_order=["talent_b", "talent_a"],
        )

        results, _ = resolve_turn_end_triggers(input_data)

        assert results[0].trigger_id == "talent_b"
        assert results[1].trigger_id == "talent_a"

    def test_condition_trigger(self):
        """Test resolving a condition trigger."""
        input_data = TurnEndInput(
            actor_id="mech1",
            triggers=[
                TurnEndTrigger(
                    trigger_id="system_immobilize",
                    trigger_name="Immobilizing System",
                    source_type="system",
                    actor_id="mech1",
                    effect_data={"type": "condition", "condition": "immobilized"},
                )
            ],
        )

        results, new_effects = resolve_turn_end_triggers(input_data)

        assert len(results) == 1
        assert "immobilized" in results[0].effect_summary
        effect = list(new_effects.values())[0]
        assert effect.effect_type == "condition"
        assert effect.effect_data["condition"] == "immobilized"

    def test_disengage_trigger(self):
        """Test resolving a disengage trigger."""
        input_data = TurnEndInput(
            actor_id="mech1",
            triggers=[
                TurnEndTrigger(
                    trigger_id="disengage",
                    trigger_name="Disengage",
                    source_type="custom",
                    actor_id="mech1",
                    effect_data={"type": "disengage"},
                )
            ],
        )

        results, new_effects = resolve_turn_end_triggers(input_data)

        assert len(results) == 1
        assert (
            "engagement" in results[0].effect_summary.lower()
            or "disengage" in results[0].effect_summary.lower()
        )
        assert len(new_effects) == 1
        effect = list(new_effects.values())[0]
        assert effect.effect_type == "disengage"
        assert effect.duration_type == "end_of_turn"


class TestExpireEndOfTurnEffects:
    """Tests for expire_end_of_turn_effects function."""

    def test_expire_end_of_turn_effects(self):
        """Test that end_of_turn effects are expired."""
        effects = {
            "disengage_1": TurnEndEffectState(
                effect_id="disengage_1",
                effect_type="disengage",
                target_id="mech1",
                duration_type="end_of_turn",
                applied_by="mech1",
            ),
            "buff_1": TurnEndEffectState(
                effect_id="buff_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
            ),
        }

        remaining, expired = expire_end_of_turn_effects(effects)

        assert "disengage_1" in expired
        assert "buff_1" not in expired
        assert "buff_1" in remaining
        assert len(remaining) == 1

    def test_all_effects_expire(self):
        """Test when all effects expire."""
        effects = {
            "effect_1": TurnEndEffectState(
                effect_id="effect_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_turn",
                applied_by="mech1",
            ),
            "effect_2": TurnEndEffectState(
                effect_id="effect_2",
                effect_type="cover_grant",
                target_id="mech1",
                duration_type="end_of_turn",
                applied_by="mech1",
            ),
        }

        remaining, expired = expire_end_of_turn_effects(effects)

        assert len(remaining) == 0
        assert len(expired) == 2

    def test_no_effects_expire(self):
        """Test when no effects expire."""
        effects = {
            "effect_1": TurnEndEffectState(
                effect_id="effect_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
            ),
        }

        remaining, expired = expire_end_of_turn_effects(effects)

        assert len(remaining) == 1
        assert len(expired) == 0


class TestAdvanceEndOfNextTurnEffects:
    """Tests for advance_end_of_next_turn_effects function."""

    def test_advance_end_of_next_turn_to_end_of_turn(self):
        """Test end_of_next_turn effects become end_of_turn."""
        effects = {
            "buff_1": TurnEndEffectState(
                effect_id="buff_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
            ),
        }

        updated, for_next_turn = advance_end_of_next_turn_effects(effects)

        assert updated["buff_1"].duration_type == "end_of_turn"
        assert "buff_1" in for_next_turn

    def test_scene_duration_not_affected(self):
        """Test scene duration effects are not affected."""
        effects = {
            "buff_1": TurnEndEffectState(
                effect_id="buff_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
            ),
            "scene_1": TurnEndEffectState(
                effect_id="scene_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="scene",
                applied_by="mech1",
            ),
        }

        updated, for_next_turn = advance_end_of_next_turn_effects(effects)

        assert updated["buff_1"].duration_type == "end_of_turn"
        assert updated["scene_1"].duration_type == "scene"
        assert "scene_1" not in for_next_turn


class TestResolveTurnEnd:
    """Tests for resolve_turn_end function."""

    def test_full_turn_end_no_triggers(self):
        """Test turn end with no triggers."""
        input_data = TurnEndInput(
            actor_id="mech1",
            round_number=1,
            turn_number=3,
            active_effects={},
        )

        result = resolve_turn_end(input_data)

        assert result.actor_id == "mech1"
        assert result.round_number == 1
        assert result.turn_number == 3
        assert len(result.triggers_resolved) == 0
        assert "No turn-end effects" in result.status_summary

    def test_full_turn_end_with_trigger(self):
        """Test turn end with a trigger."""
        input_data = TurnEndInput(
            actor_id="mech1",
            round_number=1,
            turn_number=3,
            triggers=[
                TurnEndTrigger(
                    trigger_id="skirmisher",
                    trigger_name="Skirmisher",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "accuracy", "value": 1},
                )
            ],
        )

        result = resolve_turn_end(input_data)

        assert len(result.triggers_resolved) == 1
        assert "accuracy" in result.status_summary
        assert len(result.new_effects) == 1

    def test_full_turn_end_with_effect_expiration(self):
        """Test turn end with effects expiring."""
        input_data = TurnEndInput(
            actor_id="mech1",
            round_number=1,
            turn_number=3,
            active_effects={
                "disengage_1": TurnEndEffectState(
                    effect_id="disengage_1",
                    effect_type="disengage",
                    target_id="mech1",
                    duration_type="end_of_turn",
                    applied_by="mech1",
                ),
            },
        )

        result = resolve_turn_end(input_data)

        assert "disengage_1" in result.effects_expired
        assert len(result.effects_expired) == 1

    def test_full_turn_end_multiple_effects_and_triggers(self):
        """Test turn end with multiple effects and triggers."""
        input_data = TurnEndInput(
            actor_id="mech1",
            round_number=2,
            turn_number=5,
            triggers=[
                TurnEndTrigger(
                    trigger_id="talent_1",
                    trigger_name="Talent 1",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "evasion", "value": 2},
                    order_priority=10,
                ),
                TurnEndTrigger(
                    trigger_id="system_1",
                    trigger_name="System 1",
                    source_type="system",
                    actor_id="mech1",
                    effect_data={"type": "cover_grant", "cover_type": "soft"},
                    order_priority=5,
                ),
            ],
            active_effects={
                "disengage_1": TurnEndEffectState(
                    effect_id="disengage_1",
                    effect_type="disengage",
                    target_id="mech1",
                    duration_type="end_of_turn",
                    applied_by="mech1",
                ),
            },
            specified_order=["system_1", "talent_1"],
        )

        result = resolve_turn_end(input_data)

        assert len(result.triggers_resolved) == 2
        assert result.triggers_resolved[0].trigger_id == "system_1"
        assert result.triggers_resolved[1].trigger_id == "talent_1"
        assert "disengage_1" in result.effects_expired
        assert len(result.new_effects) == 2


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_end_of_turn_effect(self):
        """Test create_end_of_turn_effect helper."""
        effect = create_end_of_turn_effect(
            effect_id="test",
            effect_type="buff",
            duration_type="end_of_next_turn",
            applied_by="mech1",
            target_id="mech1",
            effect_data={"value": 1},
        )

        assert effect.effect_id == "test"
        assert effect.effect_type == "buff"
        assert effect.duration_type == "end_of_next_turn"

    def test_create_end_of_turn_trigger(self):
        """Test create_end_of_turn_trigger helper."""
        trigger = create_end_of_turn_trigger(
            trigger_id="test_trigger",
            trigger_name="Test Trigger",
            source_type="talent",
            actor_id="mech1",
            effect_data={"type": "buff"},
            order_priority=5,
        )

        assert trigger.trigger_id == "test_trigger"
        assert trigger.source_type == "talent"
        assert trigger.order_priority == 5

    def test_has_active_turn_end_effect_found(self):
        """Test has_active_turn_end_effect when effect exists."""
        effects = {
            "buff_1": TurnEndEffectState(
                effect_id="buff_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
            ),
        }

        assert has_active_turn_end_effect(effects, "buff") is True

    def test_has_active_turn_end_effect_not_found(self):
        """Test has_active_turn_end_effect when effect doesn't exist."""
        effects = {
            "buff_1": TurnEndEffectState(
                effect_id="buff_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
            ),
        }

        assert has_active_turn_end_effect(effects, "cover_grant") is False

    def test_get_active_effects_by_type(self):
        """Test get_active_effects_by_type filter."""
        effects = {
            "buff_1": TurnEndEffectState(
                effect_id="buff_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
            ),
            "buff_2": TurnEndEffectState(
                effect_id="buff_2",
                effect_type="buff",
                target_id="mech2",
                duration_type="end_of_next_turn",
                applied_by="mech1",
            ),
            "cover_1": TurnEndEffectState(
                effect_id="cover_1",
                effect_type="cover_grant",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
            ),
        }

        buff_effects = get_active_effects_by_type(effects, "buff")

        assert len(buff_effects) == 2
        assert all(e.effect_type == "buff" for e in buff_effects)

    def test_apply_turn_end_effect_to_state_disengage(self):
        """Test applying disengage effect to combatant state."""
        effects = {
            "disengage_1": TurnEndEffectState(
                effect_id="disengage_1",
                effect_type="disengage",
                target_id="mech1",
                duration_type="end_of_turn",
                applied_by="mech1",
            ),
        }

        result = apply_turn_end_effect_to_state({}, effects)

        assert result.get("ignores_engagement") is True
        assert result.get("prevents_reactions") is True

    def test_apply_turn_end_effect_to_state_buff(self):
        """Test applying buff effect to combatant state."""
        effects = {
            "buff_1": TurnEndEffectState(
                effect_id="buff_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
                effect_data={"buff_type": "accuracy", "value": 1},
            ),
        }

        result = apply_turn_end_effect_to_state({}, effects)

        assert result.get("accuracy_bonus") == 1

    def test_apply_turn_end_effect_to_state_cover(self):
        """Test applying cover grant effect to combatant state."""
        effects = {
            "cover_1": TurnEndEffectState(
                effect_id="cover_1",
                effect_type="cover_grant",
                target_id="ally1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
                effect_data={"cover_type": "soft"},
            ),
        }

        result = apply_turn_end_effect_to_state({}, effects)

        assert result.get("cover_grant") == "soft"

    def test_apply_turn_end_effect_to_state_multiple(self):
        """Test applying multiple effects to combatant state."""
        effects = {
            "buff_1": TurnEndEffectState(
                effect_id="buff_1",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
                effect_data={"buff_type": "accuracy", "value": 1},
            ),
            "buff_2": TurnEndEffectState(
                effect_id="buff_2",
                effect_type="buff",
                target_id="mech1",
                duration_type="end_of_next_turn",
                applied_by="mech1",
                effect_data={"buff_type": "evasion", "value": 2},
            ),
        }

        result = apply_turn_end_effect_to_state({}, effects)

        assert result.get("accuracy_bonus") == 1
        assert result.get("defense_bonus") == 2


class TestTriggerOrderPriority:
    """Tests for trigger ordering logic."""

    def test_same_priority_order_preserved(self):
        """Test that same priority triggers maintain input order."""
        input_data = TurnEndInput(
            actor_id="mech1",
            triggers=[
                TurnEndTrigger(
                    trigger_id="trigger_a",
                    trigger_name="Trigger A",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "accuracy", "value": 1},
                    order_priority=10,
                ),
                TurnEndTrigger(
                    trigger_id="trigger_b",
                    trigger_name="Trigger B",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "evasion", "value": 2},
                    order_priority=10,
                ),
            ],
        )

        results, _ = resolve_turn_end_triggers(input_data)

        assert results[0].trigger_id == "trigger_a"
        assert results[1].trigger_id == "trigger_b"

    def test_mixed_specified_and_unspecified_order(self):
        """Test handling of mixed specified and unspecified order."""
        input_data = TurnEndInput(
            actor_id="mech1",
            triggers=[
                TurnEndTrigger(
                    trigger_id="trigger_a",
                    trigger_name="Trigger A",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "accuracy", "value": 1},
                    order_priority=10,
                ),
                TurnEndTrigger(
                    trigger_id="trigger_b",
                    trigger_name="Trigger B",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "evasion", "value": 2},
                    order_priority=5,
                ),
                TurnEndTrigger(
                    trigger_id="trigger_c",
                    trigger_name="Trigger C",
                    source_type="talent",
                    actor_id="mech1",
                    effect_data={"type": "buff", "buff_type": "defense", "value": 1},
                    order_priority=15,
                ),
            ],
            specified_order=["trigger_a"],
        )

        results, _ = resolve_turn_end_triggers(input_data)

        assert results[0].trigger_id == "trigger_a"
        assert results[1].trigger_id == "trigger_b"
        assert results[2].trigger_id == "trigger_c"
