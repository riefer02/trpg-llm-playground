"""Tests for progression and cooldown effect primitives."""

import pytest
from core.shared.effects import (
    ProgressionState,
    ProgressionEffect,
    GateProgressionEffect,
    PerTargetCounter,
    PerTargetCounterEffect,
    CooldownState,
    CooldownEffect,
    MechanicalEffect,
    StatusGrant,
    ConditionType,
    TriggerType,
)


class TestProgressionState:
    """Tests for ProgressionState model."""

    def test_default_values(self):
        state = ProgressionState()
        assert state.current_gate == 1
        assert state.max_gate == 4
        assert state.reset_on == "scene_end"
        assert state.per_target is True
        assert state.target_id is None

    def test_custom_values(self):
        state = ProgressionState(
            current_gate=2,
            max_gate=4,
            reset_on="rest",
            per_target=True,
            target_id="enemy_1",
        )
        assert state.current_gate == 2
        assert state.max_gate == 4
        assert state.reset_on == "rest"
        assert state.per_target is True
        assert state.target_id == "enemy_1"

    def test_gate_bounds(self):
        with pytest.raises(ValueError):
            ProgressionState(current_gate=0)
        with pytest.raises(ValueError):
            ProgressionState(current_gate=5)


class TestGateProgressionEffect:
    """Tests for GateProgressionEffect model."""

    def test_basic_gate(self):
        effect = GateProgressionEffect(
            gate_number=1,
            effect=MechanicalEffect(),
        )
        assert effect.gate_number == 1
        assert effect.prerequisite_gate is None

    def test_gate_with_prerequisite(self):
        effect = GateProgressionEffect(
            gate_number=3,
            prerequisite_gate=2,
            effect=MechanicalEffect(),
        )
        assert effect.gate_number == 3
        assert effect.prerequisite_gate == 2

    def test_gate_with_condition(self):
        effect = GateProgressionEffect(
            gate_number=2,
            prerequisite_gate=1,
            effect=MechanicalEffect(),
            condition="stunned",
        )
        assert effect.condition == "stunned"


class TestProgressionEffect:
    """Tests for ProgressionEffect model."""

    def test_empty_gates(self):
        effect = ProgressionEffect(
            progression_name="test_progression",
        )
        assert effect.progression_name == "test_progression"
        assert effect.gates == []
        assert effect.max_gate == 4

    def test_with_gates(self):
        effect = ProgressionEffect(
            progression_name="OSIRIS_Gates",
            reset_on="rest",
            max_gate=4,
            gates=[
                GateProgressionEffect(
                    gate_number=1,
                    effect=MechanicalEffect(),
                ),
                GateProgressionEffect(
                    gate_number=2,
                    prerequisite_gate=1,
                    effect=MechanicalEffect(),
                ),
            ],
        )
        assert len(effect.gates) == 2
        assert effect.gates[0].gate_number == 1
        assert effect.gates[1].prerequisite_gate == 1


class TestPerTargetCounter:
    """Tests for PerTargetCounter model."""

    def test_default_values(self):
        counter = PerTargetCounter(effect_id="stun_effect")
        assert counter.effect_id == "stun_effect"
        assert counter.current_count == 0
        assert counter.max_count == 1
        assert counter.reset_on == "scene_end"

    def test_custom_values(self):
        counter = PerTargetCounter(
            effect_id="test_effect",
            current_count=2,
            max_count=3,
            reset_on="rest",
            target_id="target_1",
        )
        assert counter.current_count == 2
        assert counter.max_count == 3


class TestPerTargetCounterEffect:
    """Tests for PerTargetCounterEffect model."""

    def test_basic_effect(self):
        effect = PerTargetCounterEffect(
            effect_id="stun_effect",
            max_count=1,
            reset_on="scene_end",
            effect=MechanicalEffect(
                status_grants=[
                    StatusGrant(status="stunned", target="enemy"),
                ]
            ),
        )
        assert effect.effect_id == "stun_effect"
        assert effect.max_count == 1
        assert len(effect.effect.status_grants) == 1

    def test_with_condition(self):
        effect = PerTargetCounterEffect(
            effect_id="test_effect",
            effect=MechanicalEffect(),
            condition="adjacent",
        )
        assert effect.condition == "adjacent"


class TestCooldownState:
    """Tests for CooldownState model."""

    def test_default_values(self):
        state = CooldownState(effect_id="ability_1")
        assert state.effect_id == "ability_1"
        assert state.turns_remaining == 0
        assert state.duration == 1
        assert state.trigger_on is None
        assert state.reset_on == "scene_end"

    def test_custom_values(self):
        state = CooldownState(
            effect_id="test_ability",
            turns_remaining=2,
            duration=3,
            trigger_on="on_hit",
            reset_on="turn_end",
            per_target=True,
        )
        assert state.turns_remaining == 2
        assert state.duration == 3
        assert state.trigger_on == "on_hit"


class TestCooldownEffect:
    """Tests for CooldownEffect model."""

    def test_basic_effect(self):
        effect = CooldownEffect(
            effect_id="ability_1",
            duration=2,
            effect=MechanicalEffect(),
        )
        assert effect.effect_id == "ability_1"
        assert effect.duration == 2

    def test_with_trigger(self):
        effect = CooldownEffect(
            effect_id="test_ability",
            duration=1,
            trigger_on="on_hit",
            reset_on="scene_end",
            effect=MechanicalEffect(),
        )
        assert effect.trigger_on == "on_hit"


class TestMechanicalEffectProgressionFields:
    """Tests for MechanicalEffect progression fields."""

    def test_empty_progression_effects(self):
        effect = MechanicalEffect()
        assert effect.progression_effects == []
        assert effect.per_target_counter_effects == []
        assert effect.cooldown_effects == []

    def test_with_progression_effect(self):
        effect = MechanicalEffect(
            progression_effects=[
                ProgressionEffect(
                    progression_name="test",
                    gates=[
                        GateProgressionEffect(
                            gate_number=1,
                            effect=MechanicalEffect(),
                        ),
                    ],
                ),
            ],
        )
        assert len(effect.progression_effects) == 1
        assert effect.progression_effects[0].progression_name == "test"

    def test_with_per_target_counter_effect(self):
        effect = MechanicalEffect(
            per_target_counter_effects=[
                PerTargetCounterEffect(
                    effect_id="test",
                    effect=MechanicalEffect(),
                ),
            ],
        )
        assert len(effect.per_target_counter_effects) == 1

    def test_with_cooldown_effect(self):
        effect = MechanicalEffect(
            cooldown_effects=[
                CooldownEffect(
                    effect_id="test",
                    effect=MechanicalEffect(),
                ),
            ],
        )
        assert len(effect.cooldown_effects) == 1


class TestConditionTypeLiterals:
    """Tests for condition type literals."""

    def test_out_of_phase_exists(self):
        assert "out_of_phase" in ConditionType.__args__

    def test_on_inflict_exists(self):
        assert "on_inflict" in TriggerType.__args__


class TestProgressionInMechanicalEffectIsEmpty:
    """Tests for is_empty method with progression fields."""

    def test_progression_effect_makes_non_empty(self):
        effect = MechanicalEffect(
            progression_effects=[
                ProgressionEffect(progression_name="test", gates=[]),
            ],
        )
        assert not effect.is_empty()

    def test_per_target_counter_effect_makes_non_empty(self):
        effect = MechanicalEffect(
            per_target_counter_effects=[
                PerTargetCounterEffect(effect_id="test", effect=MechanicalEffect()),
            ],
        )
        assert not effect.is_empty()

    def test_cooldown_effect_makes_non_empty(self):
        effect = MechanicalEffect(
            cooldown_effects=[
                CooldownEffect(effect_id="test", effect=MechanicalEffect()),
            ],
        )
        assert not effect.is_empty()
