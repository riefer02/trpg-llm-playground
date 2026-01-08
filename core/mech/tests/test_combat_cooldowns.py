"""Tests for cooldown trigger integration."""

import pytest
from core.shared.effects import (
    CooldownState,
    CooldownEffect,
    CooldownResetTrigger,
    TriggerType,
    MechanicalEffect,
)
from core.mech.combat_resolution import (
    check_action_on_cooldown,
    apply_cooldown,
    decrement_cooldowns_on_turn_start,
    decrement_cooldowns_on_turn_end,
    decrement_cooldowns_on_round_end,
    reset_cooldowns_on_scene_end,
    get_cooldown_state,
    CooldownCheckResult,
    CooldownApplicationResult,
    CooldownDecrementResult,
)
from core.mech.combat_state import (
    CombatStats,
    CombatResources,
    CombatantState,
)
from core.mech.validation.combat_validation import (
    validate_combat_scenario,
    _index_cooldown_states,
    _check_action_on_cooldown,
    CombatValidationIssue,
)
from core.mech.combat_state import (
    MechCombatScenario,
    CombatRound,
    CombatTurn,
    ActionUse,
    AppliedPerTargetEffect,
)


class TestCooldownStateResolution:
    """Tests for cooldown resolution helpers."""

    def test_check_action_not_on_cooldown(self):
        """Test checking an action that is not on cooldown."""
        cooldown_states: dict[str, CooldownState] = {}
        result = check_action_on_cooldown(
            actor_cooldown_states=cooldown_states,
            effect_id="ability_1",
        )
        assert result.is_on_cooldown is False
        assert result.effect_id == "ability_1"
        assert result.turns_remaining is None

    def test_check_action_on_cooldown_global(self):
        """Test checking an action that is on cooldown (global)."""
        cooldown_states = {
            "ability_1": CooldownState(
                effect_id="ability_1",
                turns_remaining=2,
                duration=2,
            )
        }
        result = check_action_on_cooldown(
            actor_cooldown_states=cooldown_states,
            effect_id="ability_1",
        )
        assert result.is_on_cooldown is True
        assert result.effect_id == "ability_1"
        assert result.turns_remaining == 2

    def test_check_action_on_cooldown_per_target(self):
        """Test checking an action that is on cooldown for a specific target."""
        cooldown_states = {
            "ability_1:target_a": CooldownState(
                effect_id="ability_1",
                turns_remaining=1,
                duration=1,
                per_target=True,
                target_id="target_a",
            )
        }
        result = check_action_on_cooldown(
            actor_cooldown_states=cooldown_states,
            effect_id="ability_1",
            target_id="target_a",
        )
        assert result.is_on_cooldown is True
        assert result.turns_remaining == 1
        assert result.target_id == "target_a"

    def test_check_action_cooldown_expired(self):
        """Test checking an action whose cooldown has expired."""
        cooldown_states = {
            "ability_1": CooldownState(
                effect_id="ability_1",
                turns_remaining=0,
                duration=1,
            )
        }
        result = check_action_on_cooldown(
            actor_cooldown_states=cooldown_states,
            effect_id="ability_1",
        )
        assert result.is_on_cooldown is False
        assert result.turns_remaining == 0

    def test_apply_cooldown_new(self):
        """Test applying a cooldown for a new effect."""
        cooldown_states: dict[str, CooldownState] = {}
        result = apply_cooldown(
            actor_cooldown_states=cooldown_states,
            effect_id="ability_1",
            duration=2,
        )
        assert result.applied is True
        assert result.effect_id == "ability_1"
        assert result.turns_remaining == 2
        assert result.duration == 2
        assert result.previous_turns_remaining is None
        assert "ability_1" in cooldown_states
        assert cooldown_states["ability_1"].turns_remaining == 2

    def test_apply_cooldown_replaces_existing(self):
        """Test applying a cooldown replaces an existing one."""
        cooldown_states = {
            "ability_1": CooldownState(
                effect_id="ability_1",
                turns_remaining=1,
                duration=1,
            )
        }
        result = apply_cooldown(
            actor_cooldown_states=cooldown_states,
            effect_id="ability_1",
            duration=3,
        )
        assert result.applied is True
        assert result.turns_remaining == 3
        assert result.previous_turns_remaining == 1
        assert cooldown_states["ability_1"].turns_remaining == 3

    def test_apply_cooldown_per_target(self):
        """Test applying a per-target cooldown."""
        cooldown_states: dict[str, CooldownState] = {}
        result = apply_cooldown(
            actor_cooldown_states=cooldown_states,
            effect_id="ability_1",
            duration=1,
            target_id="target_a",
        )
        assert result.applied is True
        assert result.target_id == "target_a"
        key = "ability_1:target_a"
        assert key in cooldown_states
        assert cooldown_states[key].per_target is True
        assert cooldown_states[key].target_id == "target_a"

    def test_decrement_cooldowns_on_turn_start(self):
        """Test decrementing cooldowns at turn start."""
        cooldown_states = {
            "ability_1": CooldownState(
                effect_id="ability_1",
                turns_remaining=2,
                duration=2,
                reset_on="turn_start",
            ),
            "ability_2": CooldownState(
                effect_id="ability_2",
                turns_remaining=1,
                duration=1,
                reset_on="turn_end",
            ),
        }
        results = decrement_cooldowns_on_turn_start(
            actor_cooldown_states=cooldown_states
        )
        assert len(results) == 1
        assert results[0].effect_id == "ability_1"
        assert results[0].turns_remaining_before == 2
        assert results[0].turns_remaining_after == 1
        assert cooldown_states["ability_1"].turns_remaining == 1
        assert "ability_2" in cooldown_states

    def test_decrement_cooldowns_removes_expired(self):
        """Test that expired cooldowns are removed when reaching 0."""
        cooldown_states = {
            "ability_1": CooldownState(
                effect_id="ability_1",
                turns_remaining=1,
                duration=1,
                reset_on="turn_start",
            ),
        }
        results = decrement_cooldowns_on_turn_start(
            actor_cooldown_states=cooldown_states
        )
        assert len(results) == 1
        assert results[0].turns_remaining_after == 0
        assert "ability_1" not in cooldown_states

    def test_decrement_cooldowns_on_turn_end(self):
        """Test decrementing cooldowns at turn end."""
        cooldown_states = {
            "ability_1": CooldownState(
                effect_id="ability_1",
                turns_remaining=2,
                duration=2,
                reset_on="turn_end",
            ),
        }
        results = decrement_cooldowns_on_turn_end(actor_cooldown_states=cooldown_states)
        assert len(results) == 1
        assert results[0].effect_id == "ability_1"
        assert results[0].turns_remaining_before == 2
        assert results[0].turns_remaining_after == 1
        assert cooldown_states["ability_1"].turns_remaining == 1

    def test_decrement_cooldowns_on_round_end(self):
        """Test decrementing cooldowns at round end."""
        cooldown_states = {
            "ability_1": CooldownState(
                effect_id="ability_1",
                turns_remaining=3,
                duration=3,
                reset_on="round_end",
            ),
            "ability_2": CooldownState(
                effect_id="ability_2",
                turns_remaining=1,
                duration=1,
                reset_on="scene_end",
            ),
        }
        results = decrement_cooldowns_on_round_end(
            actor_cooldown_states=cooldown_states
        )
        assert len(results) == 1
        assert results[0].effect_id == "ability_1"
        assert cooldown_states["ability_1"].turns_remaining == 2
        assert "ability_2" in cooldown_states

    def test_reset_cooldowns_on_scene_end(self):
        """Test resetting cooldowns at scene end."""
        cooldown_states = {
            "ability_1": CooldownState(
                effect_id="ability_1",
                turns_remaining=2,
                duration=2,
                reset_on="scene_end",
            ),
            "ability_2": CooldownState(
                effect_id="ability_2",
                turns_remaining=1,
                duration=1,
                reset_on="never",
            ),
            "ability_3": CooldownState(
                effect_id="ability_3",
                turns_remaining=1,
                duration=1,
                reset_on="full_repair",
            ),
        }
        cleared = reset_cooldowns_on_scene_end(actor_cooldown_states=cooldown_states)
        assert set(cleared) == {"ability_1", "ability_3"}
        assert "ability_1" not in cooldown_states
        assert "ability_3" not in cooldown_states
        assert "ability_2" in cooldown_states

    def test_get_cooldown_state(self):
        """Test getting the current cooldown state for an effect."""
        cooldown_states = {
            "ability_1": CooldownState(
                effect_id="ability_1",
                turns_remaining=2,
                duration=2,
            ),
        }
        state = get_cooldown_state(
            actor_cooldown_states=cooldown_states,
            effect_id="ability_1",
        )
        assert state is not None
        assert state.turns_remaining == 2

        state = get_cooldown_state(
            actor_cooldown_states=cooldown_states,
            effect_id="nonexistent",
        )
        assert state is None


class TestCooldownIndexing:
    """Tests for cooldown state indexing."""

    def test_index_cooldown_states_global_only(self):
        """Test indexing cooldown states with only global cooldowns."""
        combatant = CombatantState(
            id="test_mech",
            name="Test Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_2",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
            cooldown_states={
                "ability_1": CooldownState(
                    effect_id="ability_1",
                    turns_remaining=2,
                    duration=2,
                ),
            },
        )
        global_cds, per_target_cds = _index_cooldown_states(combatant)
        assert "ability_1" in global_cds
        assert len(per_target_cds) == 0

    def test_index_cooldown_states_per_target(self):
        """Test indexing cooldown states with per-target cooldowns."""
        combatant = CombatantState(
            id="test_mech",
            name="Test Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_2",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
            cooldown_states={
                "ability_1:target_a": CooldownState(
                    effect_id="ability_1",
                    turns_remaining=1,
                    duration=1,
                    per_target=True,
                    target_id="target_a",
                ),
            },
        )
        global_cds, per_target_cds = _index_cooldown_states(combatant)
        assert len(global_cds) == 0
        assert "ability_1:target_a" in per_target_cds


class TestCooldownValidation:
    """Tests for cooldown validation in combat scenarios."""

    def test_action_blocked_by_cooldown(self):
        """Test that actions are blocked when on cooldown."""
        actor = CombatantState(
            id="mech_a",
            name="Mech A",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_2",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
            cooldown_states={
                "skirmish": CooldownState(
                    effect_id="skirmish",
                    turns_remaining=2,
                    duration=2,
                ),
            },
        )
        target = CombatantState(
            id="target_b",
            name="Target B",
            side="hostiles",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
        )

        scenario = MechCombatScenario(
            combatants=[actor, target],
            rounds=[
                CombatRound(
                    round_index=1,
                    turns=[
                        CombatTurn(
                            actor_id="mech_a",
                            actions=[
                                ActionUse(
                                    action_id="skirmish",
                                    action_type="full",
                                    target_id="target_b",
                                    applied_per_target_effects=[
                                        AppliedPerTargetEffect(
                                            effect_id="skirmish",
                                            target_id="target_b",
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        validation = validate_combat_scenario(scenario)
        assert validation.valid is False
        cooldown_issues = [
            i for i in validation.issues if i.code == "action_on_cooldown"
        ]
        assert len(cooldown_issues) == 1
        assert "skirmish" in cooldown_issues[0].message
        assert "2 turns remaining" in cooldown_issues[0].message

    def test_action_not_blocked_cooldown_expired(self):
        """Test that actions are not blocked when cooldown has expired."""
        actor = CombatantState(
            id="mech_a",
            name="Mech A",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_2",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
            cooldown_states={
                "skirmish": CooldownState(
                    effect_id="skirmish",
                    turns_remaining=0,
                    duration=1,
                ),
            },
        )
        target = CombatantState(
            id="target_b",
            name="Target B",
            side="hostiles",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
        )

        scenario = MechCombatScenario(
            combatants=[actor, target],
            rounds=[
                CombatRound(
                    round_index=1,
                    turns=[
                        CombatTurn(
                            actor_id="mech_a",
                            actions=[
                                ActionUse(
                                    action_id="skirmish",
                                    action_type="full",
                                    target_id="target_b",
                                    applied_per_target_effects=[
                                        AppliedPerTargetEffect(
                                            effect_id="skirmish",
                                            target_id="target_b",
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        validation = validate_combat_scenario(scenario)
        cooldown_issues = [
            i for i in validation.issues if i.code == "action_on_cooldown"
        ]
        assert len(cooldown_issues) == 0

    def test_action_allowed_no_cooldown(self):
        """Test that actions are allowed when no cooldown exists."""
        actor = CombatantState(
            id="mech_a",
            name="Mech A",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_2",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
        )
        target = CombatantState(
            id="target_b",
            name="Target B",
            side="hostiles",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
        )

        scenario = MechCombatScenario(
            combatants=[actor, target],
            rounds=[
                CombatRound(
                    round_index=1,
                    turns=[
                        CombatTurn(
                            actor_id="mech_a",
                            actions=[
                                ActionUse(
                                    action_id="skirmish",
                                    action_type="full",
                                    target_id="target_b",
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        validation = validate_combat_scenario(scenario)
        assert validation.valid is True


class TestCheckActionOnCooldownHelper:
    """Tests for the _check_action_on_cooldown helper."""

    def test_no_cooldown_issues(self):
        """Test that no issues are raised when action is not on cooldown."""
        global_cds: dict[str, CooldownState] = {}
        per_target_cds: dict[str, CooldownState] = {}
        actor_cooldowns = (global_cds, per_target_cds)

        issues: list[CombatValidationIssue] = []
        action = ActionUse(
            action_id="skirmish",
            action_type="full",
            target_id="target_a",
            applied_per_target_effects=[
                AppliedPerTargetEffect(
                    effect_id="skirmish",
                    target_id="target_a",
                )
            ],
        )

        is_blocked = _check_action_on_cooldown(action, actor_cooldowns, issues)
        assert is_blocked is False
        assert len(issues) == 0

    def test_cooldown_blocks_action(self):
        """Test that issues are raised when action is on cooldown."""
        global_cds = {
            "skirmish": CooldownState(
                effect_id="skirmish",
                turns_remaining=2,
                duration=2,
            )
        }
        per_target_cds: dict[str, CooldownState] = {}
        actor_cooldowns = (global_cds, per_target_cds)

        issues: list[CombatValidationIssue] = []
        action = ActionUse(
            action_id="skirmish",
            action_type="full",
            target_id="target_a",
            applied_per_target_effects=[
                AppliedPerTargetEffect(
                    effect_id="skirmish",
                    target_id="target_a",
                )
            ],
        )

        is_blocked = _check_action_on_cooldown(action, actor_cooldowns, issues)
        assert is_blocked is True
        assert len(issues) == 1
        assert issues[0].code == "action_on_cooldown"
