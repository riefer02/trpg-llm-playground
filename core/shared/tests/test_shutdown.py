"""Tests for Shut Down and Boot Up action resolution."""

import pytest
from core.shared.shutdown import (
    resolve_shutdown,
    apply_shutdown_result,
    resolve_boot_up,
    apply_boot_up_result,
    ShutDownInput,
    BootUpInput,
    ShutDownRule,
    BootUpRule,
    SHUT_DOWN_ENDED_EFFECTS,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
)


@pytest.fixture
def test_combatant() -> CombatantState:
    """Create a test combatant for Shut Down tests."""
    return CombatantState(
        id="test_mech",
        name="Test Mech",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=10,
            evasion=10,
            e_defense=10,
            armor=0,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=8,
            heat_current=5,
            heat_cap=10,
        ),
        statuses=["exposed"],
        conditions=["impaired", "slowed"],
    )


@pytest.fixture
def combatant_with_lock_on() -> CombatantState:
    """Create a test combatant with lock_on status."""
    return CombatantState(
        id="test_mech",
        name="Test Mech",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=10,
            evasion=10,
            e_defense=10,
            armor=0,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=6,
            heat_cap=10,
        ),
        statuses=["lock_on", "exposed"],
        conditions=["impaired", "jammed"],
    )


@pytest.fixture
def shutdown_combatant() -> CombatantState:
    """Create a test combatant in shutdown status."""
    return CombatantState(
        id="test_mech",
        name="Test Mech",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=10,
            evasion=10,
            e_defense=10,
            armor=0,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=10,
        ),
        statuses=["shutdown"],
        conditions=[],
    )


class TestResolveShutDown:
    """Tests for Shut Down resolution (pure logic)."""

    def test_shutdown_clears_heat(self):
        """Test Shut Down clears heat to 0."""
        result = resolve_shutdown(ShutDownInput(actor_id="test_mech"))

        assert result.heat_cleared is True
        assert result.exposed_cleared is True
        assert result.shutdown_status_applied is True

    def test_shutdown_ends_tech_effects(self):
        """Test Shut Down ends tech effects."""
        result = resolve_shutdown(ShutDownInput(actor_id="test_mech"))

        assert len(result.tech_effects_ended) > 0
        assert "Lock On" in result.tech_effects_ended
        assert "Impaired" in result.tech_effects_ended
        assert "Slowed" in result.tech_effects_ended

    def test_shutdown_custom_rules(self):
        """Test Shut Down with custom rules."""
        custom_rules = ShutDownRule(
            clears_heat=True,
            clears_exposed=False,
            ends_tech_effects=True,
            reshackles_ai=True,
        )
        result = resolve_shutdown(
            ShutDownInput(actor_id="test_mech"), rules=custom_rules
        )

        assert result.heat_cleared is True
        assert result.exposed_cleared is False
        assert result.ai_reshackled is True

    def test_shutdown_no_heat_clear(self):
        """Test Shut Down without heat clearing."""
        custom_rules = ShutDownRule(
            clears_heat=False,
            clears_exposed=True,
            ends_tech_effects=False,
            reshackles_ai=False,
        )
        result = resolve_shutdown(
            ShutDownInput(actor_id="test_mech"), rules=custom_rules
        )

        assert result.heat_cleared is False
        assert result.exposed_cleared is True
        assert result.tech_effects_ended == []

    def test_shutdown_default_rules(self):
        """Test Shut Down with default rules."""
        from core.shared.shutdown import DEFAULT_SHUTDOWN_RULES

        result = resolve_shutdown(
            ShutDownInput(actor_id="test_mech"), rules=DEFAULT_SHUTDOWN_RULES
        )

        assert result.heat_cleared is True
        assert result.exposed_cleared is True


class TestApplyShutDownResult:
    """Tests for applying Shut Down results to combatant state."""

    def test_apply_shutdown_clears_heat(self, test_combatant: CombatantState):
        """Test applying Shut Down clears heat."""
        result = resolve_shutdown(ShutDownInput(actor_id="test_mech"))
        applied = apply_shutdown_result(test_combatant, result)

        assert applied.heat_cleared is True
        assert applied.updated_combatant.resources.heat_current == 0

    def test_apply_shutdown_clears_exposed(self, test_combatant: CombatantState):
        """Test applying Shut Down clears exposed status."""
        result = resolve_shutdown(ShutDownInput(actor_id="test_mech"))
        applied = apply_shutdown_result(test_combatant, result)

        assert applied.exposed_cleared is True
        assert "exposed" not in applied.updated_combatant.statuses
        assert "exposed" in applied.statuses_removed

    def test_apply_shutdown_adds_shutdown_status(self, test_combatant: CombatantState):
        """Test applying Shut Down adds shutdown status."""
        result = resolve_shutdown(ShutDownInput(actor_id="test_mech"))
        applied = apply_shutdown_result(test_combatant, result)

        assert applied.shutdown_status_added is True
        assert "shutdown" in applied.updated_combatant.statuses

    def test_apply_shutdown_removes_tech_conditions(
        self, combatant_with_lock_on: CombatantState
    ):
        """Test Shut Down removes tech-related statuses and conditions."""
        result = resolve_shutdown(ShutDownInput(actor_id="test_mech"))
        applied = apply_shutdown_result(combatant_with_lock_on, result)

        assert "lock_on" in applied.statuses_removed
        assert "impaired" in applied.conditions_removed
        assert "jammed" in applied.conditions_removed

    def test_apply_shutdown_preserves_other_statuses(
        self, test_combatant: CombatantState
    ):
        """Test Shut Down preserves non-tech statuses."""
        test_combatant_with_engaged = test_combatant.model_copy(
            update={"statuses": ["exposed", "engaged"]}
        )
        result = resolve_shutdown(ShutDownInput(actor_id="test_mech"))
        applied = apply_shutdown_result(test_combatant_with_engaged, result)

        assert "engaged" in applied.updated_combatant.statuses
        assert "exposed" not in applied.updated_combatant.statuses

    def test_apply_shutdown_custom_rules(self, test_combatant: CombatantState):
        """Test Shut Down with custom rules."""
        custom_rules = ShutDownRule(
            clears_heat=False,
            clears_exposed=True,
            ends_tech_effects=False,
            reshackles_ai=False,
        )
        result = resolve_shutdown(
            ShutDownInput(actor_id="test_mech"), rules=custom_rules
        )
        applied = apply_shutdown_result(test_combatant, result)

        assert applied.heat_cleared is False
        assert applied.updated_combatant.resources.heat_current == 5
        assert applied.exposed_cleared is True


class TestResolveBootUp:
    """Tests for Boot Up resolution (pure logic)."""

    def test_boot_up_with_pilot(self):
        """Test Boot Up when pilot is present."""
        result = resolve_boot_up(BootUpInput(actor_id="test_pilot", is_piloting=True))

        assert result.shutdown_status_ended is True
        assert result.was_piloting is True
        assert len(result.validation_errors) == 0

    def test_boot_up_without_pilot_fails(self):
        """Test Boot Up fails when pilot is not present."""
        result = resolve_boot_up(BootUpInput(actor_id="test_pilot", is_piloting=False))

        assert len(result.validation_errors) == 1
        assert "Must be piloting" in result.validation_errors[0]

    def test_boot_up_custom_rules_no_pilot_required(self):
        """Test Boot Up with custom rules (no pilot required)."""
        custom_rules = BootUpRule(requires_pilot=False)
        result = resolve_boot_up(
            BootUpInput(actor_id="test_pilot", is_piloting=False),
            rules=custom_rules,
        )

        assert len(result.validation_errors) == 0

    def test_boot_up_default_rules(self):
        """Test Boot Up with default rules."""
        from core.shared.shutdown import DEFAULT_BOOTUP_RULES

        result = resolve_boot_up(
            BootUpInput(actor_id="test_pilot", is_piloting=True),
            rules=DEFAULT_BOOTUP_RULES,
        )

        assert result.shutdown_status_ended is True
        assert result.pilot_required is True


class TestApplyBootUpResult:
    """Tests for applying Boot Up results to combatant state."""

    def test_apply_boot_up_removes_shutdown(self, shutdown_combatant: CombatantState):
        """Test Boot Up removes shutdown status."""
        result = resolve_boot_up(BootUpInput(actor_id="test_pilot", is_piloting=True))
        applied = apply_boot_up_result(shutdown_combatant, result)

        assert applied.shutdown_status_removed is True
        assert "shutdown" not in applied.updated_combatant.statuses
        assert "shutdown" in applied.statuses_removed

    def test_apply_boot_up_no_shutdown_to_remove(self, test_combatant: CombatantState):
        """Test Boot Up on non-shutdown combatant."""
        result = resolve_boot_up(BootUpInput(actor_id="test_pilot", is_piloting=True))
        applied = apply_boot_up_result(test_combatant, result)

        assert applied.shutdown_status_removed is False

    def test_apply_boot_up_fails_gracefully(self, shutdown_combatant: CombatantState):
        """Test Boot Up with validation error returns error in resolution result."""
        result = resolve_boot_up(BootUpInput(actor_id="test_pilot", is_piloting=False))

        assert len(result.validation_errors) == 1
        assert "Must be piloting" in result.validation_errors[0]

    def test_apply_boot_up_preserves_other_statuses(
        self, shutdown_combatant: CombatantState
    ):
        """Test Boot Up preserves other statuses."""
        shutdown_with_engaged = shutdown_combatant.model_copy(
            update={"statuses": ["shutdown", "engaged"]}
        )
        result = resolve_boot_up(BootUpInput(actor_id="test_pilot", is_piloting=True))
        applied = apply_boot_up_result(shutdown_with_engaged, result)

        assert "shutdown" not in applied.updated_combatant.statuses
        assert "engaged" in applied.updated_combatant.statuses


class TestShutDownEndedEffects:
    """Tests for Shut Down ended effects list."""

    def test_ended_effects_includes_all_tech_effects(self):
        """Test that all expected tech effects are in the ended effects list."""
        effect_ids = [e.effect_id for e in SHUT_DOWN_ENDED_EFFECTS]

        assert "lock_on" in effect_ids
        assert "impaired" in effect_ids
        assert "slowed" in effect_ids
        assert "jammed" in effect_ids
        assert "stunned" in effect_ids

    def test_ended_effects_have_correct_types(self):
        """Test that effects have correct type classifications."""
        for effect in SHUT_DOWN_ENDED_EFFECTS:
            if effect.effect_id == "lock_on":
                assert effect.effect_type == "status"


class TestShutDownBootUpIntegration:
    """Integration tests for Shut Down and Boot Up flow."""

    def test_shutdown_then_boot_up_flow(self, test_combatant: CombatantState):
        """Test full Shut Down -> Boot Up flow."""
        shutdown_input = ShutDownInput(actor_id="test_mech")
        shutdown_result = resolve_shutdown(shutdown_input)
        after_shutdown = apply_shutdown_result(test_combatant, shutdown_result)

        assert "shutdown" in after_shutdown.updated_combatant.statuses
        assert after_shutdown.updated_combatant.resources.heat_current == 0

        boot_input = BootUpInput(actor_id="test_pilot", is_piloting=True)
        boot_result = resolve_boot_up(boot_input)
        after_boot = apply_boot_up_result(after_shutdown.updated_combatant, boot_result)

        assert "shutdown" not in after_boot.updated_combatant.statuses
        assert after_boot.shutdown_status_removed is True

    def test_shutdown_preserves_HP(self, test_combatant: CombatantState):
        """Test Shut Down preserves HP while clearing heat."""
        shutdown_result = resolve_shutdown(ShutDownInput(actor_id="test_mech"))
        applied = apply_shutdown_result(test_combatant, shutdown_result)

        assert applied.updated_combatant.resources.hp_current == 8
        assert applied.updated_combatant.resources.heat_current == 0


class TestShutDownRules:
    """Tests for Shut Down rule configuration."""

    def test_default_shutdown_rules(self):
        """Test default Shut Down rules have correct settings."""
        from core.shared.shutdown import DEFAULT_SHUTDOWN_RULES

        assert DEFAULT_SHUTDOWN_RULES.clears_heat is True
        assert DEFAULT_SHUTDOWN_RULES.clears_exposed is True
        assert DEFAULT_SHUTDOWN_RULES.ends_tech_effects is True

    def test_custom_ended_conditions(self):
        """Test custom Shut Down rules with specific conditions."""
        rules = ShutDownRule(
            clears_heat=False,
            clears_exposed=False,
            ends_tech_effects=False,
            reshackles_ai=False,
        )
        result = resolve_shutdown(ShutDownInput(actor_id="test_mech"), rules=rules)

        assert result.heat_cleared is False
        assert result.exposed_cleared is False
        assert result.tech_effects_ended == []

    def test_default_bootup_rules(self):
        """Test default Boot Up rules have correct settings."""
        from core.shared.shutdown import DEFAULT_BOOTUP_RULES

        assert DEFAULT_BOOTUP_RULES.requires_pilot is True
        assert DEFAULT_BOOTUP_RULES.clears_shutdown_status is True
