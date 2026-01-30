"""End-to-end tests for frame core power activation in combat.

Tests that frame core powers can be activated, apply effects correctly,
and respect the once-per-mission limitation.
"""


from core.tests.e2e_helpers import (
    make_pilot_with_talents,
    make_combatant_from_pilot,
    make_combatant,
    make_enemy_combatant,
    make_duel_scenario,
)
from core.mech.combat_state import (
    MechCombatScenario,
    CombatTurn,
)
from core.mech.combat_execution import (
    ActionExecutionInput,
    execute_action,
    get_available_actions,
    start_turn,
)
from core.mech.action_economy import ActionEconomyState
from core.mech.frame import (
    collect_frame_trait_effects,
    get_core_power_effects,
)
from core.mech.compendium import get_frame_definition
from core.shared.effects import MechanicalEffect, AccuracyModifier


class TestCorePowerAvailability:
    """Tests for core power availability in action lists."""

    def test_core_power_in_protocols_when_available(self):
        """Core power shows in protocol list when available."""
        pilot = make_pilot_with_talents("TEST", [])
        combatant = make_combatant_from_pilot(pilot, "gms_everest", (0, 0))

        # Verify combatant has core power available
        assert combatant.core_power_available is True
        assert combatant.core_power_active is False

        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()

        result = get_available_actions(scenario, combatant.id, economy)
        protocol_ids = [p.action_id for p in result.protocols]

        # Should include activate_core_power if frame has a core system
        if combatant.core_power_effects is not None:
            assert "activate_core_power" in protocol_ids

    def test_core_power_not_in_protocols_when_used(self):
        """Core power not in protocols after it's been used."""
        pilot = make_pilot_with_talents("TEST", [])
        combatant = make_combatant_from_pilot(pilot, "gms_everest", (0, 0))

        # Manually mark core power as used
        combatant = combatant.model_copy(update={"core_power_available": False})

        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()

        result = get_available_actions(scenario, combatant.id, economy)
        protocol_ids = [p.action_id for p in result.protocols]

        assert "activate_core_power" not in protocol_ids

    def test_core_power_not_in_protocols_without_core_system(self):
        """Core power not available if frame has no core system."""
        # Create combatant without core power effects
        combatant = make_combatant(
            id="test_mech",
            core_power_available=True,
            core_power_effects=None,  # No core system
        )

        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()

        result = get_available_actions(scenario, combatant.id, economy)
        protocol_ids = [p.action_id for p in result.protocols]

        assert "activate_core_power" not in protocol_ids


class TestCorePowerActivation:
    """Tests for activating core powers."""

    def test_activate_core_power_success(self):
        """Successfully activating core power updates state."""
        core_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )
        combatant = make_combatant(
            id="test_mech",
            core_power_available=True,
            core_power_active=False,
            core_power_effects=core_effect,
        )

        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()
        turn = CombatTurn(actor_id="test_mech")

        action = ActionExecutionInput(
            actor_id="test_mech",
            action_id="activate_core_power",
            action_type="protocol",
        )

        new_scenario, _, _, result = execute_action(scenario, turn, economy, action)

        assert result.success
        updated = next(c for c in new_scenario.combatants if c.id == "test_mech")
        assert updated.core_power_available is False  # Used
        assert updated.core_power_active is True  # Now active

    def test_activate_core_power_fails_when_already_used(self):
        """Cannot activate core power if already used this mission."""
        core_effect = MechanicalEffect()
        combatant = make_combatant(
            id="test_mech",
            core_power_available=False,  # Already used
            core_power_active=False,
            core_power_effects=core_effect,
        )

        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()
        turn = CombatTurn(actor_id="test_mech")

        action = ActionExecutionInput(
            actor_id="test_mech",
            action_id="activate_core_power",
            action_type="protocol",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action)

        assert not result.success
        assert result.error is not None
        assert "already used" in result.error.lower()

    def test_activate_core_power_fails_when_already_active(self):
        """Cannot activate core power if already active."""
        core_effect = MechanicalEffect()
        combatant = make_combatant(
            id="test_mech",
            core_power_available=True,
            core_power_active=True,  # Already active
            core_power_effects=core_effect,
        )

        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()
        turn = CombatTurn(actor_id="test_mech")

        action = ActionExecutionInput(
            actor_id="test_mech",
            action_id="activate_core_power",
            action_type="protocol",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action)

        assert not result.success
        assert result.error is not None
        assert "already active" in result.error.lower()


class TestCorePowerOncePerMission:
    """Tests for the once-per-mission core power limitation."""

    def test_core_power_once_per_mission(self):
        """Core power can only be used once per mission."""
        core_effect = MechanicalEffect()
        combatant = make_combatant(
            id="test_mech",
            core_power_available=True,
            core_power_active=False,
            core_power_effects=core_effect,
        )

        scenario = MechCombatScenario(combatants=[combatant])
        economy = ActionEconomyState()
        turn = CombatTurn(actor_id="test_mech")

        action = ActionExecutionInput(
            actor_id="test_mech",
            action_id="activate_core_power",
            action_type="protocol",
        )

        # First activation succeeds
        scenario, turn, economy, result1 = execute_action(scenario, turn, economy, action)
        assert result1.success

        # Second activation fails
        scenario, turn, economy, result2 = execute_action(scenario, turn, economy, action)
        assert not result2.success
        assert result2.error is not None
        assert "already" in result2.error.lower()

    def test_core_power_state_persists_through_combat(self):
        """Core power state persists through multiple turns."""
        pilot = make_pilot_with_talents("TEST", [])
        attacker = make_combatant_from_pilot(pilot, "gms_everest", (0, 0))
        defender = make_enemy_combatant()

        scenario = make_duel_scenario(attacker, defender)

        # Activate core power
        scenario, turn_result = start_turn(scenario, attacker.id)
        economy = turn_result.economy
        turn = CombatTurn(actor_id=attacker.id)

        if attacker.core_power_effects is not None:
            action = ActionExecutionInput(
                actor_id=attacker.id,
                action_id="activate_core_power",
                action_type="protocol",
            )
            scenario, turn, economy, _ = execute_action(scenario, turn, economy, action)

            # Verify state
            updated = next(c for c in scenario.combatants if c.id == attacker.id)
            assert updated.core_power_available is False
            assert updated.core_power_active is True

            # Execute an attack
            attack_action = ActionExecutionInput(
                actor_id=attacker.id,
                action_id="skirmish",
                action_type="quick",
                target_ids=[defender.id],
            )
            scenario, turn, economy, _ = execute_action(scenario, turn, economy, attack_action)

            # Core power state should persist
            updated = next(c for c in scenario.combatants if c.id == attacker.id)
            assert updated.core_power_available is False
            assert updated.core_power_active is True


class TestFrameTraitEffects:
    """Tests for frame trait effects on combatants."""

    def test_frame_traits_collected_on_combatant_creation(self):
        """Frame trait effects are collected when creating combatant."""
        pilot = make_pilot_with_talents("TEST", [])
        combatant = make_combatant_from_pilot(pilot, "gms_everest", (0, 0))

        # GMS Everest should have frame trait effects
        assert isinstance(combatant.frame_trait_effects, list)

    def test_frame_definition_has_traits(self):
        """Frame definitions include trait information."""
        frame = get_frame_definition("gms_everest")
        assert frame is not None
        assert isinstance(frame.traits, list)

    def test_collect_frame_trait_effects_function(self):
        """collect_frame_trait_effects returns list of effects."""
        frame = get_frame_definition("gms_everest")
        assert frame is not None

        effects = collect_frame_trait_effects(frame)
        assert isinstance(effects, list)

    def test_get_core_power_effects_function(self):
        """get_core_power_effects returns core system effects or None."""
        frame = get_frame_definition("gms_everest")
        assert frame is not None

        effects = get_core_power_effects(frame)
        # May be None or MechanicalEffect depending on frame definition
        assert effects is None or isinstance(effects, MechanicalEffect)


class TestCorePowerEffectsApplication:
    """Tests for core power effects being applied in combat."""

    def test_core_power_accuracy_bonus_applies_when_active(self):
        """Core power accuracy bonuses apply when core power is active."""
        from core.mech.combat_helpers import _get_talent_accuracy_modifiers

        # Create combatant with core power that gives +1 accuracy
        core_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=2, applies_to="all")]
        )
        combatant = make_combatant(
            id="test_mech",
            core_power_available=False,  # Already used
            core_power_active=True,  # But active
            core_power_effects=core_effect,
        )

        # Get accuracy modifiers (should include core power)
        acc_mod, _ = _get_talent_accuracy_modifiers(
            combatant,
            is_ranged=True,
        )

        # Core power should contribute +2 accuracy
        assert acc_mod == 2

    def test_core_power_effects_not_applied_when_inactive(self):
        """Core power effects don't apply when not activated."""
        from core.mech.combat_helpers import _get_talent_accuracy_modifiers

        # Create combatant with core power that gives +2 accuracy
        core_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=2, applies_to="all")]
        )
        combatant = make_combatant(
            id="test_mech",
            core_power_available=True,  # Available but not used
            core_power_active=False,  # Not active
            core_power_effects=core_effect,
        )

        # Get accuracy modifiers (should NOT include core power)
        acc_mod, _ = _get_talent_accuracy_modifiers(
            combatant,
            is_ranged=True,
        )

        # Core power should NOT contribute
        assert acc_mod == 0


class TestCorePowerWithTalents:
    """Tests for core powers working alongside talent effects."""

    def test_core_power_stacks_with_talents(self):
        """Core power effects stack with talent effects."""
        from core.mech.combat_helpers import _get_talent_accuracy_modifiers
        from core.shared.effects import MechanicalEffect, AccuracyModifier

        # Talent gives +1 accuracy
        talent_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )

        # Core power gives +2 accuracy
        core_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=2, applies_to="all")]
        )

        combatant = make_combatant(
            id="test_mech",
            talent_effects=[talent_effect],
            core_power_available=False,
            core_power_active=True,  # Core power active
            core_power_effects=core_effect,
        )

        # Get total accuracy modifiers
        acc_mod, _ = _get_talent_accuracy_modifiers(
            combatant,
            is_ranged=True,
        )

        # Should get +1 (talent) + +2 (core power) = +3
        assert acc_mod == 3

    def test_frame_traits_stack_with_talents_and_core(self):
        """Frame traits, talents, and core power all stack."""
        from core.mech.combat_helpers import _get_talent_accuracy_modifiers
        from core.shared.effects import MechanicalEffect, AccuracyModifier

        # Talent gives +1
        talent_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )

        # Frame trait gives +1
        frame_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )

        # Core power gives +1
        core_effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )

        combatant = make_combatant(
            id="test_mech",
            talent_effects=[talent_effect],
            frame_trait_effects=[frame_effect],
            core_power_available=False,
            core_power_active=True,
            core_power_effects=core_effect,
        )

        acc_mod, _ = _get_talent_accuracy_modifiers(
            combatant,
            is_ranged=True,
        )

        # Should get +1 (talent) + +1 (frame) + +1 (core) = +3
        assert acc_mod == 3
