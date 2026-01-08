"""Tests for core/shared/id_helpers.py"""

import pytest
from pydantic import BaseModel, Field

from core.shared.id_helpers import (
    IdField,
    PilotIdField,
    MechIdField,
    WeaponIdField,
    SystemIdField,
    CombatantIdField,
    ActionIdField,
    EffectIdField,
    DeployableIdField,
    DroneIdField,
    TemplateIdField,
)
from core.shared.ids import (
    PilotId,
    MechId,
    WeaponId,
    SystemId,
    CombatantId,
    ActionId,
    EffectId,
    DeployableId,
    DroneId,
    TemplateId,
)


class TestIdFieldCoercion:
    """Test that IdField[T] properly coerces strings to typed IDs."""

    def test_pilot_id_coercion(self):
        """Test PilotId coercion from string."""

        class TestModel(BaseModel):
            pilot_id: PilotIdField

        model = TestModel(pilot_id="p1")
        assert model.pilot_id == "p1"
        assert isinstance(model.pilot_id, str)

    def test_weapon_id_coercion(self):
        """Test WeaponId coercion from string."""

        class TestModel(BaseModel):
            weapon_id: WeaponIdField

        model = TestModel(weapon_id="w_heavy_cannon")
        assert model.weapon_id == "w_heavy_cannon"

    def test_system_id_coercion(self):
        """Test SystemId coercion from string."""

        class TestModel(BaseModel):
            system_id: SystemIdField

        model = TestModel(system_id="s_coolant_injection")
        assert model.system_id == "s_coolant_injection"

    def test_combatant_id_coercion(self):
        """Test CombatantId coercion from string."""

        class TestModel(BaseModel):
            combatant_id: CombatantIdField

        model = TestModel(combatant_id="c_player_1")
        assert model.combatant_id == "c_player_1"

    def test_multiple_id_fields(self):
        """Test model with multiple typed ID fields."""

        class TestModel(BaseModel):
            pilot_id: PilotIdField
            mech_id: MechIdField
            weapon_id: WeaponIdField

        model = TestModel(pilot_id="p1", mech_id="m_everest", weapon_id="w_rifle")
        assert model.pilot_id == "p1"
        assert model.mech_id == "m_everest"
        assert model.weapon_id == "w_rifle"

    def test_action_id_coercion(self):
        """Test ActionId coercion."""

        class TestModel(BaseModel):
            action_id: ActionIdField

        model = TestModel(action_id="a_overwatch")
        assert model.action_id == "a_overwatch"

    def test_effect_id_coercion(self):
        """Test EffectId coercion."""

        class TestModel(BaseModel):
            effect_id: EffectIdField

        model = TestModel(effect_id="e_damage_bonus")
        assert model.effect_id == "e_damage_bonus"

    def test_deployable_id_coercion(self):
        """Test DeployableId coercion."""

        class TestModel(BaseModel):
            deployable_id: DeployableIdField

        model = TestModel(deployable_id="d_turret_1")
        assert model.deployable_id == "d_turret_1"

    def test_drone_id_coercion(self):
        """Test DroneId coercion."""

        class TestModel(BaseModel):
            drone_id: DroneIdField

        model = TestModel(drone_id="dr_hive_1")
        assert model.drone_id == "dr_hive_1"

    def test_template_id_coercion(self):
        """Test TemplateId coercion."""

        class TestModel(BaseModel):
            template_id: TemplateIdField

        model = TestModel(template_id="t_assault")
        assert model.template_id == "t_assault"


class TestIdFieldWithField:
    """Test IdField[T] with Pydantic Field for defaults."""

    def test_id_field_with_default(self):
        """Test ID field with default value."""

        class TestModel(BaseModel):
            pilot_id: PilotIdField = "p_default"

        model = TestModel()
        assert model.pilot_id == "p_default"

    def test_id_field_with_factory(self):
        """Test ID field with default_factory."""

        class TestModel(BaseModel):
            pilot_ids: list[PilotIdField] = Field(default_factory=list)

        model = TestModel()
        assert model.pilot_ids == []


class TestBackwardCompatibility:
    """Test backward compatibility with existing code patterns."""

    def test_string_assignment_still_works(self):
        """Ensure existing code assigning strings continues to work."""

        class TestModel(BaseModel):
            pilot_id: PilotIdField
            target_ids: list[CombatantIdField] = Field(default_factory=list)

        model = TestModel(pilot_id="p1", target_ids=["c_enemy_1", "c_enemy_2"])
        assert model.pilot_id == "p1"
        assert len(model.target_ids) == 2

    def test_empty_string_allowed(self):
        """Test that empty strings are allowed (for uninitialized IDs)."""

        class TestModel(BaseModel):
            pilot_id: PilotIdField = ""

        model = TestModel()
        assert model.pilot_id == ""


class TestIdFieldFactory:
    """Test the IdField[T] factory function."""

    def test_id_field_creates_correct_type(self):
        """Test that IdField creates properly typed fields."""
        from core.shared.models import FrozenModel

        class TestModel(FrozenModel):
            pilot_id: PilotIdField

        model = TestModel(pilot_id="p1")
        assert model.pilot_id == "p1"

    def test_id_field_with_system(self):
        """Test IdField with SystemId."""
        from core.shared.models import FrozenModel

        class TestModel(FrozenModel):
            system_id: SystemIdField

        model = TestModel(system_id="s_enhanced_sensors")
        assert model.system_id == "s_enhanced_sensors"


class TestIdEquality:
    """Test equality comparisons with typed IDs."""

    def test_same_id_values_equal(self):
        """Test that same ID values are equal."""
        from core.shared.models import FrozenModel

        class TestModel(FrozenModel):
            pilot_id: PilotIdField

        model1 = TestModel(pilot_id="p1")
        model2 = TestModel(pilot_id="p1")

        assert model1.pilot_id == model2.pilot_id

    def test_different_id_values_not_equal(self):
        """Test that different ID values are not equal."""
        from core.shared.models import FrozenModel

        class TestModel(FrozenModel):
            pilot_id: PilotIdField

        model1 = TestModel(pilot_id="p1")
        model2 = TestModel(pilot_id="p2")

        assert model1.pilot_id != model2.pilot_id


class TestIdExports:
    """Test that all ID field types are exported."""

    def test_all_fields_exported(self):
        """Verify all convenience field types are in __all__."""
        from core.shared import id_helpers

        expected_exports = [
            "IdField",
            "PilotIdField",
            "MechIdField",
            "WeaponIdField",
            "SystemIdField",
            "CombatantIdField",
            "ActionIdField",
            "EffectIdField",
            "DeployableIdField",
            "DroneIdField",
            "TemplateIdField",
        ]

        for export in expected_exports:
            assert hasattr(id_helpers, export), f"Missing export: {export}"


class TestMixedIdTypes:
    """Test models using multiple different ID types."""

    def test_combat_action_with_multiple_ids(self):
        """Test a model representing a combat action with multiple ID types."""

        class CombatAction(BaseModel):
            attacker_id: CombatantIdField
            target_id: CombatantIdField
            weapon_id: WeaponIdField
            action_id: ActionIdField

        action = CombatAction(
            attacker_id="c_player_1",
            target_id="c_enemy_1",
            weapon_id="w_rifle",
            action_id="a_attack",
        )

        assert action.attacker_id == "c_player_1"
        assert action.target_id == "c_enemy_1"
        assert action.weapon_id == "w_rifle"
        assert action.action_id == "a_attack"

    def test_drone_deployment(self):
        """Test a model for drone deployment."""

        class DroneDeployment(BaseModel):
            drone_id: DroneIdField
            owner_id: CombatantIdField
            deployable_id: DeployableIdField
            activation_action_id: ActionIdField

        deployment = DroneDeployment(
            drone_id="dr_turret_alpha",
            owner_id="c_player_1",
            deployable_id="d_turret_alpha",
            activation_action_id="a_deploy",
        )

        assert deployment.drone_id == "dr_turret_alpha"
        assert deployment.owner_id == "c_player_1"
        assert deployment.deployable_id == "d_turret_alpha"


class TestCombatHelpersTypedIds:
    """Test typed IDs in combat_helpers.py models."""

    def test_attack_sequence_input_typed_ids(self):
        """Test AttackSequenceInput with typed IDs."""
        from core.shared.combat_helpers import AttackSequenceInput

        input_model = AttackSequenceInput(
            attacker_id="c_player_1",
            target_ids=["c_enemy_1", "c_enemy_2"],
            attack_bonus=5,
            defense_value=10,
            base_damage=6,
            damage_type="kinetic",
        )

        assert input_model.attacker_id == "c_player_1"
        assert len(input_model.target_ids) == 2

    def test_attack_sequence_input_drone_id(self):
        """Test AttackSequenceInput with optional drone_id."""
        from core.shared.combat_helpers import AttackSequenceInput

        input_model = AttackSequenceInput(
            attacker_id="c_player_1",
            target_ids=["c_enemy_1"],
            attack_bonus=3,
            defense_value=10,
            base_damage=4,
            damage_type="energy",
            drone_assisted=True,
            drone_id="dr_turret_1",
        )

        assert input_model.drone_assisted is True
        assert input_model.drone_id == "dr_turret_1"

    def test_movement_input_typed_id(self):
        """Test MovementInput with typed mover_id."""
        from core.shared.combat_helpers import MovementInput

        input_model = MovementInput(
            mover_id="c_player_1",
            start_position=(0, 0),
            end_position=(3, 0),
            speed=5,
        )

        assert input_model.mover_id == "c_player_1"

    def test_movement_result_typed_id(self):
        """Test MovementResult with typed mover_id."""
        from core.shared.combat_helpers import MovementResult

        result = MovementResult(
            mover_id="c_player_1",
            start=(0, 0),
            end=(3, 0),
            spaces_moved=3,
            path_valid=True,
            path_hexes=[(0, 0), (1, 0), (2, 0), (3, 0)],
        )

        assert result.mover_id == "c_player_1"

    def test_full_action_turn_input_typed_id(self):
        """Test FullActionTurnInput with typed actor_id."""
        from core.shared.combat_helpers import FullActionTurnInput

        input_model = FullActionTurnInput(
            actor_id="c_player_1",
            position=(0, 0),
            action_choice="full",
            full_action_type="attack",
        )

        assert input_model.actor_id == "c_player_1"

    def test_full_action_turn_result_typed_id(self):
        """Test FullActionTurnResult with typed actor_id."""
        from core.shared.combat_helpers import FullActionTurnResult

        result = FullActionTurnResult(
            actor_id="c_player_1",
            action_type="full",
        )

        assert result.actor_id == "c_player_1"

    def test_turret_drone_attack_input_typed_ids(self):
        """Test TurretDroneAttackInput with typed IDs."""
        from core.shared.combat_helpers import TurretDroneAttackInput

        input_model = TurretDroneAttackInput(
            drone_id="dr_turret_alpha",
            owner_id="c_player_1",
            drone_position=(1, 1),
            ally_attack_hit=True,
            ally_id="c_player_2",
            ally_position=(2, 2),
            target_id="c_enemy_1",
            target_position=(3, 3),
        )

        assert input_model.drone_id == "dr_turret_alpha"
        assert input_model.owner_id == "c_player_1"
        assert input_model.ally_id == "c_player_2"
        assert input_model.target_id == "c_enemy_1"

    def test_latch_drone_input_typed_ids(self):
        """Test LatchDroneInput with typed IDs."""
        from core.shared.combat_helpers import LatchDroneInput

        input_model = LatchDroneInput(
            drone_id="dr_latch_1",
            owner_id="c_player_1",
            target_mech_id="m_enemy_everest",
            mode="mount",
        )

        assert input_model.drone_id == "dr_latch_1"
        assert input_model.owner_id == "c_player_1"
        assert input_model.target_mech_id == "m_enemy_everest"

    def test_restock_drone_input_typed_ids(self):
        """Test RestockDroneInput with typed IDs."""
        from core.shared.combat_helpers import RestockDroneInput

        input_model = RestockDroneInput(
            drone_id="dr_restock_1",
            owner_id="c_player_1",
            activating_combatant_id="c_player_2",
            activating_combatant_position=(0, 0),
            action_choice="cool",
        )

        assert input_model.drone_id == "dr_restock_1"
        assert input_model.owner_id == "c_player_1"
        assert input_model.activating_combatant_id == "c_player_2"

    def test_iceout_drone_input_typed_ids(self):
        """Test ICEOUTDroneInput with typed IDs."""
        from core.shared.combat_helpers import ICEOUTDroneInput

        input_model = ICEOUTDroneInput(
            drone_id="dr_iceout_1",
            owner_id="c_player_1",
            drone_position=(1, 1),
            zone_target_ids=["c_enemy_1", "c_enemy_2"],
        )

        assert input_model.drone_id == "dr_iceout_1"
        assert input_model.owner_id == "c_player_1"
        assert len(input_model.zone_target_ids) == 2

    def test_tracking_drone_input_typed_ids(self):
        """Test TrackingDroneInput with typed IDs."""
        from core.shared.combat_helpers import TrackingDroneInput

        input_model = TrackingDroneInput(
            drone_id="dr_tracking_1",
            owner_id="c_player_1",
            drone_position=(1, 1),
            target_id="c_enemy_1",
            target_position=(3, 3),
        )

        assert input_model.drone_id == "dr_tracking_1"
        assert input_model.owner_id == "c_player_1"
        assert input_model.target_id == "c_enemy_1"

    def test_hive_drone_input_typed_ids(self):
        """Test HiveDroneInput with typed IDs."""
        from core.shared.combat_helpers import HiveDroneInput

        input_model = HiveDroneInput(
            drone_id="dr_hive_1",
            owner_id="c_player_1",
            drone_position=(1, 1),
            zone_target_ids=["c_enemy_1", "c_enemy_2", "c_enemy_3"],
        )

        assert input_model.drone_id == "dr_hive_1"
        assert input_model.owner_id == "c_player_1"
        assert len(input_model.zone_target_ids) == 3

    def test_geometry_validation_result_typed_ids(self):
        """Test GeometryValidationResult with typed affected_target_ids."""
        from core.shared.combat_helpers import GeometryValidationResult
        from core.shared.combat_helpers import CombatantId

        result = GeometryValidationResult(
            is_valid=True,
            affected_target_ids=[CombatantId("c_enemy_1"), CombatantId("c_enemy_2")],
            reason="Burst 1 pattern",
        )

        assert result.is_valid is True
        assert len(result.affected_target_ids) == 2


class TestEffectsSystemTypedIds:
    """Test typed IDs in effects/core.py models."""

    def test_reaction_condition_reaction_id(self):
        """Test ReactionCondition with typed reaction_id."""
        from core.shared.effects import ReactionCondition

        condition = ReactionCondition(reaction_id="brace")
        assert condition.reaction_id == "brace"

    def test_reaction_condition_is_attack_only(self):
        """Test ReactionCondition with only is_attack set."""
        from core.shared.effects import ReactionCondition

        condition = ReactionCondition(is_attack=True)
        assert condition.is_attack is True
        assert condition.reaction_id is None

    def test_reaction_trigger_effect_reaction_id(self):
        """Test ReactionTriggerEffect with typed reaction_id."""
        from core.shared.effects import ReactionTriggerEffect

        effect = ReactionTriggerEffect(
            reaction_id="overwatch",
            trigger_events=["enemy_enters_threat"],
        )
        assert effect.reaction_id == "overwatch"

    def test_dice_pool_effect_weapon_id(self):
        """Test DicePoolEffect with typed weapon_id."""
        from core.shared.effects import DicePoolEffect

        effect = DicePoolEffect(
            pool_name="test_pool",
            weapon_id="w_heavy_cannon",
        )
        assert effect.weapon_id == "w_heavy_cannon"

    def test_weapon_grant_effect_weapon_id(self):
        """Test WeaponGrantEffect with typed weapon_id."""
        from core.shared.effects import WeaponGrantEffect

        effect = WeaponGrantEffect(
            weapon_id="w_heavy_cannon",
            name="Heavy Cannon",
            size="heavy",
            weapon_type="cqb",
        )
        assert effect.weapon_id == "w_heavy_cannon"

    def test_weapon_grant_effect_no_weapon_id(self):
        """Test WeaponGrantEffect with no weapon_id."""
        from core.shared.effects import WeaponGrantEffect

        effect = WeaponGrantEffect(
            name="Aux Rifle",
            size="aux",
            weapon_type="rifle",
        )
        assert effect.weapon_id is None

    def test_mode_effect_action_ids(self):
        """Test ModeEffect with typed activation/deactivation action IDs."""
        from core.shared.effects import ModeEffect

        effect = ModeEffect(
            name="Reserve Power Mode",
            activation_action_id="shutdown",
            activation_action_type="quick",
            deactivation_action_id="boot_up",
            deactivation_action_type="full",
        )
        assert effect.activation_action_id == "shutdown"
        assert effect.deactivation_action_id == "boot_up"

    def test_mode_effect_only_activation(self):
        """Test ModeEffect with only activation action ID."""
        from core.shared.effects import ModeEffect

        effect = ModeEffect(
            name="Single Mode",
            activation_action_id="activate",
            activation_action_type="quick",
        )
        assert effect.activation_action_id == "activate"
        assert effect.deactivation_action_id is None

    def test_progression_state_target_id(self):
        """Test ProgressionState with typed target_id."""
        from core.shared.effects import ProgressionState

        state = ProgressionState(
            current_gate=2,
            max_gate=4,
            target_id="c_enemy_1",
        )
        assert state.target_id == "c_enemy_1"

    def test_per_target_counter_typed_ids(self):
        """Test PerTargetCounter with typed effect_id and target_id."""
        from core.shared.effects import PerTargetCounter

        counter = PerTargetCounter(
            effect_id="basilisk_stun",
            max_count=1,
            target_id="c_enemy_1",
        )
        assert counter.effect_id == "basilisk_stun"
        assert counter.target_id == "c_enemy_1"

    def test_per_target_counter_effect_typed_id(self):
        """Test PerTargetCounterEffect with typed effect_id."""
        from core.shared.effects import PerTargetCounterEffect
        from core.shared.effects import MechanicalEffect

        effect = PerTargetCounterEffect(
            effect_id="basilisk_stun",
            max_count=1,
            effect=MechanicalEffect(),
        )
        assert effect.effect_id == "basilisk_stun"

    def test_cooldown_state_typed_ids(self):
        """Test CooldownState with typed effect_id and target_id."""
        from core.shared.effects import CooldownState

        state = CooldownState(
            effect_id="ability_cooldown",
            duration=1,
            target_id="c_enemy_1",
        )
        assert state.effect_id == "ability_cooldown"
        assert state.target_id == "c_enemy_1"
