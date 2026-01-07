"""
Tests for Drone Turn Integration Module

Tests all turn-phase integration functions per PR2 rules:
- Start of turn processing
- End of turn processing
- Drone movement on owner's turn
- Reaction trigger checks
- Zone management for ICEOUT and Hive drones
"""

import unittest
from core.shared.drone_turn import (
    DroneTurnPhase,
    DroneType,
    DroneReactionTrigger,
    DroneTurnStartInput,
    DroneTurnStartResult,
    resolve_drone_turn_start,
    DroneTurnEndInput,
    DroneTurnEndResult,
    resolve_drone_turn_end,
    DroneMovementInput,
    DroneMovementResult,
    resolve_drone_movement,
    DroneReactionCheckInput,
    DroneReactionCheckResult,
    resolve_drone_reaction_check,
    DroneTurnInput,
    DroneTurnResult,
    resolve_drone_turn,
    DroneZoneManagementInput,
    DroneZoneManagementResult,
    resolve_drone_zone_management,
)
from core.shared.models import FrozenModel
from core.shared.heat import MeltdownState
from core.mech.grid import HexCoord, HexPosition
from core.mech.combat_state import (
    DeployableState,
    DeployableKind,
    MechCombatScenario,
)


def make_test_scenario() -> MechCombatScenario:
    """Create a basic test scenario."""
    return MechCombatScenario(
        combatants=[],
        terrain=None,
        deployables={},
    )


def make_test_drones(
    can_act: bool = True,
    can_move: bool = True,
    is_destroyed: bool = False,
) -> dict[str, DeployableState]:
    """Create sample drones for testing."""
    return {
        "turret-1": DeployableState(
            id="turret-1",
            name="Turret Drone",
            kind="drone",
            owner_id="mech-1",
            position=HexPosition(coord=HexCoord(q=5, r=5)),
            size=1,
            hp=10,
            max_hp=10,
            evasion=10,
            e_defense=10,
            is_active=True,
            can_act=can_act,
            can_move=can_move,
            acts_on_owner_turn=True,
            is_destroyed=is_destroyed,
        ),
    }


class TestDroneTurnStartInput(unittest.TestCase):
    """Tests for DroneTurnStartInput model."""

    def _make_deployed_drones(self) -> dict[str, DeployableState]:
        """Create sample deployed drones for testing."""
        return {
            "turret-1": DeployableState(
                id="turret-1",
                name="Turret Drone",
                kind="drone",
                owner_id="mech-1",
                position=HexPosition(coord=HexCoord(q=5, r=5)),
                size=1,
                hp=10,
                max_hp=10,
                evasion=10,
                e_defense=10,
                is_active=True,
                can_act=True,
                can_move=True,
                acts_on_owner_turn=True,
            ),
            "restock-1": DeployableState(
                id="restock-1",
                name="Restock Drone",
                kind="drone",
                owner_id="mech-1",
                position=HexPosition(coord=HexCoord(q=6, r=5)),
                size=1,
                hp=10,
                max_hp=10,
                evasion=10,
                e_defense=10,
                is_active=True,
                can_act=False,
                can_move=True,
                acts_on_owner_turn=True,
                is_armed=False,
            ),
        }

    def test_basic_input_creation(self):
        """Test basic DroneTurnStartInput creation."""
        drones = self._make_deployed_drones()
        input_data = DroneTurnStartInput(
            owner_id="mech-1",
            deployed_drones=drones,
            current_turn=3,
            tier=2,
        )
        self.assertEqual(input_data.owner_id, "mech-1")
        self.assertEqual(input_data.current_turn, 3)
        self.assertEqual(input_data.tier, 2)
        self.assertEqual(len(input_data.deployed_drones), 2)

    def test_latch_active_mode_input(self):
        """Test input with latch drone in active mode."""
        input_data = DroneTurnStartInput(
            owner_id="mech-1",
            deployed_drones={},
            current_turn=1,
            latch_drone_active=True,
            latch_drone_target_id="mech-2",
        )
        self.assertTrue(input_data.latch_drone_active)
        self.assertEqual(input_data.latch_drone_target_id, "mech-2")


class TestDroneTurnStartResult(unittest.TestCase):
    """Tests for DroneTurnStartResult model."""

    def test_basic_result_creation(self):
        """Test basic DroneTurnStartResult creation."""
        result = DroneTurnStartResult(
            drones_ready_to_act=["turret-1"],
            drones_needing_movement=["restock-1"],
            heat_to_owner=1,
            accuracy_bonus=1,
        )
        self.assertEqual(len(result.drones_ready_to_act), 1)
        self.assertEqual(result.drones_ready_to_act[0], "turret-1")
        self.assertEqual(result.heat_to_owner, 1)
        self.assertEqual(result.accuracy_bonus, 1)

    def test_latch_effects_result(self):
        """Test result with latch active effects."""
        result = DroneTurnStartResult(
            drones_ready_to_act=[],
            drones_needing_movement=[],
            latch_active_effects={"accuracy_bonus": True, "condition_immunity": True},
            conditions_immunized=[
                "impaired",
                "jammed",
                "slowed",
                "shredded",
                "immobilized",
            ],
            accuracy_bonus=1,
        )
        self.assertIn("accuracy_bonus", result.latch_active_effects)
        self.assertIn("condition_immunity", result.latch_active_effects)
        self.assertEqual(len(result.conditions_immunized), 5)


class TestResolveDroneTurnStart(unittest.TestCase):
    """Tests for resolve_drone_turn_start function."""

    def test_no_drones(self):
        """Test turn start with no deployed drones."""
        result = resolve_drone_turn_start(
            DroneTurnStartInput(
                owner_id="mech-1",
                deployed_drones={},
                current_turn=1,
            )
        )
        self.assertEqual(len(result.drones_ready_to_act), 0)
        self.assertEqual(len(result.drones_needing_movement), 0)
        self.assertEqual(result.heat_to_owner, 0)

    def test_drones_ready_to_act(self):
        """Test drones that can act are identified."""
        drones = make_test_drones(can_act=True, can_move=True)
        result = resolve_drone_turn_start(
            DroneTurnStartInput(
                owner_id="mech-1",
                deployed_drones=drones,
                current_turn=1,
            )
        )
        self.assertIn("turret-1", result.drones_ready_to_act)
        self.assertIn("turret-1", result.drones_needing_movement)

    def test_destroyed_drones_excluded(self):
        """Test destroyed drones are excluded from ready list."""
        drones = make_test_drones(can_act=True, is_destroyed=True)
        result = resolve_drone_turn_start(
            DroneTurnStartInput(
                owner_id="mech-1",
                deployed_drones=drones,
                current_turn=1,
            )
        )
        self.assertNotIn("turret-1", result.drones_ready_to_act)
        self.assertIn("turret-1", result.drones_to_deactivate)

    def test_latch_active_mode_heat(self):
        """Test active latch mode adds 1 heat to owner."""
        result = resolve_drone_turn_start(
            DroneTurnStartInput(
                owner_id="mech-1",
                deployed_drones={},
                current_turn=1,
                latch_drone_active=True,
                latch_drone_target_id="mech-2",
            )
        )
        self.assertEqual(result.heat_to_owner, 1)
        self.assertIn("accuracy_bonus", result.latch_active_effects)
        self.assertIn("condition_immunity", result.latch_active_effects)
        self.assertEqual(result.accuracy_bonus, 1)

    def test_no_latch_mode_no_heat(self):
        """Test no heat when latch is not active."""
        result = resolve_drone_turn_start(
            DroneTurnStartInput(
                owner_id="mech-1",
                deployed_drones={},
                current_turn=1,
                latch_drone_active=False,
            )
        )
        self.assertEqual(result.heat_to_owner, 0)
        self.assertEqual(result.accuracy_bonus, 0)


class TestDroneTurnEndInput(unittest.TestCase):
    """Tests for DroneTurnEndInput model."""

    def _make_deployed_drones(self) -> dict[str, DeployableState]:
        """Create sample deployed drones for testing."""
        return {
            "restock-1": DeployableState(
                id="restock-1",
                name="Restock Drone",
                kind="drone",
                owner_id="mech-1",
                position=HexPosition(coord=HexCoord(q=6, r=5)),
                size=1,
                hp=10,
                max_hp=10,
                evasion=10,
                e_defense=10,
                is_active=True,
                can_act=False,
                can_move=True,
                acts_on_owner_turn=True,
                is_armed=False,
            ),
        }

    def test_basic_input_creation(self):
        """Test basic DroneTurnEndInput creation."""
        drones = self._make_deployed_drones()
        input_data = DroneTurnEndInput(
            owner_id="mech-1",
            deployed_drones=drones,
            current_turn=3,
            tier=2,
        )
        self.assertEqual(input_data.owner_id, "mech-1")
        self.assertEqual(input_data.current_turn, 3)
        self.assertEqual(len(input_data.deployed_drones), 1)

    def test_stunned_owner_input(self):
        """Test input with stunned owner."""
        input_data = DroneTurnEndInput(
            owner_id="mech-1",
            deployed_drones={},
            current_turn=1,
            owner_is_stunned=True,
            latch_drone_active=True,
        )
        self.assertTrue(input_data.owner_is_stunned)


class TestDroneTurnEndResult(unittest.TestCase):
    """Tests for DroneTurnEndResult model."""

    def test_basic_result_creation(self):
        """Test basic DroneTurnEndResult creation."""
        result = DroneTurnEndResult(
            drones_to_prime=["restock-1"],
            latch_mode_end=False,
        )
        self.assertIn("restock-1", result.drones_to_prime)
        self.assertFalse(result.latch_mode_end)

    def test_latch_mode_ends(self):
        """Test latch mode ending."""
        result = DroneTurnEndResult(
            drones_to_prime=[],
            latch_mode_end=True,
        )
        self.assertTrue(result.latch_mode_end)


class TestResolveDroneTurnEnd(unittest.TestCase):
    """Tests for resolve_drone_turn_end function."""

    def _make_drones(
        self,
        is_destroyed: bool = False,
        is_active: bool = True,
    ) -> dict[str, DeployableState]:
        """Create sample drones for testing."""
        return {
            "turret-1": DeployableState(
                id="turret-1",
                name="Turret Drone",
                kind="drone",
                owner_id="mech-1",
                position=HexPosition(coord=HexCoord(q=5, r=5)),
                size=1,
                hp=10,
                max_hp=10,
                evasion=10,
                e_defense=10,
                is_active=is_active,
                can_act=True,
                can_move=True,
                acts_on_owner_turn=True,
                is_destroyed=is_destroyed,
            ),
        }

    def test_no_drones(self):
        """Test turn end with no deployed drones."""
        result = resolve_drone_turn_end(
            DroneTurnEndInput(
                owner_id="mech-1",
                deployed_drones={},
                current_turn=1,
            )
        )
        self.assertEqual(len(result.drones_to_prime), 0)
        self.assertFalse(result.latch_mode_end)

    def test_destroyed_drones_deactivated(self):
        """Test destroyed drones are deactivated."""
        drones = self._make_drones(is_destroyed=True)
        result = resolve_drone_turn_end(
            DroneTurnEndInput(
                owner_id="mech-1",
                deployed_drones=drones,
                current_turn=1,
            )
        )
        self.assertIn("turret-1", result.drones_to_deactivate)

    def test_latch_mode_ends_on_stun(self):
        """Test active latch mode ends when owner is stunned."""
        result = resolve_drone_turn_end(
            DroneTurnEndInput(
                owner_id="mech-1",
                deployed_drones={},
                current_turn=1,
                owner_is_stunned=True,
                latch_drone_active=True,
                latch_drone_target_id="mech-2",
            )
        )
        self.assertTrue(result.latch_mode_end)

    def test_latch_mode_continues_without_stun(self):
        """Test active latch mode continues without stun."""
        result = resolve_drone_turn_end(
            DroneTurnEndInput(
                owner_id="mech-1",
                deployed_drones={},
                current_turn=1,
                owner_is_stunned=False,
                latch_drone_active=True,
                latch_drone_target_id="mech-2",
            )
        )
        self.assertFalse(result.latch_mode_end)


class TestDroneMovementInput(unittest.TestCase):
    """Tests for DroneMovementInput model."""

    def test_basic_input_creation(self):
        """Test basic DroneMovementInput creation."""
        scenario = make_test_scenario()
        input_data = DroneMovementInput(
            drone_id="turret-1",
            current_scenario=scenario,
            destination=HexPosition(coord=HexCoord(q=7, r=5)),
            drone_speed=4,
        )
        self.assertEqual(input_data.drone_id, "turret-1")
        self.assertEqual(input_data.drone_speed, 4)

    def test_force_movement_cost(self):
        """Test input with forced movement cost."""
        scenario = make_test_scenario()
        input_data = DroneMovementInput(
            drone_id="turret-1",
            current_scenario=scenario,
            destination=HexPosition(coord=HexCoord(q=7, r=5)),
            drone_speed=4,
            force_movement_cost=3,
        )
        self.assertEqual(input_data.force_movement_cost, 3)


class TestDroneMovementResult(unittest.TestCase):
    """Tests for DroneMovementResult model."""

    def test_basic_result_creation(self):
        """Test basic DroneMovementResult creation."""
        result = DroneMovementResult(
            movement_successful=True,
            path_clear=True,
            spaces_moved=2,
            total_movement_cost=2,
            new_position=HexPosition(coord=HexCoord(q=7, r=5)),
        )
        self.assertTrue(result.movement_successful)
        self.assertEqual(result.spaces_moved, 2)

    def test_failed_movement_result(self):
        """Test movement failure result."""
        result = DroneMovementResult(
            movement_successful=False,
            path_clear=False,
            spaces_moved=0,
            total_movement_cost=5,
            reason="Movement cost exceeds speed",
        )
        self.assertFalse(result.movement_successful)
        self.assertIsNone(result.new_position)


class TestResolveDroneMovement(unittest.TestCase):
    """Tests for resolve_drone_movement function."""

    def test_drone_not_found(self):
        """Test movement when drone is not found."""
        scenario = make_test_scenario()
        result = resolve_drone_movement(
            DroneMovementInput(
                drone_id="turret-1",
                current_scenario=scenario,
                destination=HexPosition(coord=HexCoord(q=7, r=5)),
                drone_speed=4,
            )
        )
        self.assertFalse(result.movement_successful)
        self.assertIn("not found", result.reason)

    def test_successful_movement(self):
        """Test successful drone movement."""
        scenario = make_test_scenario()
        from core.mech.combat_state import (
            CombatantState,
            CombatStats,
            CombatResources,
            MechInventory,
        )

        drone = CombatantState(
            id="turret-1",
            name="Turret Drone",
            kind="mech",
            side="players",
            position=HexPosition(coord=HexCoord(q=5, r=5)),
            stats=CombatStats(
                size="size_half",
                hp_max=10,
                evasion=10,
                e_defense=10,
                sensor_range=5,
            ),
            resources=CombatResources(hp_current=10),
            inventory=MechInventory(mounts=[], systems=[]),
            statuses=[],
            conditions=[],
        )
        scenario = MechCombatScenario(
            combatants=[drone],
            terrain=None,
            deployables={},
        )
        result = resolve_drone_movement(
            DroneMovementInput(
                drone_id="turret-1",
                current_scenario=scenario,
                destination=HexPosition(coord=HexCoord(q=7, r=5)),
                drone_speed=4,
                force_movement_cost=2,
            )
        )
        self.assertTrue(result.movement_successful)
        self.assertEqual(result.spaces_moved, 2)
        self.assertEqual(result.total_movement_cost, 2)

    def test_movement_too_far(self):
        """Test movement that exceeds drone speed."""
        scenario = make_test_scenario()
        result = resolve_drone_movement(
            DroneMovementInput(
                drone_id="turret-1",
                current_scenario=scenario,
                destination=HexPosition(coord=HexCoord(q=15, r=5)),
                drone_speed=4,
            )
        )
        self.assertFalse(result.movement_successful)
        self.assertEqual(result.spaces_moved, 0)


class TestDroneReactionCheckInput(unittest.TestCase):
    """Tests for DroneReactionCheckInput model."""

    def test_turret_trigger_input(self):
        """Test input for turret drone reaction trigger."""
        input_data = DroneReactionCheckInput(
            trigger_type="ally_hit_within_range",
            drone_id="turret-1",
            deployed_drones={},
            current_scenario=make_test_scenario(),
            ally_id="mech-2",
            ally_position=HexPosition(coord=HexCoord(q=5, r=6)),
            target_id="enemy-1",
            target_position=HexPosition(coord=HexCoord(q=5, r=7)),
            ally_attack_hit=True,
        )
        self.assertEqual(input_data.trigger_type, "ally_hit_within_range")
        self.assertEqual(input_data.drone_id, "turret-1")
        self.assertTrue(input_data.ally_attack_hit)

    def test_restock_adjacent_start_trigger(self):
        """Test input for restock drone adjacent start trigger."""
        input_data = DroneReactionCheckInput(
            trigger_type="ally_adjacent_start",
            drone_id="restock-1",
            deployed_drones={},
            current_scenario=make_test_scenario(),
            ally_id="mech-2",
            ally_position=HexPosition(coord=HexCoord(q=6, r=5)),
        )
        self.assertEqual(input_data.trigger_type, "ally_adjacent_start")


class TestDroneReactionCheckResult(unittest.TestCase):
    """Tests for DroneReactionCheckResult model."""

    def test_reaction_available(self):
        """Test result when reaction is available."""
        result = DroneReactionCheckResult(
            reaction_available=True,
            drone_id="turret-1",
            drone_type="turret",
            activation_ready=True,
            range_check=True,
            conditions_met=["ally_within_range_10", "ally_attack_hit"],
        )
        self.assertTrue(result.reaction_available)
        self.assertEqual(result.drone_type, "turret")

    def test_reaction_not_available(self):
        """Test result when reaction is not available."""
        result = DroneReactionCheckResult(
            reaction_available=False,
            drone_id="turret-1",
            conditions_failed=["ally_out_of_range"],
            reason="Ally out of range",
        )
        self.assertFalse(result.reaction_available)
        self.assertIn("ally_out_of_range", result.conditions_failed)


class TestResolveDroneReactionCheck(unittest.TestCase):
    """Tests for resolve_drone_reaction_check function."""

    def test_drone_not_found(self):
        """Test reaction check when drone is not found."""
        result = resolve_drone_reaction_check(
            DroneReactionCheckInput(
                trigger_type="ally_hit_within_range",
                drone_id="turret-1",
                deployed_drones={},
                current_scenario=make_test_scenario(),
            )
        )
        self.assertFalse(result.reaction_available)
        self.assertIn("drone_not_found", result.conditions_failed)

    def test_drone_destroyed(self):
        """Test reaction check when drone is destroyed."""
        drones = make_test_drones()
        drones["turret-1"] = DeployableState(
            id="turret-1",
            name="Turret Drone",
            kind="drone",
            owner_id="mech-1",
            position=HexPosition(coord=HexCoord(q=5, r=5)),
            size=1,
            hp=0,
            max_hp=10,
            is_destroyed=True,
        )
        result = resolve_drone_reaction_check(
            DroneReactionCheckInput(
                trigger_type="ally_hit_within_range",
                drone_id="turret-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
            )
        )
        self.assertFalse(result.reaction_available)
        self.assertIn("drone_destroyed", result.conditions_failed)

    def test_turret_ally_hit_within_range(self):
        """Test turret reaction when ally hits within range."""
        drones = make_test_drones()
        result = resolve_drone_reaction_check(
            DroneReactionCheckInput(
                trigger_type="ally_hit_within_range",
                drone_id="turret-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
                ally_position=HexPosition(coord=HexCoord(q=5, r=6)),
                target_position=HexPosition(coord=HexCoord(q=5, r=7)),
                ally_attack_hit=True,
            )
        )
        self.assertTrue(result.reaction_available)
        self.assertIn("ally_within_range_10", result.conditions_met)
        self.assertTrue(result.range_check)

    def test_turret_ally_hit_out_of_range(self):
        """Test turret reaction when ally is out of range."""
        drones = make_test_drones()
        result = resolve_drone_reaction_check(
            DroneReactionCheckInput(
                trigger_type="ally_hit_within_range",
                drone_id="turret-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
                ally_position=HexPosition(coord=HexCoord(q=20, r=20)),
                target_position=HexPosition(coord=HexCoord(q=20, r=21)),
                ally_attack_hit=True,
            )
        )
        self.assertFalse(result.reaction_available)
        self.assertIn("ally_out_of_range", result.conditions_failed)

    def test_turret_ally_missed(self):
        """Test turret cannot react when ally misses."""
        drones = make_test_drones()
        result = resolve_drone_reaction_check(
            DroneReactionCheckInput(
                trigger_type="ally_hit_within_range",
                drone_id="turret-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
                ally_position=HexPosition(coord=HexCoord(q=5, r=6)),
                target_position=HexPosition(coord=HexCoord(q=5, r=7)),
                ally_attack_hit=False,
            )
        )
        self.assertFalse(result.reaction_available)
        self.assertIn("ally_attack_missed", result.conditions_failed)

    def test_restock_ally_adjacent(self):
        """Test restock drone when ally is adjacent."""
        drones = make_test_drones()
        drones["restock-1"] = DeployableState(
            id="restock-1",
            name="Restock Drone",
            kind="drone",
            owner_id="mech-1",
            position=HexPosition(coord=HexCoord(q=6, r=5)),
            size=1,
            hp=10,
            max_hp=10,
            evasion=10,
            e_defense=10,
            is_active=True,
            is_armed=True,
        )
        result = resolve_drone_reaction_check(
            DroneReactionCheckInput(
                trigger_type="ally_adjacent_start",
                drone_id="restock-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
                ally_position=HexPosition(coord=HexCoord(q=6, r=5)),
            )
        )
        self.assertTrue(result.reaction_available)
        self.assertIn("ally_adjacent", result.conditions_met)
        self.assertTrue(result.activation_ready)

    def test_restock_ally_not_adjacent(self):
        """Test restock drone when ally is not adjacent."""
        drones = make_test_drones()
        drones["restock-1"] = DeployableState(
            id="restock-1",
            name="Restock Drone",
            kind="drone",
            owner_id="mech-1",
            position=HexPosition(coord=HexCoord(q=6, r=5)),
            size=1,
            hp=10,
            max_hp=10,
            evasion=10,
            e_defense=10,
            is_active=True,
            is_armed=True,
        )
        result = resolve_drone_reaction_check(
            DroneReactionCheckInput(
                trigger_type="ally_adjacent_start",
                drone_id="restock-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
                ally_position=HexPosition(coord=HexCoord(q=10, r=10)),
            )
        )
        self.assertFalse(result.reaction_available)
        self.assertIn("ally_not_adjacent", result.conditions_failed)


class TestDroneTurnInput(unittest.TestCase):
    """Tests for DroneTurnInput model."""

    def test_basic_input_creation(self):
        """Test basic DroneTurnInput creation."""
        drones = make_test_drones()
        input_data = DroneTurnInput(
            owner_id="mech-1",
            deployed_drones=drones,
            current_turn=1,
        )
        self.assertEqual(input_data.owner_id, "mech-1")
        self.assertEqual(input_data.current_turn, 1)
        self.assertEqual(len(input_data.deployed_drones), 1)

    def test_latch_active_mode_input(self):
        """Test input with active latch mode."""
        input_data = DroneTurnInput(
            owner_id="mech-1",
            deployed_drones={},
            current_turn=5,
            tier=2,
            owner_is_stunned=False,
            latch_drone_active=True,
            latch_drone_target_id="mech-2",
        )
        self.assertTrue(input_data.latch_drone_active)
        self.assertEqual(input_data.latch_drone_target_id, "mech-2")


class TestDroneTurnResult(unittest.TestCase):
    """Tests for DroneTurnResult model."""

    def test_basic_result_creation(self):
        """Test basic DroneTurnResult creation."""
        start_result = DroneTurnStartResult(
            drones_ready_to_act=["turret-1"],
            drones_needing_movement=["turret-1"],
        )
        end_result = DroneTurnEndResult()
        result = DroneTurnResult(
            start_result=start_result,
            end_result=end_result,
            total_heat_to_owner=1,
            drones_ready=["turret-1"],
            drones_can_move=["turret-1"],
        )
        self.assertEqual(result.total_heat_to_owner, 1)
        self.assertIn("turret-1", result.drones_ready)


class TestResolveDroneTurn(unittest.TestCase):
    """Tests for resolve_drone_turn function."""

    def test_full_turn_processing(self):
        """Test complete drone turn processing."""
        drones = make_test_drones()
        result = resolve_drone_turn(
            DroneTurnInput(
                owner_id="mech-1",
                deployed_drones=drones,
                current_turn=1,
            )
        )
        self.assertIn("turret-1", result.drones_ready)
        self.assertIn("turret-1", result.drones_can_move)
        self.assertEqual(result.total_heat_to_owner, 0)

    def test_latch_active_mode_heat(self):
        """Test drone turn with active latch mode adds heat."""
        drones = make_test_drones()
        result = resolve_drone_turn(
            DroneTurnInput(
                owner_id="mech-1",
                deployed_drones=drones,
                current_turn=1,
                latch_drone_active=True,
                latch_drone_target_id="mech-2",
            )
        )
        self.assertEqual(result.total_heat_to_owner, 1)


class TestDroneZoneManagementInput(unittest.TestCase):
    """Tests for DroneZoneManagementInput model."""

    def test_iceout_zone_input(self):
        """Test ICEOUT zone management input."""
        input_data = DroneZoneManagementInput(
            drone_id="iceout-1",
            deployed_drones={},
            current_scenario=make_test_scenario(),
        )
        self.assertEqual(input_data.drone_id, "iceout-1")

    def test_move_zone_input(self):
        """Test zone movement input."""
        input_data = DroneZoneManagementInput(
            drone_id="iceout-1",
            deployed_drones={},
            current_scenario=make_test_scenario(),
            new_position=HexPosition(coord=HexCoord(q=10, r=5)),
        )
        self.assertIsNotNone(input_data.new_position)


class TestDroneZoneManagementResult(unittest.TestCase):
    """Tests for DroneZoneManagementResult model."""

    def test_iceout_zone_result(self):
        """Test ICEOUT zone result."""
        result = DroneZoneManagementResult(
            zone_type="iceout",
            zone_active=True,
            zone_position=HexPosition(coord=HexCoord(q=5, r=5)),
            zone_radius=1,
            effects_applied=["tech_immunity"],
        )
        self.assertEqual(result.zone_type, "iceout")
        self.assertEqual(result.zone_radius, 1)
        self.assertIn("tech_immunity", result.effects_applied)

    def test_hive_zone_result(self):
        """Test Hive zone result."""
        result = DroneZoneManagementResult(
            zone_type="hive",
            zone_active=True,
            zone_position=HexPosition(coord=HexCoord(q=8, r=5)),
            zone_radius=2,
            effects_applied=["soft_cover", "entry_damage"],
        )
        self.assertEqual(result.zone_type, "hive")
        self.assertEqual(result.zone_radius, 2)

    def test_moved_zone_result(self):
        """Test zone that was moved."""
        result = DroneZoneManagementResult(
            zone_type="iceout",
            zone_active=True,
            zone_position=HexPosition(coord=HexCoord(q=10, r=5)),
            zone_radius=1,
            moved=True,
        )
        self.assertTrue(result.moved)


class TestResolveDroneZoneManagement(unittest.TestCase):
    """Tests for resolve_drone_zone_management function."""

    def test_iceout_zone_identified(self):
        """Test ICEOUT zone is correctly identified."""
        drones = {
            "iceout-1": DeployableState(
                id="iceout-1",
                name="ICEOUT Drone",
                kind="drone",
                owner_id="mech-1",
                position=HexPosition(coord=HexCoord(q=5, r=5)),
                size=1,
                hp=10,
                max_hp=10,
                evasion=10,
                e_defense=10,
                is_active=True,
            ),
        }
        result = resolve_drone_zone_management(
            DroneZoneManagementInput(
                drone_id="iceout-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
            )
        )
        self.assertEqual(result.zone_type, "iceout")
        self.assertTrue(result.zone_active)
        self.assertEqual(result.zone_radius, 1)
        self.assertIn("tech_immunity", result.effects_applied)

    def test_hive_zone_identified(self):
        """Test Hive zone is correctly identified."""
        drones = {
            "hive-1": DeployableState(
                id="hive-1",
                name="Hive Drone",
                kind="drone",
                owner_id="mech-1",
                position=HexPosition(coord=HexCoord(q=8, r=5)),
                size=1,
                hp=10,
                max_hp=10,
                evasion=10,
                e_defense=10,
                is_active=True,
            ),
        }
        result = resolve_drone_zone_management(
            DroneZoneManagementInput(
                drone_id="hive-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
            )
        )
        self.assertEqual(result.zone_type, "hive")
        self.assertTrue(result.zone_active)
        self.assertEqual(result.zone_radius, 2)
        self.assertIn("soft_cover", result.effects_applied)
        self.assertIn("entry_damage", result.effects_applied)

    def test_zone_movement(self):
        """Test zone can be moved."""
        drones = {
            "iceout-1": DeployableState(
                id="iceout-1",
                name="ICEOUT Drone",
                kind="drone",
                owner_id="mech-1",
                position=HexPosition(coord=HexCoord(q=5, r=5)),
                size=1,
                hp=10,
                max_hp=10,
                evasion=10,
                e_defense=10,
                is_active=True,
            ),
        }
        result = resolve_drone_zone_management(
            DroneZoneManagementInput(
                drone_id="iceout-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
                new_position=HexPosition(coord=HexCoord(q=10, r=5)),
            )
        )
        self.assertTrue(result.moved)
        self.assertEqual(result.zone_position.coord.q, 10)

    def test_drone_not_found(self):
        """Test zone management when drone is not found."""
        drones = make_test_drones()
        result = resolve_drone_zone_management(
            DroneZoneManagementInput(
                drone_id="nonexistent",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
            )
        )
        self.assertFalse(result.zone_active)

    def test_drone_destroyed(self):
        """Test zone management when drone is destroyed."""
        drones = make_test_drones()
        drones["iceout-1"] = DeployableState(
            id="iceout-1",
            name="ICEOUT Drone",
            kind="drone",
            owner_id="mech-1",
            position=HexPosition(coord=HexCoord(q=5, r=5)),
            size=1,
            hp=0,
            max_hp=10,
            is_destroyed=True,
        )
        result = resolve_drone_zone_management(
            DroneZoneManagementInput(
                drone_id="iceout-1",
                deployed_drones=drones,
                current_scenario=make_test_scenario(),
            )
        )
        self.assertFalse(result.zone_active)

    def test_non_zone_drone(self):
        """Test zone management for non-zone drone."""
        result = resolve_drone_zone_management(
            DroneZoneManagementInput(
                drone_id="turret-1",
                deployed_drones=make_test_drones(),
                current_scenario=make_test_scenario(),
            )
        )
        self.assertFalse(result.zone_active)


if __name__ == "__main__":
    unittest.main()
