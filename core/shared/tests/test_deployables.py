"""
Tests for deployable interactions module.

Tests mine detection/disarm/detonation and drone activation/movement
per PR2 5082-5088 rules.
"""

from __future__ import annotations

from typing import Any
import unittest

from core.shared.deployables import (
    MineType,
    DroneActionType,
    MineEffectProfile,
    get_mine_effect_profile,
    get_default_detection_dc,
    get_default_disarm_dc,
    MineDetectionInput,
    MineDetectionResult,
    resolve_mine_detection,
    MineDisarmInput,
    MineDisarmResult,
    resolve_mine_disarm,
    DroneActivationInput,
    DroneActivationResult,
    resolve_drone_activation,
    DroneMovementInput,
    DroneMovementResult,
    resolve_drone_movement,
    MineDetonationInput,
    MineDetonationResult,
    resolve_mine_detonation,
    should_arm_mine,
    arm_mines_at_turn_start,
    create_mine,
    create_drone,
    create_deployable,
    can_detect_mine,
    is_adjacent_to_mine,
)
from core.shared.enums import DamageType, SaveType
from core.shared.models import FrozenModel
from core.shared.heat import MeltdownState
from core.mech.grid import HexCoord, HexPosition
from core.mech.combat_state import (
    DeployableState,
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    MechInventory,
)


class TestMineEffectProfiles(unittest.TestCase):
    """Tests for mine effect profiles with tier scaling."""

    def test_explosive_mine_tier1(self):
        """Explosive mine at tier 1 has DC 10."""
        profile = get_mine_effect_profile("explosive", tier=1)
        self.assertEqual(profile.mine_type, "explosive")
        self.assertEqual(profile.base_damage, 6)
        self.assertEqual(profile.damage_type, "explosive")
        self.assertEqual(profile.burst_radius, 1)
        self.assertEqual(profile.save_type, "agility")
        self.assertEqual(profile.save_difficulty, 10)

    def test_explosive_mine_tier3(self):
        """Explosive mine at tier 3 has DC 14."""
        profile = get_mine_effect_profile("explosive", tier=3)
        self.assertEqual(profile.save_difficulty, 14)

    def test_shroud_mine(self):
        """Shroud mine creates a zone with no damage."""
        profile = get_mine_effect_profile("shroud", tier=1)
        self.assertEqual(profile.mine_type, "shroud")
        self.assertEqual(profile.base_damage, 0)
        self.assertEqual(profile.burst_radius, 3)
        self.assertIsNone(profile.save_type)
        self.assertEqual(profile.special_effect, "zone")

    def test_breaching_mine(self):
        """Breaching mine targets hull save."""
        profile = get_mine_effect_profile("breaching", tier=1)
        self.assertEqual(profile.mine_type, "breaching")
        self.assertEqual(profile.save_type, "hull")
        self.assertEqual(profile.save_difficulty, 12)

    def test_cluster_mine(self):
        """Cluster mine has larger burst radius."""
        profile = get_mine_effect_profile("cluster", tier=1)
        self.assertEqual(profile.mine_type, "cluster")
        self.assertEqual(profile.burst_radius, 2)

    def test_emp_mine(self):
        """EMP mine deals energy damage targeting systems."""
        profile = get_mine_effect_profile("emp", tier=1)
        self.assertEqual(profile.mine_type, "emp")
        self.assertEqual(profile.damage_type, "energy")
        self.assertEqual(profile.save_type, "systems")


class TestDefaultDCs(unittest.TestCase):
    """Tests for default detection/disarm DCs by tier."""

    def test_detection_dc_tier1(self):
        """Tier 1 NPCs have DC 10 for detection."""
        self.assertEqual(get_default_detection_dc(1), 10)

    def test_detection_dc_tier2(self):
        """Tier 2 NPCs have DC 12 for detection."""
        self.assertEqual(get_default_detection_dc(2), 12)

    def test_detection_dc_tier3(self):
        """Tier 3 NPCs have DC 14 for detection."""
        self.assertEqual(get_default_detection_dc(3), 14)

    def test_disarm_dc_matches_detection(self):
        """Disarm DC matches detection DC by tier."""
        for tier in [1, 2, 3]:
            self.assertEqual(
                get_default_disarm_dc(tier), get_default_detection_dc(tier)
            )


class TestMineDetection(unittest.TestCase):
    """Tests for mine detection resolution per PR2 5087."""

    def test_detection_success(self):
        """Successful detection when roll + bonus >= DC."""
        result = resolve_mine_detection(
            MineDetectionInput(
                detector_id="mech-1",
                mine_id="mine-1",
                detector_systems_bonus=5,
                mine_detection_dc=10,
                force_roll=10,
            )
        )
        self.assertTrue(result.detected)
        self.assertEqual(result.roll, 10)
        self.assertEqual(result.total, 15)
        self.assertEqual(result.dc, 10)
        self.assertEqual(result.success_margin, 5)

    def test_detection_failure(self):
        """Failed detection when roll + bonus < DC."""
        result = resolve_mine_detection(
            MineDetectionInput(
                detector_id="mech-1",
                mine_id="mine-1",
                detector_systems_bonus=2,
                mine_detection_dc=15,
                force_roll=5,
            )
        )
        self.assertFalse(result.detected)
        self.assertEqual(result.roll, 5)
        self.assertEqual(result.total, 7)
        self.assertEqual(result.dc, 15)
        self.assertEqual(result.success_margin, -8)

    def test_detection_at_boundary(self):
        """Detection at exactly DC is a success."""
        result = resolve_mine_detection(
            MineDetectionInput(
                detector_id="mech-1",
                mine_id="mine-1",
                detector_systems_bonus=3,
                mine_detection_dc=10,
                force_roll=7,
            )
        )
        self.assertTrue(result.detected)
        self.assertEqual(result.success_margin, 0)

    def test_detection_uses_default_dc(self):
        """When no DC provided, uses default of 10."""
        result = resolve_mine_detection(
            MineDetectionInput(
                detector_id="mech-1",
                mine_id="mine-1",
                detector_systems_bonus=5,
            )
        )
        self.assertEqual(result.dc, 10)


class TestMineDisarm(unittest.TestCase):
    """Tests for mine disarm resolution per PR2 5088."""

    def test_disarm_success(self):
        """Successful disarm when roll + bonus >= DC."""
        result = resolve_mine_disarm(
            MineDisarmInput(
                disarmer_id="mech-1",
                mine_id="mine-1",
                disarmer_systems_bonus=6,
                mine_disarm_dc=12,
                force_roll=8,
            )
        )
        self.assertTrue(result.disarmed)
        self.assertEqual(result.roll, 8)
        self.assertEqual(result.total, 14)
        self.assertEqual(result.dc, 12)
        self.assertEqual(result.success_margin, 2)

    def test_disarm_failure(self):
        """Failed disarm when roll + bonus < DC."""
        result = resolve_mine_disarm(
            MineDisarmInput(
                disarmer_id="mech-1",
                mine_id="mine-1",
                disarmer_systems_bonus=3,
                mine_disarm_dc=15,
                force_roll=5,
            )
        )
        self.assertFalse(result.disarmed)
        self.assertEqual(result.roll, 5)
        self.assertEqual(result.total, 8)
        self.assertEqual(result.dc, 15)
        self.assertEqual(result.success_margin, -7)

    def test_disarm_uses_default_dc(self):
        """When no DC provided, uses default of 10."""
        result = resolve_mine_disarm(
            MineDisarmInput(
                disarmer_id="mech-1",
                mine_id="mine-1",
                disarmer_systems_bonus=5,
            )
        )
        self.assertEqual(result.dc, 10)


class TestDroneActivation(unittest.TestCase):
    """Tests for drone activation on owner's turn per PR2 5070-5074."""

    def test_drone_pass_action(self):
        """Drone can pass its turn."""
        result = resolve_drone_activation(
            DroneActivationInput(
                drone_id="drone-1",
                owner_id="mech-1",
                action_type="pass",
            )
        )
        self.assertEqual(result.action_taken, "pass")
        self.assertTrue(result.success)
        self.assertEqual(result.reason, "Drone passes its turn")

    def test_drone_move_requires_destination(self):
        """Move action requires destination."""
        result = resolve_drone_activation(
            DroneActivationInput(
                drone_id="drone-1",
                owner_id="mech-1",
                action_type="move",
            )
        )
        self.assertFalse(result.success)
        self.assertIn("requires destination", result.reason)

    def test_drone_move_success(self):
        """Move action with destination succeeds."""
        dest = HexPosition(coord=HexCoord(q=3, r=2))
        result = resolve_drone_activation(
            DroneActivationInput(
                drone_id="drone-1",
                owner_id="mech-1",
                action_type="move",
                move_destination=dest,
            )
        )
        self.assertEqual(result.action_taken, "move")
        self.assertTrue(result.success)
        self.assertEqual(result.new_position, dest)

    def test_drone_attack_requires_target(self):
        """Attack action requires target."""
        result = resolve_drone_activation(
            DroneActivationInput(
                drone_id="drone-1",
                owner_id="mech-1",
                action_type="attack",
            )
        )
        self.assertFalse(result.success)
        self.assertIn("requires target", result.reason)

    def test_drone_attack_success(self):
        """Attack action with target succeeds."""
        result = resolve_drone_activation(
            DroneActivationInput(
                drone_id="drone-1",
                owner_id="mech-1",
                action_type="attack",
                attack_target_id="enemy-1",
            )
        )
        self.assertEqual(result.action_taken, "attack")
        self.assertTrue(result.success)
        self.assertEqual(result.reason, "Drone attacks enemy-1")


class TestDroneMovement(unittest.TestCase):
    """Tests for drone movement following normal Lancer rules."""

    def _make_scenario_with_drone(
        self, drone_pos: tuple[int, int]
    ) -> MechCombatScenario:
        """Create scenario with a drone."""
        drone = CombatantState(
            id="drone-1",
            name="Test Drone",
            kind="mech",
            side="players",
            position=HexPosition(coord=HexCoord(q=drone_pos[0], r=drone_pos[1])),
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
        return MechCombatScenario(
            combatants=[drone],
            grapples=[],
            rounds=[],
            terrain=None,
            environment="standard",
            deployables={},
        )

    def test_drone_not_found(self):
        """Returns failure when drone not in scenario."""
        scenario = self._make_scenario_with_drone((0, 0))
        result = resolve_drone_movement(
            DroneMovementInput(
                drone_id="nonexistent-drone",
                destination=HexPosition(coord=HexCoord(q=2, r=0)),
                current_scenario=scenario,
                drone_speed=4,
            )
        )
        self.assertFalse(result.movement_successful)
        self.assertIn("not found", result.reason)

    def test_drone_movement_within_speed(self):
        """Drone can move within its speed."""
        scenario = self._make_scenario_with_drone((0, 0))
        result = resolve_drone_movement(
            DroneMovementInput(
                drone_id="drone-1",
                destination=HexPosition(coord=HexCoord(q=2, r=0)),
                current_scenario=scenario,
                drone_speed=4,
                force_movement_cost=2,
            )
        )
        self.assertTrue(result.movement_successful)
        self.assertTrue(result.path_clear)
        self.assertEqual(result.spaces_moved, 2)
        self.assertEqual(result.total_movement_cost, 2)

    def test_drone_movement_exceeds_speed(self):
        """Drone cannot move beyond its speed."""
        scenario = self._make_scenario_with_drone((0, 0))
        result = resolve_drone_movement(
            DroneMovementInput(
                drone_id="drone-1",
                destination=HexPosition(coord=HexCoord(q=5, r=0)),
                current_scenario=scenario,
                drone_speed=4,
                force_movement_cost=5,
            )
        )
        self.assertFalse(result.movement_successful)
        self.assertFalse(result.path_clear)
        self.assertEqual(result.total_movement_cost, 5)


class TestMineArming(unittest.TestCase):
    """Tests for mine arming at turn start per PR2 5083-5084."""

    def _make_mine(self, is_armed: bool, arming_turn: int | None) -> DeployableState:
        """Create a mine with specified state."""
        return DeployableState(
            id="mine-1",
            name="Test Mine",
            kind="mine",
            owner_id="mech-1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            size=1,
            hp=10,
            max_hp=10,
            is_armed=is_armed,
            arming_turn=arming_turn,
            trigger_on_adjacent_entry=True,
        )

    def test_unarmed_mine_should_arm(self):
        """Unarmed mine with arming_turn=2 should arm at turn 2."""
        mine = self._make_mine(is_armed=False, arming_turn=2)
        self.assertTrue(should_arm_mine(mine, current_turn=2))

    def test_unarmed_mine_not_ready(self):
        """Unarmed mine with arming_turn=3 should not arm at turn 2."""
        mine = self._make_mine(is_armed=False, arming_turn=3)
        self.assertFalse(should_arm_mine(mine, current_turn=2))

    def test_already_armed_mine(self):
        """Already armed mine should not arm again."""
        mine = self._make_mine(is_armed=True, arming_turn=2)
        self.assertFalse(should_arm_mine(mine, current_turn=3))

    def test_no_arming_turn_set(self):
        """Mine with no arming_turn should not arm."""
        mine = self._make_mine(is_armed=False, arming_turn=None)
        self.assertFalse(should_arm_mine(mine, current_turn=5))


class TestArmMinesAtTurnStart(unittest.TestCase):
    """Tests for arming mines at turn start."""

    def _make_scenario_with_mines(
        self, mine_states: list[dict[str, Any]]
    ) -> MechCombatScenario:
        """Create scenario with mines in various states."""
        mines: dict[str, DeployableState] = {}
        for i, state in enumerate(mine_states):
            mines[f"mine-{i}"] = DeployableState(
                id=f"mine-{i}",
                name=f"Mine {i}",
                kind="mine",
                owner_id="mech-1",
                position=HexPosition(coord=HexCoord(q=i, r=0)),
                size=1,
                hp=10,
                max_hp=10,
                is_armed=state.get("is_armed", False),
                arming_turn=state.get("arming_turn"),
                trigger_on_adjacent_entry=True,
            )
        return MechCombatScenario(
            combatants=[],
            grapples=[],
            rounds=[],
            terrain=None,
            environment="standard",
            deployables=mines,
        )

    def test_arm_single_mine(self):
        """One mine arms at turn start."""
        scenario = self._make_scenario_with_mines(
            [
                {"is_armed": False, "arming_turn": 2},
            ]
        )
        updated, armed = arm_mines_at_turn_start(scenario, current_turn=2)
        self.assertEqual(len(armed), 1)
        self.assertEqual(armed[0], "mine-0")
        self.assertTrue(updated.deployables["mine-0"].is_armed)

    def test_arm_multiple_mines(self):
        """Multiple mines arm at turn start."""
        scenario = self._make_scenario_with_mines(
            [
                {"is_armed": False, "arming_turn": 2},
                {"is_armed": False, "arming_turn": 2},
                {"is_armed": False, "arming_turn": 3},
            ]
        )
        updated, armed = arm_mines_at_turn_start(scenario, current_turn=2)
        self.assertEqual(len(armed), 2)
        self.assertIn("mine-0", armed)
        self.assertIn("mine-1", armed)

    def test_no_mines_ready(self):
        """No mines arm when none are ready."""
        scenario = self._make_scenario_with_mines(
            [
                {"is_armed": False, "arming_turn": 3},
            ]
        )
        updated, armed = arm_mines_at_turn_start(scenario, current_turn=2)
        self.assertEqual(len(armed), 0)
        self.assertFalse(updated.deployables["mine-0"].is_armed)

    def test_already_armed_mine_not_in_list(self):
        """Already armed mines are not returned."""
        scenario = self._make_scenario_with_mines(
            [
                {"is_armed": True, "arming_turn": 2},
                {"is_armed": False, "arming_turn": 2},
            ]
        )
        updated, armed = arm_mines_at_turn_start(scenario, current_turn=2)
        self.assertEqual(len(armed), 1)
        self.assertIn("mine-1", armed)


class TestCreateDeployables(unittest.TestCase):
    """Tests for deployable factory functions with PR2 defaults."""

    def test_create_mine_defaults(self):
        """Create mine has correct PR2 defaults."""
        pos = HexPosition(coord=HexCoord(q=0, r=0))
        mine = create_mine(
            id="mine-1",
            name="Explosive Mine",
            owner_id="mech-1",
            position=pos,
            mine_type="explosive",
            tier=1,
        )
        self.assertEqual(mine.kind, "mine")
        self.assertEqual(mine.size, 1)
        self.assertEqual(mine.hp, 10)
        self.assertEqual(mine.max_hp, 10)
        self.assertEqual(mine.evasion, 5)
        self.assertEqual(mine.armor, 0)
        self.assertFalse(mine.is_armed)
        self.assertTrue(mine.trigger_on_adjacent_entry)

    def test_create_drone_defaults(self):
        """Create drone has correct PR2 defaults."""
        pos = HexPosition(coord=HexCoord(q=0, r=0))
        drone = create_drone(
            id="drone-1",
            name="Turret Drone",
            owner_id="mech-1",
            position=pos,
        )
        self.assertEqual(drone.kind, "drone")
        self.assertEqual(drone.size, 1)
        self.assertEqual(drone.hp, 10)
        self.assertEqual(drone.evasion, 10)
        self.assertEqual(drone.armor, 0)
        self.assertEqual(drone.e_defense, 10)
        self.assertTrue(drone.acts_on_owner_turn)
        self.assertFalse(drone.can_act)
        self.assertFalse(drone.can_move)

    def test_create_drone_with_actions(self):
        """Create drone with action capabilities."""
        pos = HexPosition(coord=HexCoord(q=0, r=0))
        drone = create_drone(
            id="drone-1",
            name="Restock Drone",
            owner_id="mech-1",
            position=pos,
            can_act=True,
            can_move=True,
            speed=4,
        )
        self.assertTrue(drone.can_act)
        self.assertTrue(drone.can_move)

    def test_create_deployable_defaults(self):
        """Create deployable has correct PR2 defaults."""
        pos = HexPosition(coord=HexCoord(q=0, r=0))
        deployable = create_deployable(
            id="cover-1",
            name="Deployable Cover",
            owner_id="mech-1",
            position=pos,
            size=1,
            cover="soft",
        )
        self.assertEqual(deployable.kind, "deployable")
        self.assertEqual(deployable.size, 1)
        self.assertEqual(deployable.hp, 10)
        self.assertEqual(deployable.evasion, 5)
        self.assertEqual(deployable.cover, "soft")

    def test_create_deployable_size_scaling(self):
        """Create deployable with size > 1 has scaled HP."""
        pos = HexPosition(coord=HexCoord(q=0, r=0))
        deployable = create_deployable(
            id="bunker-1",
            name="Portable Bunker",
            owner_id=None,
            position=pos,
            size=2,
            armor=2,
        )
        self.assertEqual(deployable.size, 2)
        self.assertEqual(deployable.hp, 20)
        self.assertEqual(deployable.armor, 2)


class TestCanDetectMine(unittest.TestCase):
    """Tests for mine detection range check."""

    def test_within_sensor_range(self):
        """Mine within sensor range can be detected."""
        detector_pos = HexPosition(coord=HexCoord(q=0, r=0))
        mine_pos = HexPosition(coord=HexCoord(q=3, r=0))
        self.assertTrue(can_detect_mine(detector_pos, mine_pos, sensor_range=5))

    def test_outside_sensor_range(self):
        """Mine outside sensor range cannot be detected."""
        detector_pos = HexPosition(coord=HexCoord(q=0, r=0))
        mine_pos = HexPosition(coord=HexCoord(q=6, r=0))
        self.assertFalse(can_detect_mine(detector_pos, mine_pos, sensor_range=5))

    def test_no_position(self):
        """Combatant with no position cannot detect."""
        mine_pos = HexPosition(coord=HexCoord(q=3, r=0))
        self.assertFalse(can_detect_mine(None, mine_pos, sensor_range=5))


class TestIsAdjacentToMine(unittest.TestCase):
    """Tests for adjacency check for mine disarm."""

    def test_adjacent(self):
        """Adjacent combatant can disarm."""
        combatant_pos = HexPosition(coord=HexCoord(q=0, r=0))
        mine_pos = HexPosition(coord=HexCoord(q=0, r=1))
        self.assertTrue(is_adjacent_to_mine(combatant_pos, mine_pos))

    def test_not_adjacent(self):
        """Non-adjacent combatant cannot disarm."""
        combatant_pos = HexPosition(coord=HexCoord(q=0, r=0))
        mine_pos = HexPosition(coord=HexCoord(q=2, r=0))
        self.assertFalse(is_adjacent_to_mine(combatant_pos, mine_pos))

    def test_same_position(self):
        """Same position is not adjacent."""
        pos = HexPosition(coord=HexCoord(q=0, r=0))
        self.assertFalse(is_adjacent_to_mine(pos, pos))

    def test_no_position(self):
        """Combatant with no position cannot be adjacent."""
        mine_pos = HexPosition(coord=HexCoord(q=0, r=0))
        self.assertFalse(is_adjacent_to_mine(None, mine_pos))


class TestMineDetonation(unittest.TestCase):
    """Tests for mine detonation resolution."""

    def _make_scenario_with_mine(self) -> MechCombatScenario:
        """Create scenario with a mine."""
        mine = DeployableState(
            id="mine-1",
            name="Cluster Mine",
            kind="mine",
            owner_id="mech-1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            size=1,
            hp=10,
            max_hp=10,
            is_armed=True,
        )
        return MechCombatScenario(
            combatants=[],
            grapples=[],
            rounds=[],
            terrain=None,
            environment="standard",
            deployables={"mine-1": mine},
        )

    def test_detonation_simple(self):
        """Simple detonation without saves."""
        scenario = self._make_scenario_with_mine()
        input_data = MineDetonationInput(
            mine_id="mine-1",
            triggerer_id="mech-1",
            scenario=scenario,
            effect_profile=MineEffectProfile(
                mine_type="cluster",
                base_damage=4,
                damage_type="explosive",
                burst_radius=2,
            ),
            tier=1,
        )
        result = resolve_mine_detonation(input_data)
        self.assertTrue(result.detonated)
        self.assertIn("burst 2", result.reason)


if __name__ == "__main__":
    unittest.main()
