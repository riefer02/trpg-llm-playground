"""Tests for Unified Movement Resolution Module

Per PR2 3729-3930: Comprehensive movement system tests.

Movement Rules Tested:
- Regular movement within speed
- Difficult terrain cost calculation
- Engagement stop rules (PR2 3818-3819)
- Obstruction blocking (PR2 3812-3815)
- Flight validation (PR2 3919-3922)
- Teleport validation (PR2 3897-3902)
- Pathfinding helpers (hex_line, cube_round)
"""

from __future__ import annotations

import unittest
from core.shared.movement import (
    MovementMode,
    MovementInput,
    MovementResult,
    hex_line_simple,
    cube_round,
    check_engagement_stop,
    check_obstructions,
    validate_flight,
    validate_teleport,
    resolve_movement,
    DroneMovementInput,
    DroneMovementResult,
    resolve_drone_movement,
)
from core.shared.heat import MeltdownState
from core.shared.enums import SizeClass
from core.shared.terrain import TerrainHex
from core.mech.grid import HexCoord, HexPosition
from core.mech.terrain import TerrainMap, TerrainHex as TerrainHexType
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
    MechInventory,
    MechCombatScenario,
    CombatSide,
)


def make_test_scenario(
    combatants: list[CombatantState] | None = None,
    terrain: TerrainMap | None = None,
) -> MechCombatScenario:
    """Create a basic test scenario."""
    return MechCombatScenario(
        combatants=combatants or [],
        terrain=terrain,
        deployables={},
    )


def make_test_combatant(
    combatant_id: str = "mech-1",
    position: tuple[int, int] = (0, 0),
    size: SizeClass = "size_1",
    side: CombatSide = "players",
) -> CombatantState:
    """Create a test combatant."""
    return CombatantState(
        id=combatant_id,
        name=f"Test Combatant {combatant_id}",
        side=side,
        kind="mech",
        stats=CombatStats(
            size=size,
            hp_max=10,
            evasion=10,
            e_defense=10,
            sensor_range=5,
        ),
        resources=CombatResources(hp_current=10),
        position=HexPosition(coord=HexCoord(q=position[0], r=position[1])),
        inventory=MechInventory(mounts=[], systems=[]),
        statuses=[],
        conditions=[],
    )


def make_difficult_terrain(
    coords: list[tuple[int, int]],
    elevation: int = 0,
) -> TerrainMap:
    """Create terrain map with difficult terrain at specified coordinates."""
    tiles = []
    for q, r in coords:
        tiles.append(
            TerrainHex(
                coord=HexCoord(q=q, r=r),
                elevation=elevation,
                difficult=True,
                dangerous=False,
            )
        )
    return TerrainMap(tiles=tiles)


class TestMovementInput(unittest.TestCase):
    """Tests for MovementInput model."""

    def test_basic_creation(self):
        """Test basic MovementInput creation with required fields."""
        scenario = make_test_scenario()
        input_data = MovementInput(
            entity_id="mech-1",
            destination=HexPosition(coord=HexCoord(q=5, r=0)),
            current_scenario=scenario,
            speed=5,
        )
        self.assertEqual(input_data.entity_id, "mech-1")
        self.assertEqual(input_data.speed, 5)
        self.assertEqual(input_data.mode, "ground")

    def test_full_options(self):
        """Test MovementInput with all optional fields."""
        scenario = make_test_scenario()
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0), HexCoord(q=2, r=0)]
        input_data = MovementInput(
            entity_id="mech-1",
            destination=HexPosition(coord=HexCoord(q=5, r=0)),
            current_scenario=scenario,
            speed=5,
            mode="flight",
            ignore_engagement=True,
            ignore_reactions=True,
            force_movement_cost=3,
            force_path=path,
        )
        self.assertEqual(input_data.mode, "flight")
        self.assertTrue(input_data.ignore_engagement)
        self.assertEqual(input_data.force_movement_cost, 3)


class TestMovementResult(unittest.TestCase):
    """Tests for MovementResult model."""

    def test_success_result(self):
        """Test successful movement result."""
        result = MovementResult(
            movement_successful=True,
            path_clear=True,
            spaces_moved=3,
            total_movement_cost=3,
            new_position=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain_costs=[1, 1, 1],
            terrain_encountered=[],
            reason="moved 3 spaces, cost 3/5",
        )
        self.assertTrue(result.movement_successful)
        self.assertEqual(result.spaces_moved, 3)

    def test_failure_result(self):
        """Test failed movement result."""
        result = MovementResult(
            movement_successful=False,
            path_clear=False,
            spaces_moved=0,
            total_movement_cost=0,
            new_position=None,
            engagement_stopped=True,
            reason="stopped by engagement, cost 0/5",
        )
        self.assertFalse(result.movement_successful)
        self.assertTrue(result.engagement_stopped)


class TestHexLineSimple(unittest.TestCase):
    """Tests for hex_line_simple pathfinding helper."""

    def test_same_position(self):
        """Line from position to itself returns single hex."""
        start = HexCoord(q=0, r=0)
        end = HexCoord(q=0, r=0)
        result = hex_line_simple(start, end)
        self.assertEqual(result, [start])

    def test_adjacent_hex(self):
        """Line to adjacent hex returns two positions."""
        start = HexCoord(q=0, r=0)
        end = HexCoord(q=1, r=0)
        result = hex_line_simple(start, end)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], start)
        self.assertEqual(result[-1], end)

    def test_distant_hex(self):
        """Line to distant hex returns intermediate hexes."""
        start = HexCoord(q=0, r=0)
        end = HexCoord(q=5, r=0)
        result = hex_line_simple(start, end)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0], start)
        self.assertEqual(result[-1], end)


class TestCubeRound(unittest.TestCase):
    """Tests for cube_round helper function."""

    def test_integer_cube(self):
        """Integer coordinates round to themselves."""
        result = cube_round((1.0, 2.0, -3.0))
        self.assertEqual(result, (1, 2, -3))

    def test_fractional_cube(self):
        """Fractional coordinates round correctly."""
        result = cube_round((1.3, 2.6, -3.9))
        self.assertEqual(result, (1, 3, -4))

    def test_tiebreaker(self):
        """Tiebreaker uses max diff rule."""
        result = cube_round((1.5, 2.5, -4.0))
        self.assertEqual(result, (2, 2, -4))


class TestCheckEngagementStop(unittest.TestCase):
    """Tests for engagement stop validation per PR2 3818-3819."""

    def _make_scenario_with_hostiles(
        self,
        mech_pos: tuple[int, int],
        hostile_pos: tuple[int, int],
        hostile_size: SizeClass = "size_1",
    ) -> MechCombatScenario:
        """Create scenario with moving combatant and hostile."""
        moving = make_test_combatant("mech-1", mech_pos, "size_1")
        hostile = make_test_combatant(
            "hostile-1",
            hostile_pos,
            hostile_size,
            side="hostiles",
        )
        return make_test_scenario([moving, hostile])

    def test_no_hostiles(self):
        """No hostiles means no engagement stop."""
        scenario = make_test_scenario([make_test_combatant("mech-1", (0, 0))])
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_1", path, scenario)
        self.assertFalse(should_stop)

    def test_hostile_not_adjacent(self):
        """Hostile not adjacent doesn't trigger stop."""
        scenario = self._make_scenario_with_hostiles((0, 0), (5, 5))
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_1", path, scenario)
        self.assertFalse(should_stop)

    def test_adjacent_same_size_hostile(self):
        """Must stop when adjacent to same-size hostile."""
        scenario = self._make_scenario_with_hostiles((0, 0), (2, 0))
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_1", path, scenario)
        self.assertTrue(should_stop)
        self.assertEqual(pos, HexCoord(q=1, r=0))

    def test_adjacent_larger_hostile(self):
        """Must stop when adjacent to larger hostile."""
        scenario = self._make_scenario_with_hostiles((0, 0), (2, 0), "size_2")
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_1", path, scenario)
        self.assertTrue(should_stop)

    def test_adjacent_smaller_hostile(self):
        """Can pass adjacent to smaller hostile."""
        scenario = self._make_scenario_with_hostiles((0, 0), (1, 0), "size_half")
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_1", path, scenario)
        self.assertFalse(should_stop)

    def test_ignore_engagement(self):
        """ignore_engagement flag skips check."""
        scenario = self._make_scenario_with_hostiles((0, 0), (1, 0))
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop(
            "mech-1", "size_1", path, scenario, ignore_engagement=True
        )
        self.assertFalse(should_stop)

    def test_allied_not_hostile(self):
        """Allied characters don't trigger engagement."""
        allied = make_test_combatant("ally-1", (1, 0), "size_1")
        moving = make_test_combatant("mech-1", (0, 0))
        scenario = make_test_scenario([moving, allied])
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_1", path, scenario)
        self.assertFalse(should_stop)

    def test_size_2_mech_engages_at_distance_2(self):
        """Size 2 mech stops when within 2 hexes of size 1+ hostile."""
        # Size 2 mech at (0,0), size 1 hostile at (3,0)
        moving = make_test_combatant("mech-1", (0, 0), size="size_2")
        hostile = make_test_combatant("hostile-1", (3, 0), "size_1", side="hostiles")
        scenario = make_test_scenario([moving, hostile])

        # Path from (0,0) to (2,0) - hex (2,0) is distance 1 from hostile at (3,0)
        # But with size 2 adjacency, distance 2 should trigger
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0), HexCoord(q=2, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_2", path, scenario)

        # Size 2 mech should NOT stop at distance 2 from hostile when hostile is SMALLER
        # Wait - size_1 hostile is same or smaller than size_2, so no stop
        self.assertFalse(should_stop)

    def test_size_2_mech_engages_same_size_at_distance_2(self):
        """Size 2 mech stops when within adjacency range of same-size hostile."""
        # Size 2 mech at (0,0), size 2 hostile at (3,0)
        moving = make_test_combatant("mech-1", (0, 0), size="size_2")
        hostile = make_test_combatant("hostile-1", (3, 0), "size_2", side="hostiles")
        scenario = make_test_scenario([moving, hostile])

        # Path from (0,0) towards hostile. Hex (1,0) is distance 2 from hostile at (3,0)
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_2", path, scenario)

        # Size 2 vs size 2: adjacency distance is 2
        # At (1,0), distance to (3,0) is 2, which is within adjacency
        # Same size triggers stop
        self.assertTrue(should_stop)
        self.assertEqual(pos, HexCoord(q=1, r=0))

    def test_size_3_mech_engages_at_distance_3(self):
        """Size 3 mech stops when within 3 hexes of same/larger hostile."""
        moving = make_test_combatant("mech-1", (0, 0), size="size_3")
        hostile = make_test_combatant("hostile-1", (4, 0), "size_3", side="hostiles")
        scenario = make_test_scenario([moving, hostile])

        # Moving from (0,0) to (1,0). Distance from (1,0) to (4,0) is 3.
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_3", path, scenario)

        # Size 3 adjacency distance is 3, and hostile is same size
        self.assertTrue(should_stop)
        self.assertEqual(pos, HexCoord(q=1, r=0))

    def test_size_1_mech_engages_size_3_hostile_at_distance_3(self):
        """Size 1 mech stops at distance 3 from size 3 hostile (larger hostile)."""
        moving = make_test_combatant("mech-1", (0, 0), size="size_1")
        hostile = make_test_combatant("hostile-1", (4, 0), "size_3", side="hostiles")
        scenario = make_test_scenario([moving, hostile])

        # Moving from (0,0) to (1,0). Distance from (1,0) to (4,0) is 3.
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        should_stop, pos = check_engagement_stop("mech-1", "size_1", path, scenario)

        # Size 1 vs size 3: adjacency distance is 3 (max of sizes)
        # Hostile is larger, so must stop
        self.assertTrue(should_stop)
        self.assertEqual(pos, HexCoord(q=1, r=0))


class TestCheckObstructions(unittest.TestCase):
    """Tests for obstruction checking per PR2 3812-3815."""

    def _make_scenario_with_obstruction(
        self,
        mech_pos: tuple[int, int],
        obstacle_pos: tuple[int, int],
        obstacle_size: SizeClass,
    ) -> MechCombatScenario:
        """Create scenario with moving combatant and obstruction."""
        moving = make_test_combatant("mech-1", mech_pos, "size_1")
        obstacle = make_test_combatant(
            "obstacle-1",
            obstacle_pos,
            obstacle_size,
            side="hostiles",
        )
        return make_test_scenario([moving, obstacle])

    def test_no_obstructions(self):
        """Empty path is clear."""
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        scenario = make_test_scenario([make_test_combatant("mech-1", (0, 0))])
        blocked, idx = check_obstructions(path, "mech-1", "size_1", scenario, "ground")
        self.assertFalse(blocked)

    def test_larger_obstacle_blocks(self):
        """Larger obstacle blocks path."""
        scenario = self._make_scenario_with_obstruction((0, 0), (1, 0), "size_2")
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        blocked, idx = check_obstructions(path, "mech-1", "size_1", scenario, "ground")
        self.assertTrue(blocked)
        self.assertEqual(idx, 1)

    def test_smaller_does_not_block(self):
        """Smaller character doesn't block."""
        scenario = self._make_scenario_with_obstruction((0, 0), (1, 0), "size_half")
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        blocked, idx = check_obstructions(path, "mech-1", "size_1", scenario, "ground")
        self.assertFalse(blocked)

    def test_same_size_blocks(self):
        """Same-size character blocks."""
        scenario = self._make_scenario_with_obstruction((0, 0), (1, 0), "size_1")
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        blocked, idx = check_obstructions(path, "mech-1", "size_1", scenario, "ground")
        self.assertTrue(blocked)

    def test_ally_does_not_block(self):
        """Allied character doesn't block."""
        ally = make_test_combatant("ally-1", (1, 0), "size_1")
        moving = make_test_combatant("mech-1", (0, 0))
        scenario = make_test_scenario([moving, ally])
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        blocked, idx = check_obstructions(path, "mech-1", "size_1", scenario, "ground")
        self.assertFalse(blocked)

    def test_flight_ignores_obstructions(self):
        """Flight mode ignores obstructions."""
        scenario = self._make_scenario_with_obstruction((0, 0), (1, 0), "size_2")
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        blocked, idx = check_obstructions(path, "mech-1", "size_1", scenario, "flight")
        self.assertFalse(blocked)

    def test_teleport_ignores_obstructions(self):
        """Teleport mode ignores obstructions."""
        scenario = self._make_scenario_with_obstruction((0, 0), (1, 0), "size_2")
        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        blocked, idx = check_obstructions(
            path, "mech-1", "size_1", scenario, "teleport"
        )
        self.assertFalse(blocked)


class TestValidateFlight(unittest.TestCase):
    """Tests for flight validation per PR2 3919-3922."""

    def test_ground_mode_always_valid(self):
        """Ground mode doesn't need flight validation."""
        valid, reason = validate_flight([], "ground", None)
        self.assertTrue(valid)

    def test_flight_no_movement_fails(self):
        """Flying without movement fails (unless hover)."""
        path = [HexCoord(q=0, r=0)]
        valid, reason = validate_flight(path, "flight", None)
        self.assertFalse(valid)
        self.assertIn("fall", reason.lower())

    def test_hover_stationary_allowed(self):
        """Hover mode allows stationary position."""
        path = [HexCoord(q=0, r=0)]
        valid, reason = validate_flight(path, "hover", None)
        self.assertTrue(valid)

    def test_flight_with_movement_valid(self):
        """Flying with movement is valid."""
        path = [
            HexPosition(coord=HexCoord(q=0, r=0)),
            HexPosition(coord=HexCoord(q=1, r=0)),
        ]
        valid, reason = validate_flight(path, "flight", None)
        self.assertTrue(valid)

    def test_hover_stationary_allowed(self):
        """Hover mode allows stationary position."""
        path = [HexPosition(coord=HexCoord(q=0, r=0))]
        valid, reason = validate_flight(path, "hover", None)
        self.assertTrue(valid)


class TestValidateTeleport(unittest.TestCase):
    """Tests for teleport validation per PR2 3897-3902."""

    def _make_scenario_with_target(
        self,
        target_pos: tuple[int, int],
    ) -> MechCombatScenario:
        """Create scenario with target at position."""
        target = make_test_combatant(
            "target-1",
            target_pos,
            "size_1",
            side="hostiles",
        )
        return make_test_scenario([target])

    def test_teleport_surface_to_surface(self):
        """Teleport from surface to surface is valid."""
        start = HexPosition(coord=HexCoord(q=0, r=0), elevation=0)
        end = HexPosition(coord=HexCoord(q=5, r=0), elevation=0)
        scenario = self._make_scenario_with_target((10, 0))
        valid, reason = validate_teleport(start, end, "mech-1", scenario)
        self.assertTrue(valid)

    def test_teleport_from_midair_fails(self):
        """Teleport from mid-air fails."""
        start = HexPosition(coord=HexCoord(q=0, r=0), elevation=5)
        end = HexPosition(coord=HexCoord(q=5, r=0), elevation=0)
        scenario = self._make_scenario_with_target((10, 0))
        valid, reason = validate_teleport(start, end, "mech-1", scenario)
        self.assertFalse(valid)
        self.assertIn("mid-air", reason.lower())

    def test_teleport_to_midair_fails(self):
        """Teleport to mid-air fails."""
        start = HexPosition(coord=HexCoord(q=0, r=0), elevation=0)
        end = HexPosition(coord=HexCoord(q=5, r=0), elevation=5)
        scenario = self._make_scenario_with_target((10, 0))
        valid, reason = validate_teleport(start, end, "mech-1", scenario)
        self.assertFalse(valid)
        self.assertIn("mid-air", reason.lower())

    def test_teleport_to_occupied_fails(self):
        """Teleport to occupied space fails."""
        start = HexPosition(coord=HexCoord(q=0, r=0), elevation=0)
        end = HexPosition(coord=HexCoord(q=1, r=0), elevation=0)
        scenario = self._make_scenario_with_target((1, 0))
        valid, reason = validate_teleport(start, end, "mech-1", scenario)
        self.assertFalse(valid)
        self.assertIn("occupied", reason.lower())


class TestResolveMovement(unittest.TestCase):
    """Tests for unified resolve_movement function."""

    def test_entity_not_found(self):
        """Returns failure when entity not in scenario."""
        scenario = make_test_scenario()
        result = resolve_movement(
            MovementInput(
                entity_id="nonexistent",
                destination=HexPosition(coord=HexCoord(q=5, r=0)),
                current_scenario=scenario,
                speed=5,
            )
        )
        self.assertFalse(result.movement_successful)
        self.assertIn("not found", result.reason)

    def test_movement_within_speed(self):
        """Movement within speed succeeds."""
        mech = make_test_combatant("mech-1", (0, 0), "size_1")
        scenario = make_test_scenario([mech])
        result = resolve_movement(
            MovementInput(
                entity_id="mech-1",
                destination=HexPosition(coord=HexCoord(q=3, r=0)),
                current_scenario=scenario,
                speed=5,
            )
        )
        self.assertTrue(result.movement_successful)
        self.assertEqual(result.spaces_moved, 3)

    def test_movement_exceeds_speed(self):
        """Movement beyond speed fails."""
        mech = make_test_combatant("mech-1", (0, 0), "size_1")
        scenario = make_test_scenario([mech])
        result = resolve_movement(
            MovementInput(
                entity_id="mech-1",
                destination=HexPosition(coord=HexCoord(q=5, r=0)),
                current_scenario=scenario,
                speed=3,
            )
        )
        self.assertFalse(result.movement_successful)
        self.assertEqual(result.spaces_moved, 3)
        self.assertEqual(result.total_movement_cost, 3)

    def test_engagement_stop(self):
        """Movement stops when adjacent to same-size hostile."""
        mech = make_test_combatant("mech-1", (0, 0), "size_1")
        hostile = make_test_combatant("hostile-1", (2, 1), "size_1", side="hostiles")
        scenario = make_test_scenario([mech, hostile])
        result = resolve_movement(
            MovementInput(
                entity_id="mech-1",
                destination=HexPosition(coord=HexCoord(q=5, r=0)),
                current_scenario=scenario,
                speed=5,
            )
        )
        self.assertFalse(result.movement_successful)
        self.assertTrue(result.engagement_stopped)
        self.assertEqual(result.spaces_moved, 1)

    def test_ignore_engagement(self):
        """ignore_engagement allows passing hostiles."""
        mech = make_test_combatant("mech-1", (0, 0), "size_1")
        hostile = make_test_combatant("hostile-1", (2, 1), "size_1", side="hostiles")
        scenario = make_test_scenario([mech, hostile])
        result = resolve_movement(
            MovementInput(
                entity_id="mech-1",
                destination=HexPosition(coord=HexCoord(q=5, r=0)),
                current_scenario=scenario,
                speed=5,
                ignore_engagement=True,
            )
        )
        self.assertTrue(result.movement_successful)

    def test_force_movement_cost(self):
        """force_movement_cost overrides calculated cost for testing."""
        mech = make_test_combatant("mech-1", (0, 0), "size_1")
        scenario = make_test_scenario([mech])
        result = resolve_movement(
            MovementInput(
                entity_id="mech-1",
                destination=HexPosition(coord=HexCoord(q=3, r=0)),
                current_scenario=scenario,
                speed=5,
                force_movement_cost=10,
            )
        )
        self.assertFalse(result.movement_successful)
        self.assertEqual(result.total_movement_cost, 10)

    def test_force_path(self):
        """force_path bypasses path calculation."""
        mech = make_test_combatant("mech-1", (0, 0), "size_1")
        scenario = make_test_scenario([mech])
        path = [HexCoord(q=i, r=0) for i in range(11)]
        result = resolve_movement(
            MovementInput(
                entity_id="mech-1",
                destination=HexPosition(coord=HexCoord(q=10, r=0)),
                current_scenario=scenario,
                speed=10,
                force_path=path,
            )
        )
        self.assertTrue(result.path_clear)
        self.assertEqual(result.spaces_moved, 10)


class TestSize2PlusFootprintObstructions(unittest.TestCase):
    """Tests for Size 2+ footprint-aware obstruction checking."""

    def test_size_2_mech_blocks_adjacent_hexes(self):
        """Size 2 mech at (0,0) blocks movement through all footprint hexes."""
        # Size 2 mech at (0,0) occupies center + 6 adjacent hexes (radius 1)
        # A size_1 mech trying to move through (1,0) should be blocked
        size_2_mech = make_test_combatant("size2-mech", (0, 0), "size_2", side="hostiles")
        moving = make_test_combatant("mech-1", (-2, 0), "size_1")
        scenario = make_test_scenario([size_2_mech, moving])

        # Path from (-2,0) to (2,0) should be blocked at (-1,0) or (0,0) or (1,0)
        # since size 2 footprint includes all those hexes
        path = [HexCoord(q=-2, r=0), HexCoord(q=-1, r=0), HexCoord(q=0, r=0)]
        blocked, idx = check_obstructions(path, "mech-1", "size_1", scenario, "ground")
        self.assertTrue(blocked)

    def test_size_2_footprint_covers_seven_hexes(self):
        """Size 2 mech footprint includes center and 6 adjacent hexes."""
        # Size 2 mech at (0,0)
        size_2_mech = make_test_combatant("size2-mech", (0, 0), "size_2", side="hostiles")
        moving = make_test_combatant("mech-1", (3, 0), "size_1")
        scenario = make_test_scenario([size_2_mech, moving])

        # Test that (1, 0) is blocked (adjacent to center, within footprint)
        path_through_footprint = [HexCoord(q=3, r=0), HexCoord(q=2, r=0), HexCoord(q=1, r=0)]
        blocked, idx = check_obstructions(path_through_footprint, "mech-1", "size_1", scenario, "ground")
        self.assertTrue(blocked)
        self.assertEqual(idx, 2)  # Blocked at index 2 (hex 1,0)

    def test_larger_mech_can_move_through_smaller_footprint(self):
        """Size 2 mech can move through size 1 mech's position."""
        size_1_obstacle = make_test_combatant("obstacle", (1, 0), "size_1", side="hostiles")
        size_2_moving = make_test_combatant("mech-1", (0, 0), "size_2")
        scenario = make_test_scenario([size_1_obstacle, size_2_moving])

        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0), HexCoord(q=2, r=0)]
        blocked, idx = check_obstructions(path, "mech-1", "size_2", scenario, "ground")
        self.assertFalse(blocked)

    def test_flight_ignores_footprint(self):
        """Flying movement ignores Size 2+ footprints."""
        size_2_mech = make_test_combatant("size2-mech", (0, 0), "size_2", side="hostiles")
        moving = make_test_combatant("mech-1", (-2, 0), "size_1")
        scenario = make_test_scenario([size_2_mech, moving])

        path = [HexCoord(q=-2, r=0), HexCoord(q=-1, r=0), HexCoord(q=0, r=0)]
        blocked, idx = check_obstructions(path, "mech-1", "size_1", scenario, "flight")
        self.assertFalse(blocked)


class TestSize2PlusTeleportValidation(unittest.TestCase):
    """Tests for Size 2+ footprint-aware teleport validation."""

    def test_teleport_blocked_by_size_2_footprint(self):
        """Cannot teleport to hex within Size 2 mech's footprint."""
        size_2_mech = make_test_combatant("size2-mech", (0, 0), "size_2", side="hostiles")
        scenario = make_test_scenario([size_2_mech])

        start = HexPosition(coord=HexCoord(q=-5, r=0), elevation=0)
        # Try to teleport to (1, 0) which is within Size 2 footprint
        end = HexPosition(coord=HexCoord(q=1, r=0), elevation=0)

        valid, reason = validate_teleport(start, end, "mech-1", scenario)
        self.assertFalse(valid)
        self.assertIn("occupied", reason.lower())

    def test_teleport_to_outside_footprint_allowed(self):
        """Can teleport to hex outside Size 2 mech's footprint."""
        size_2_mech = make_test_combatant("size2-mech", (0, 0), "size_2", side="hostiles")
        scenario = make_test_scenario([size_2_mech])

        start = HexPosition(coord=HexCoord(q=-5, r=0), elevation=0)
        # (2, 0) is outside Size 2 footprint (center 0,0 with radius 1)
        end = HexPosition(coord=HexCoord(q=2, r=0), elevation=0)

        valid, reason = validate_teleport(start, end, "mech-1", scenario)
        self.assertTrue(valid)


class TestSize2PlusEngagementFootprint(unittest.TestCase):
    """Tests for Size 2+ footprint-aware engagement distance calculation."""

    def test_engagement_uses_closest_footprint_hex(self):
        """Engagement distance calculated from closest footprint hex."""
        # Size 2 hostile at (0,0), footprint includes (-1,0) to (1,0) etc.
        # Size 1 mech moving to (2,0) is adjacent to footprint hex (1,0)
        size_2_hostile = make_test_combatant("hostile", (0, 0), "size_2", side="hostiles")
        moving = make_test_combatant("mech-1", (-3, 0), "size_1")
        scenario = make_test_scenario([size_2_hostile, moving])

        # Move from (-3,0) to (2,0). At (2,0), distance to center (0,0) is 2,
        # but distance to footprint hex (1,0) is 1 - should trigger engagement
        path = [HexCoord(q=-3, r=0), HexCoord(q=-2, r=0), HexCoord(q=-1, r=0),
                HexCoord(q=0, r=0), HexCoord(q=1, r=0), HexCoord(q=2, r=0)]

        # When entering hex (2,0), the closest footprint hex is (1,0) at distance 1
        # Size 2 vs Size 1: adjacency distance is max(2,1) = 2
        # But engagement stops when ADJACENT (distance <= adj_dist)
        # At (2,0), min distance to footprint is 1, adj_dist is 2, so should stop
        should_stop, pos = check_engagement_stop("mech-1", "size_1", path, scenario)
        self.assertTrue(should_stop)

    def test_size_1_mech_adjacent_to_size_2_footprint_hex(self):
        """Size 1 mech stops adjacent to Size 2 footprint edge hex."""
        # Size 2 hostile centered at (3,0) - footprint includes (2,0), (4,0), etc.
        # Path from (0,0) towards (5,0) should stop when adjacent to (2,0)
        size_2_hostile = make_test_combatant("hostile", (3, 0), "size_2", side="hostiles")
        moving = make_test_combatant("mech-1", (0, 0), "size_1")
        scenario = make_test_scenario([size_2_hostile, moving])

        path = [HexCoord(q=0, r=0), HexCoord(q=1, r=0)]
        # At (1,0), closest footprint hex is (2,0) at distance 1
        should_stop, pos = check_engagement_stop("mech-1", "size_1", path, scenario)
        self.assertTrue(should_stop)
        self.assertEqual(pos, HexCoord(q=1, r=0))


class TestDroneMovementBackwardCompat(unittest.TestCase):
    """Tests for backward-compatible DroneMovementInput/Result."""

    def _make_scenario_with_drone(
        self,
        drone_pos: tuple[int, int],
    ) -> MechCombatScenario:
        """Create scenario with drone combatant."""
        drone = make_test_combatant(
            "drone-1",
            drone_pos,
            "size_half",
        )
        return make_test_scenario([drone])

    def test_resolve_drone_movement_basic(self):
        """Backward-compatible resolve_drone_movement works."""
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
        self.assertEqual(result.spaces_moved, 2)

    def test_resolve_drone_movement_not_found(self):
        """Returns failure when drone not found."""
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

    def test_resolve_drone_movement_exceeds_speed(self):
        """Returns failure when movement exceeds speed."""
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


if __name__ == "__main__":
    unittest.main()
