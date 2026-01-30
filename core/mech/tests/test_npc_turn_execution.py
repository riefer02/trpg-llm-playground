"""Tests for NPC turn execution and automated combat."""

from core.mech.npc_turn_execution import (
    build_target_info_list,
    get_npc_role,
    execute_npc_turn,
)
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    CombatTurn,
    CombatRound,
)
from core.mech.grid import HexPosition, HexCoord


# =============================================================================
# Test Fixtures
# =============================================================================


def make_combatant(
    id: str = "mech_1",
    name: str = "Test Mech",
    side: str = "players",
    hp_max: int = 10,
    hp_current: int = 10,
    ai_controlled: bool = False,
    npc_role: str | None = None,
    position: HexPosition | None = None,
    sensor_range: int = 10,
) -> CombatantState:
    """Create a test combatant."""
    if position is None:
        position = HexPosition(coord=HexCoord(q=0, r=0), elevation=0)
    return CombatantState(
        id=id,
        name=name,
        side=side,
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=hp_max,
            evasion=8,
            e_defense=8,
            armor=0,
            speed=4,
            sensor_range=sensor_range,
            tech_attack=0,
            grit=0,
        ),
        resources=CombatResources(
            hp_current=hp_current,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=position,
        ai_controlled=ai_controlled,
        npc_role=npc_role,
    )


def make_scenario_with_turn_order(
    combatants: list[CombatantState],
) -> MechCombatScenario:
    """Create a scenario with turn order initialized."""
    turns = [CombatTurn(actor_id=c.id) for c in combatants]
    round_1 = CombatRound(round_index=1, turns=turns)
    return MechCombatScenario(
        combatants=combatants,
        rounds=[round_1],
        grapples=[],
        deployables={},
        environment="standard",
    )


# =============================================================================
# Tests: build_target_info_list
# =============================================================================


class TestBuildTargetInfoList:
    """Tests for build_target_info_list function."""

    def test_build_target_info_includes_enemies(self):
        """Enemies in range are included in target list."""
        npc = make_combatant(
            id="npc_1",
            name="NPC",
            side="hostiles",
            ai_controlled=True,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        player = make_combatant(
            id="player_1",
            name="Player",
            side="players",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario_with_turn_order([npc, player])

        targets = build_target_info_list(scenario, npc)

        assert len(targets) == 1
        assert targets[0].id == "player_1"
        assert targets[0].distance == 2
        assert targets[0].is_ally is False

    def test_build_target_info_excludes_allies(self):
        """Same-side combatants excluded from enemy list."""
        npc1 = make_combatant(
            id="npc_1",
            name="NPC 1",
            side="hostiles",
            ai_controlled=True,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        npc2 = make_combatant(
            id="npc_2",
            name="NPC 2",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario_with_turn_order([npc1, npc2])

        targets = build_target_info_list(scenario, npc1)

        assert len(targets) == 0

    def test_build_target_info_excludes_destroyed(self):
        """0 HP combatants excluded from target list."""
        npc = make_combatant(
            id="npc_1",
            name="NPC",
            side="hostiles",
            ai_controlled=True,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        player = make_combatant(
            id="player_1",
            name="Player",
            side="players",
            hp_current=0,  # Destroyed
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario_with_turn_order([npc, player])

        targets = build_target_info_list(scenario, npc)

        assert len(targets) == 0

    def test_build_target_info_excludes_out_of_range(self):
        """Beyond sensor range excluded from target list."""
        npc = make_combatant(
            id="npc_1",
            name="NPC",
            side="hostiles",
            ai_controlled=True,
            sensor_range=5,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        player = make_combatant(
            id="player_1",
            name="Player",
            side="players",
            position=HexPosition(coord=HexCoord(q=10, r=0), elevation=0),  # Distance 10 > sensor_range 5
        )
        scenario = make_scenario_with_turn_order([npc, player])

        targets = build_target_info_list(scenario, npc)

        assert len(targets) == 0


# =============================================================================
# Tests: get_npc_role
# =============================================================================


class TestGetNpcRole:
    """Tests for get_npc_role function."""

    def test_get_npc_role_returns_stored(self):
        """Returns role from combatant if set."""
        combatant = make_combatant(npc_role="defender")
        assert get_npc_role(combatant) == "defender"

    def test_get_npc_role_defaults_to_striker(self):
        """Returns 'striker' if role is None."""
        combatant = make_combatant(npc_role=None)
        assert get_npc_role(combatant) == "striker"


# =============================================================================
# Tests: execute_npc_turn
# =============================================================================


class TestExecuteNpcTurn:
    """Tests for execute_npc_turn function."""

    def test_execute_npc_turn_skips_non_ai(self):
        """Skips if not ai_controlled."""
        player = make_combatant(
            id="player_1",
            name="Player",
            side="players",
            ai_controlled=False,
        )
        scenario = make_scenario_with_turn_order([player])

        updated_scenario, result = execute_npc_turn(scenario, "player_1", 1, 0)

        assert result.skipped is True
        assert result.skip_reason == "Actor is not AI-controlled"
        assert result.actions_taken == 0

    def test_execute_npc_turn_skips_destroyed(self):
        """Skips if HP=0."""
        npc = make_combatant(
            id="npc_1",
            name="NPC",
            side="hostiles",
            ai_controlled=True,
            hp_current=0,
        )
        scenario = make_scenario_with_turn_order([npc])

        updated_scenario, result = execute_npc_turn(scenario, "npc_1", 1, 0)

        assert result.skipped is True
        assert result.skip_reason == "Actor is destroyed"

    def test_execute_npc_turn_handles_no_targets(self):
        """Skips with reason if no targets."""
        npc = make_combatant(
            id="npc_1",
            name="NPC",
            side="hostiles",
            ai_controlled=True,
        )
        # No player combatants
        scenario = make_scenario_with_turn_order([npc])

        updated_scenario, result = execute_npc_turn(scenario, "npc_1", 1, 0)

        assert result.skipped is True
        assert result.skip_reason == "No valid targets in range"

    def test_execute_npc_turn_skips_actor_not_found(self):
        """Skips if actor not found in scenario."""
        npc = make_combatant(id="npc_1", ai_controlled=True)
        scenario = make_scenario_with_turn_order([npc])

        updated_scenario, result = execute_npc_turn(scenario, "nonexistent", 1, 0)

        assert result.skipped is True
        assert result.skip_reason == "Actor not found"

    def test_execute_npc_turn_full_cycle(self):
        """Completes start→action→end when conditions are met."""
        npc = make_combatant(
            id="npc_1",
            name="NPC",
            side="hostiles",
            ai_controlled=True,
            npc_role="striker",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        player = make_combatant(
            id="player_1",
            name="Player",
            side="players",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario_with_turn_order([npc, player])

        updated_scenario, result = execute_npc_turn(scenario, "npc_1", 1, 0)

        # Turn should not be skipped
        assert result.skipped is False

        # Should have a decision
        assert result.decision is not None
        assert result.decision.target_id == "player_1"

        # Turn start and end should be present
        assert result.turn_start is not None
        assert result.turn_end is not None
