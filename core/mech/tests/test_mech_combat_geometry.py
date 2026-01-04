from core.mech.combat_rules import AttackPatternDefinition
from core.mech.combat_state import (
    ActionUse,
    CombatantState,
    CombatResources,
    CombatRound,
    CombatStats,
    CombatTurn,
    MechCombatScenario,
)
from core.mech.combat_validation import validate_combat_scenario
from core.mech.grid import HexCoord, HexPosition
from core.mech.terrain import TerrainHex, TerrainMap


def _combatant(combatant_id: str, coord: HexCoord) -> CombatantState:
    return CombatantState(
        id=combatant_id,
        name=combatant_id.title(),
        side="players" if combatant_id == "alpha" else "hostiles",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=10,
            evasion=8,
            e_defense=8,
            armor=0,
            speed=4,
            sensor_range=10,
            tech_attack=0,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=HexPosition(coord=coord, elevation=0),
        statuses=[],
        conditions=[],
    )


def _scenario(
    action: ActionUse,
    target_coord: HexCoord,
    terrain: TerrainMap | None = None,
) -> MechCombatScenario:
    alpha = _combatant("alpha", HexCoord(q=0, r=0))
    bravo = _combatant("bravo", target_coord)
    return MechCombatScenario(
        combatants=[alpha, bravo],
        rounds=[
            CombatRound(
                round_index=1,
                turns=[
                    CombatTurn(
                        actor_id="alpha",
                        move_used=False,
                        actions=[action],
                    )
                ],
            )
        ],
        terrain=terrain,
    )


def test_line_target_outside_shape_warns() -> None:
    action = ActionUse(
        action_id="skirmish",
        action_type="quick",
        target_id="bravo",
        area_pattern=AttackPatternDefinition(pattern="line", size=3),
        area_direction=HexCoord(q=1, r=0),
        range_spaces=5,
    )
    scenario = _scenario(action, HexCoord(q=0, r=1))
    validation = validate_combat_scenario(scenario)
    codes = {issue.code for issue in validation.issues}
    assert "area_out_of_bounds" in codes


def test_cone_area_affected_outside_shape_warns() -> None:
    action = ActionUse(
        action_id="skirmish",
        action_type="quick",
        target_id="bravo",
        area_pattern=AttackPatternDefinition(pattern="cone", size=2),
        area_direction=HexCoord(q=1, r=0),
        area_affected=[HexCoord(q=0, r=1)],
        range_spaces=5,
    )
    scenario = _scenario(action, HexCoord(q=2, r=0))
    validation = validate_combat_scenario(scenario)
    codes = {issue.code for issue in validation.issues}
    assert "area_affected_not_in_shape" in codes


def test_cone_axis_mode_allows_centered_offsets() -> None:
    action = ActionUse(
        action_id="skirmish",
        action_type="quick",
        target_id="bravo",
        area_pattern=AttackPatternDefinition(pattern="cone", size=2, cone_mode="axis"),
        area_direction=HexCoord(q=1, r=0),
        area_affected=[HexCoord(q=2, r=-1)],
        range_spaces=5,
    )
    scenario = _scenario(action, HexCoord(q=2, r=0))
    validation = validate_combat_scenario(scenario)
    codes = {issue.code for issue in validation.issues}
    assert "area_affected_not_in_shape" not in codes


def test_blast_origin_out_of_range_warns() -> None:
    action = ActionUse(
        action_id="skirmish",
        action_type="quick",
        range_spaces=3,
        area_pattern=AttackPatternDefinition(pattern="blast", size=1),
        area_origin=HexPosition(coord=HexCoord(q=4, r=0), elevation=0),
    )
    scenario = _scenario(action, HexCoord(q=0, r=1))
    validation = validate_combat_scenario(scenario)
    codes = {issue.code for issue in validation.issues}
    assert "area_origin_range_exceeded" in codes


def test_blast_origin_line_of_sight_blocked_warns() -> None:
    action = ActionUse(
        action_id="skirmish",
        action_type="quick",
        range_spaces=5,
        area_pattern=AttackPatternDefinition(pattern="blast", size=1),
        area_origin=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
    )
    terrain = TerrainMap(
        tiles=[TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True)]
    )
    scenario = _scenario(action, HexCoord(q=0, r=1), terrain=terrain)
    validation = validate_combat_scenario(scenario)
    codes = {issue.code for issue in validation.issues}
    assert "area_origin_line_of_sight_blocked" in codes


def test_strict_mode_promotes_area_origin_warning() -> None:
    action = ActionUse(
        action_id="skirmish",
        action_type="quick",
        target_id="bravo",
        range_spaces=5,
        area_pattern=AttackPatternDefinition(pattern="blast", size=1),
        area_origin=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
    )
    terrain = TerrainMap(
        tiles=[TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True)]
    )
    scenario = _scenario(action, HexCoord(q=2, r=1), terrain=terrain)
    validation = validate_combat_scenario(scenario, strict=True)
    issue = next(
        issue
        for issue in validation.issues
        if issue.code == "area_origin_line_of_sight_blocked"
    )
    assert issue.severity == "error"


def test_arcing_area_origin_path_blocked_warns() -> None:
    action = ActionUse(
        action_id="skirmish",
        action_type="quick",
        range_spaces=5,
        weapon_tags=["arcing"],
        area_pattern=AttackPatternDefinition(pattern="blast", size=1),
        area_origin=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
    )
    terrain = TerrainMap(
        tiles=[TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True)]
    )
    scenario = _scenario(action, HexCoord(q=0, r=1), terrain=terrain)
    validation = validate_combat_scenario(scenario)
    codes = {issue.code for issue in validation.issues}
    assert "area_origin_path_blocked" in codes


def test_line_area_blocked_by_line_of_sight_warns() -> None:
    action = ActionUse(
        action_id="skirmish",
        action_type="quick",
        area_pattern=AttackPatternDefinition(pattern="line", size=3),
        area_direction=HexCoord(q=1, r=0),
        area_affected=[HexCoord(q=2, r=0)],
        range_spaces=5,
    )
    terrain = TerrainMap(
        tiles=[TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True)]
    )
    scenario = _scenario(action, HexCoord(q=0, r=1), terrain=terrain)
    validation = validate_combat_scenario(scenario)
    codes = {issue.code for issue in validation.issues}
    assert "area_line_of_sight_blocked" in codes
