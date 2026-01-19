"""End-to-end test helpers for combat scenarios.

Provides reusable utilities for creating realistic combat scenarios with:
- Real pilots with talents
- Real mech frames with traits and core powers
- SITREP-based scenarios with terrain and objectives
- Multi-round combat execution
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch
from core.mech.combat_execution import (
    ActionExecutionInput,
    ActionExecutionResult,
    execute_action,
    start_turn,
)
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    CombatTurn,
    CombatSide,
    CombatEnvironment,
)
from core.shared.enums import ActionType
from core.mech.compendium import get_frame_definition
from core.mech.frame import (
    collect_frame_trait_effects,
    get_core_power_effects,
)
from core.mech.grid import HexPosition, HexCoord
from core.pilot import Pilot, Talent, collect_pilot_talent_effects
from core.pilot.background import Background
from core.pilot.skill import SkillSet, PilotTrigger
from core.pilot.gear import PilotLoadout
from core.shared.effects import MechanicalEffect
from core.shared.scenario import SITREP_TEMPLATES, SitrepType
from core.shared.terrain_generation import (
    TileSetType,
    TerrainGeneratorParams,
    generate_terrain_from_sitrep,
)
from core.shared.sitrep_resolution import (
    SitrepResolution,
    create_sitrep_resolution,
    advance_sitrep_round,
    update_zone_control,
    check_victory_conditions,
)
from core.gm_toolkit.encounter_builder import (
    EncounterDifficulty,
    estimate_party_power,
    calculate_enemy_force,
)


# =============================================================================
# Pilot Factories
# =============================================================================


def make_pilot_with_talents(
    callsign: str,
    talents: list[tuple[str, int]],
    level: int | None = None,
    skills: SkillSet | None = None,
) -> Pilot:
    """Create a pilot with specified talents.

    Args:
        callsign: Pilot callsign
        talents: List of (talent_id, rank) tuples
        level: License level (defaults to sum of talent ranks)
        skills: Pilot skills (defaults to balanced skills)

    Returns:
        Pilot configured with the specified talents
    """
    talent_objs = [Talent(talent_id=tid, rank=rank) for tid, rank in talents]

    if level is None:
        level = sum(rank for _, rank in talents)

    if skills is None:
        skills = SkillSet(hull=2, agility=2, systems=2, engineering=2)

    return Pilot(
        id=f"pilot_{callsign.lower()}",
        callsign=callsign,
        name="",
        background=Background(
            id="background_test",
            name="Test Background",
            triggers=["read_a_situation", "spot", "take_someone_out", "survive"],
        ),
        level=level,
        skills=skills,
        triggers=[
            PilotTrigger(trigger_id="read_a_situation", rank=2),
            PilotTrigger(trigger_id="spot", rank=2),
        ],
        talents=talent_objs,
        pilot_gear=PilotLoadout(
            clothing="flight_suit",
            armor="light_hardsuit",
            weapons=["signature_weapon_combat"],
            gear=[],
        ),
    )


def make_ace_pilot() -> Pilot:
    """Create a pilot with ACE talent (flying bonuses)."""
    return make_pilot_with_talents("ACE", [("ace", 2)])


def make_crack_shot_pilot() -> Pilot:
    """Create a pilot with CRACK SHOT talent (ranged accuracy)."""
    return make_pilot_with_talents("SHARPSHOOTER", [("crack_shot", 2)])


def make_combined_arms_pilot() -> Pilot:
    """Create a pilot with COMBINED ARMS talent (CQC bonus)."""
    return make_pilot_with_talents("CQC", [("combined_arms", 2)])


def make_tactician_pilot() -> Pilot:
    """Create a pilot with TACTICIAN talent (positioning bonuses)."""
    return make_pilot_with_talents("TACTICIAN", [("tactician", 2)])


def make_leader_pilot() -> Pilot:
    """Create a pilot with LEADER talent (dice pool support)."""
    return make_pilot_with_talents("COMMANDER", [("leader", 2)])


def make_multi_talent_pilot() -> Pilot:
    """Create a pilot with multiple talents for complex testing."""
    return make_pilot_with_talents(
        "VETERAN",
        [
            ("crack_shot", 2),
            ("combined_arms", 1),
            ("tactician", 1),
        ],
        level=4,
    )


# =============================================================================
# Combatant Factories
# =============================================================================


def make_combatant(
    id: str = "mech_1",
    name: str = "Test Mech",
    side: CombatSide = "players",
    hp_max: int = 10,
    hp_current: int | None = None,
    evasion: int = 8,
    armor: int = 0,
    q: int = 0,
    r: int = 0,
    talent_effects: list[MechanicalEffect] | None = None,
    frame_trait_effects: list[MechanicalEffect] | None = None,
    core_power_available: bool = True,
    core_power_active: bool = False,
    core_power_effects: MechanicalEffect | None = None,
    **kwargs,
) -> CombatantState:
    """Create a test combatant with full configuration options.

    Args:
        id: Unique combatant identifier
        name: Display name
        side: Combat side ("players", "hostiles", "neutral")
        hp_max: Maximum hit points
        hp_current: Current HP (defaults to hp_max)
        evasion: Evasion stat
        armor: Armor value
        q, r: Hex coordinates
        talent_effects: Effects from pilot talents
        frame_trait_effects: Effects from frame traits
        core_power_available: Whether core power can be used
        core_power_active: Whether core power is currently active
        core_power_effects: Effects from core power

    Returns:
        Configured CombatantState
    """
    return CombatantState(
        id=id,
        name=name,
        side=side,
        kind="mech",
        stats=CombatStats(
            size=kwargs.pop("size", "size_1"),
            hp_max=hp_max,
            evasion=evasion,
            e_defense=kwargs.pop("e_defense", 8),
            armor=armor,
            speed=kwargs.pop("speed", 4),
            sensor_range=kwargs.pop("sensor_range", 10),
            tech_attack=kwargs.pop("tech_attack", 0),
            grit=kwargs.pop("grit", 0),
        ),
        resources=CombatResources(
            hp_current=hp_current if hp_current is not None else hp_max,
            heat_current=kwargs.pop("heat_current", 0),
            heat_cap=kwargs.pop("heat_cap", 6),
            structure_current=kwargs.pop("structure_current", 4),
            stress_current=kwargs.pop("stress_current", 4),
            repairs_remaining=kwargs.pop("repairs_remaining", 4),
        ),
        position=HexPosition(coord=HexCoord(q=q, r=r), elevation=kwargs.pop("elevation", 0)),
        talent_effects=talent_effects or [],
        frame_trait_effects=frame_trait_effects or [],
        core_power_available=core_power_available,
        core_power_active=core_power_active,
        core_power_effects=core_power_effects,
        **kwargs,
    )


def make_combatant_from_pilot(
    pilot: Pilot,
    frame_id: str = "gms_everest",
    position: tuple[int, int] = (0, 0),
    side: CombatSide = "players",
    combatant_id: str | None = None,
) -> CombatantState:
    """Create a combatant from a real pilot with talent effects.

    Args:
        pilot: Pilot with talents to use
        frame_id: Frame ID to use for stats and traits
        position: (q, r) hex coordinates
        side: Combat side

    Returns:
        CombatantState with pilot talents and frame traits applied
    """
    frame = get_frame_definition(frame_id)
    if frame is None:
        raise ValueError(f"Frame {frame_id} not found")

    # Collect effects
    talent_effects = collect_pilot_talent_effects(pilot)
    frame_trait_effects = collect_frame_trait_effects(frame)
    core_power_effects = get_core_power_effects(frame)

    # Calculate stats from frame and pilot skills
    base_stats = frame.base_stats
    grit = pilot.level // 2

    cid = combatant_id or f"combat_{pilot.id}"

    return CombatantState(
        id=cid,
        name=f"{pilot.callsign}'s {frame.name}",
        side=side,
        kind="mech",
        stats=CombatStats(
            size=base_stats.size,
            hp_max=base_stats.hp + (pilot.skills.hull * 2),
            evasion=base_stats.evasion,
            e_defense=base_stats.e_defense,
            armor=base_stats.armor,
            speed=base_stats.speed,
            sensor_range=base_stats.sensor_range,
            tech_attack=base_stats.tech_attack,
            grit=grit,
        ),
        resources=CombatResources(
            hp_current=base_stats.hp + (pilot.skills.hull * 2),
            heat_current=0,
            heat_cap=base_stats.heat_cap + pilot.skills.engineering,
            structure_current=base_stats.structure,
            stress_current=4,  # Default stress value
            repairs_remaining=base_stats.repair_cap,
        ),
        position=HexPosition(
            coord=HexCoord(q=position[0], r=position[1]),
            elevation=0,
        ),
        talent_effects=talent_effects,
        frame_trait_effects=frame_trait_effects,
        core_power_available=core_power_effects is not None,
        core_power_active=False,
        core_power_effects=core_power_effects,
    )


def make_enemy_combatant(
    id: str = "hostile_1",
    name: str = "Enemy Mech",
    hp_max: int = 8,
    evasion: int = 8,
    armor: int = 0,
    q: int = 5,
    r: int = 0,
    **kwargs,
) -> CombatantState:
    """Create a hostile combatant for testing."""
    return make_combatant(
        id=id,
        name=name,
        side="hostiles",
        hp_max=hp_max,
        evasion=evasion,
        armor=armor,
        q=q,
        r=r,
        **kwargs,
    )


# =============================================================================
# Scenario Factories
# =============================================================================


def make_scenario(
    combatants: list[CombatantState] | None = None,
    terrain: Any = None,
    environment: CombatEnvironment = "standard",
    sitrep_resolution: SitrepResolution | None = None,
) -> MechCombatScenario:
    """Create a basic test scenario.

    Args:
        combatants: List of combatants
        terrain: Optional terrain map
        environment: Environment type
        sitrep_resolution: Optional SITREP tracking

    Returns:
        Configured MechCombatScenario
    """
    return MechCombatScenario(
        combatants=combatants or [],
        grapples=[],
        rounds=[],
        terrain=terrain,
        environment=environment,
        deployables={},
        sitrep_resolution=sitrep_resolution,
    )


def make_sitrep_scenario(
    sitrep_type: SitrepType,
    player_combatants: list[CombatantState],
    enemy_combatants: list[CombatantState],
    tile_set: TileSetType = "urban",
    seed: int = 42,
    map_width: int = 20,
    map_height: int = 16,
) -> MechCombatScenario:
    """Create a scenario with SITREP objectives and terrain.

    Args:
        sitrep_type: SITREP type (control, escort, etc.)
        player_combatants: Player combatants
        enemy_combatants: Enemy combatants
        tile_set: Terrain tile set
        seed: Random seed for terrain
        map_width: Map width in hexes
        map_height: Map height in hexes

    Returns:
        Full scenario with terrain, SITREP resolution, and deployment
    """
    template = SITREP_TEMPLATES[sitrep_type]

    # Generate terrain
    params = TerrainGeneratorParams(
        map_width=map_width,
        map_height=map_height,
        sitrep_template=template,
        tile_set=tile_set,
        seed=seed,
    )
    generated = generate_terrain_from_sitrep(template, params)

    # Create SITREP resolution
    reserve_ids = [e.id for e in enemy_combatants[len(enemy_combatants)//2:]] if enemy_combatants else []
    resolution = create_sitrep_resolution(
        template=template,
        player_count=len(player_combatants),
        reserve_ids=reserve_ids,
        enemy_count=len(enemy_combatants),
    )

    # Combine combatants
    all_combatants = list(player_combatants) + list(enemy_combatants)

    return MechCombatScenario(
        combatants=all_combatants,
        terrain=generated.terrain_map,
        sitrep_resolution=resolution,
        environment="standard",
        grapples=[],
        rounds=[],
        deployables={},
    )


def make_duel_scenario(
    attacker: CombatantState,
    defender: CombatantState,
    environment: CombatEnvironment = "standard",
) -> MechCombatScenario:
    """Create a simple 1v1 duel scenario."""
    return make_scenario(
        combatants=[attacker, defender],
        environment=environment,
    )


def make_skirmish_scenario(
    players: list[CombatantState],
    hostiles: list[CombatantState],
) -> MechCombatScenario:
    """Create a multi-combatant skirmish scenario."""
    return make_scenario(combatants=players + hostiles)


# =============================================================================
# Combat Execution Helpers
# =============================================================================


def execute_turn(
    scenario: MechCombatScenario,
    actor_id: str,
    actions: list[tuple[str, ActionType, dict[str, Any] | None]],
) -> tuple[MechCombatScenario, list[ActionExecutionResult]]:
    """Execute a complete turn with multiple actions.

    Args:
        scenario: Current scenario state
        actor_id: ID of the acting combatant
        actions: List of (action_id, action_type, params) tuples

    Returns:
        (updated scenario, list of results)
    """
    scenario, turn_result = start_turn(scenario, actor_id)
    economy = turn_result.economy
    turn = CombatTurn(actor_id=actor_id)

    results = []
    for action_id, action_type, params in actions:
        action_input = ActionExecutionInput(
            actor_id=actor_id,
            action_id=action_id,
            action_type=action_type,
            **(params or {}),
        )
        scenario, turn, economy, result = execute_action(
            scenario, turn, economy, action_input
        )
        results.append(result)

    # Note: Not calling end_turn() here as it requires round tracking
    # For simple E2E tests, start_turn + execute_action is sufficient
    return scenario, results


def execute_attack(
    scenario: MechCombatScenario,
    attacker_id: str,
    target_id: str,
    action_id: str = "skirmish",
    action_type: ActionType = "quick",
    weapon_id: str | None = None,
    force_roll: int | None = None,
) -> tuple[MechCombatScenario, ActionExecutionResult]:
    """Execute a single attack action.

    Args:
        scenario: Current scenario
        attacker_id: Attacking combatant ID
        target_id: Target combatant ID
        action_id: Action to use (default: skirmish)
        action_type: Action type (default: quick)
        weapon_id: Optional specific weapon ID
        force_roll: Optional forced d20 roll value

    Returns:
        (updated scenario, result)
    """
    scenario, turn_result = start_turn(scenario, attacker_id)
    economy = turn_result.economy
    turn = CombatTurn(actor_id=attacker_id)

    params: dict[str, Any] = {"target_ids": [target_id]}
    if weapon_id:
        params["weapon_id"] = weapon_id

    action_input = ActionExecutionInput(
        actor_id=attacker_id,
        action_id=action_id,
        action_type=action_type,
        **params,
    )

    if force_roll is not None:
        with patch("core.shared.rolls._roll_d20") as mock:
            mock.return_value = force_roll
            scenario, turn, economy, result = execute_action(
                scenario, turn, economy, action_input
            )
    else:
        scenario, turn, economy, result = execute_action(
            scenario, turn, economy, action_input
        )

    return scenario, result


def execute_full_round(
    scenario: MechCombatScenario,
    turn_order: list[str] | None = None,
    default_action: tuple[str, ActionType] = ("skirmish", "quick"),
) -> tuple[MechCombatScenario, dict[str, list[ActionExecutionResult]]]:
    """Execute a full round with all combatants acting.

    Args:
        scenario: Current scenario
        turn_order: Optional explicit turn order (defaults to all combatants)
        default_action: Default (action_id, action_type) for each combatant

    Returns:
        (updated scenario, dict of combatant_id -> results)
    """
    if turn_order is None:
        turn_order = [c.id for c in scenario.combatants if c.resources.hp_current > 0]

    all_results: dict[str, list[ActionExecutionResult]] = {}

    for actor_id in turn_order:
        actor = next((c for c in scenario.combatants if c.id == actor_id), None)
        if actor is None or actor.resources.hp_current <= 0:
            continue

        # Find a target
        target = next(
            (c for c in scenario.combatants
             if c.side != actor.side and c.resources.hp_current > 0),
            None,
        )

        if target is None:
            continue

        scenario, result = execute_attack(
            scenario,
            actor_id,
            target.id,
            action_id=default_action[0],
            action_type=default_action[1],
        )
        all_results[actor_id] = [result]

    return scenario, all_results


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_combatant_hp(
    scenario: MechCombatScenario,
    combatant_id: str,
    expected_hp: int,
    msg: str = "",
) -> None:
    """Assert a combatant has specific HP."""
    combatant = next((c for c in scenario.combatants if c.id == combatant_id), None)
    assert combatant is not None, f"Combatant {combatant_id} not found"
    actual_hp = combatant.resources.hp_current
    assert actual_hp == expected_hp, (
        f"{msg}: Expected {combatant_id} HP to be {expected_hp}, got {actual_hp}"
    )


def assert_combatant_alive(
    scenario: MechCombatScenario,
    combatant_id: str,
) -> None:
    """Assert a combatant is still alive (HP > 0)."""
    combatant = next((c for c in scenario.combatants if c.id == combatant_id), None)
    assert combatant is not None, f"Combatant {combatant_id} not found"
    assert combatant.resources.hp_current > 0, f"Combatant {combatant_id} is dead"


def assert_combatant_destroyed(
    scenario: MechCombatScenario,
    combatant_id: str,
) -> None:
    """Assert a combatant has been destroyed (HP <= 0)."""
    combatant = next((c for c in scenario.combatants if c.id == combatant_id), None)
    assert combatant is not None, f"Combatant {combatant_id} not found"
    assert combatant.resources.hp_current <= 0, (
        f"Combatant {combatant_id} is still alive with {combatant.resources.hp_current} HP"
    )


def assert_attack_hit(result: ActionExecutionResult) -> None:
    """Assert an attack action resulted in a hit."""
    assert result.success, f"Action failed: {result.error}"
    assert result.effects_applied, "No effects applied"
    hit = result.effects_applied[0].get("hit", False)
    assert hit, "Attack missed"


def assert_attack_missed(result: ActionExecutionResult) -> None:
    """Assert an attack action resulted in a miss."""
    assert result.success, f"Action failed: {result.error}"
    assert result.effects_applied, "No effects applied"
    hit = result.effects_applied[0].get("hit", False)
    assert not hit, "Attack hit (expected miss)"


def assert_attack_crit(result: ActionExecutionResult) -> None:
    """Assert an attack was a critical hit."""
    assert result.success, f"Action failed: {result.error}"
    assert result.effects_applied, "No effects applied"
    crit = result.effects_applied[0].get("critical", False)
    assert crit, "Attack was not a critical hit"


def count_alive_on_side(scenario: MechCombatScenario, side: CombatSide) -> int:
    """Count alive combatants on a side."""
    return sum(
        1 for c in scenario.combatants
        if c.side == side and c.resources.hp_current > 0
    )


def get_total_damage_dealt(results: list[ActionExecutionResult]) -> int:
    """Sum total damage from a list of action results."""
    return sum(r.damage_dealt for r in results)


# =============================================================================
# SITREP Helpers
# =============================================================================


def update_sitrep_zone_for_side(
    scenario: MechCombatScenario,
    zone_id: str,
    side: str,
) -> MechCombatScenario:
    """Update a zone's control state for a side.

    Args:
        scenario: Current scenario
        zone_id: Zone to update
        side: "players" or "enemies"

    Returns:
        Updated scenario with zone control changed
    """
    if scenario.sitrep_resolution is None:
        return scenario

    state = "player_controlled" if side == "players" else "enemy_controlled"
    new_resolution = update_zone_control(
        scenario.sitrep_resolution,
        zone_id,
        state,
        side,
    )

    return scenario.model_copy(update={"sitrep_resolution": new_resolution})


def advance_scenario_round(scenario: MechCombatScenario) -> MechCombatScenario:
    """Advance the SITREP to the next round."""
    if scenario.sitrep_resolution is None:
        return scenario

    new_resolution = advance_sitrep_round(scenario.sitrep_resolution)
    return scenario.model_copy(update={"sitrep_resolution": new_resolution})


def check_scenario_victory(scenario: MechCombatScenario) -> MechCombatScenario:
    """Check victory conditions on the scenario."""
    if scenario.sitrep_resolution is None:
        return scenario

    new_resolution = check_victory_conditions(scenario.sitrep_resolution)
    return scenario.model_copy(update={"sitrep_resolution": new_resolution})


def get_sitrep_outcome(scenario: MechCombatScenario) -> str | None:
    """Get the SITREP victory outcome."""
    if scenario.sitrep_resolution is None:
        return None
    return scenario.sitrep_resolution.outcome
