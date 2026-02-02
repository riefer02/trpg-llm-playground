"""Converter functions for transforming core models to UI DTOs.

These functions extract and pre-compute data from MechCombatScenario
to create CombatUIState for efficient frontend rendering.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from app.backend.api.dtos.combat_ui import (
    ActionEconomyBrief,
    ActionFeedEntry,
    CombatantBrief,
    CombatSide,
    CombatUIState,
    CurrentActorState,
    DeployableBrief,
    MovementRangeData,
    ObjectiveBrief,
    PendingDecisionBrief,
    SessionStatus,
)

if TYPE_CHECKING:
    from core.mech.combat_state import (
        CombatantState,
        CombatRound,
        DeployableState,
        MechCombatScenario,
    )
    from core.mech.action_economy import ActionEconomyState
    from core.shared.decisions import PendingDecision


# Maximum number of recent actions to include in the DTO
MAX_RECENT_ACTIONS = 50


def build_combatant_brief(combatant: "CombatantState") -> CombatantBrief:
    """Extract minimal combatant data for UI rendering.

    Args:
        combatant: Full combatant state from core model

    Returns:
        CombatantBrief with only UI-essential fields
    """
    # Check if destroyed based on HP
    is_destroyed = combatant.resources.hp_current <= 0

    # Convert position to HexCoord if available
    position = combatant.position.coord if combatant.position else None

    return CombatantBrief(
        id=combatant.id,
        name=combatant.name,
        side=combatant.side,  # type: ignore[arg-type]
        frame_id=combatant.frame_id,
        position=position,
        hp_current=combatant.resources.hp_current,
        hp_max=combatant.stats.hp_max,
        heat_current=combatant.resources.heat_current,
        heat_cap=combatant.resources.heat_cap,
        structure_current=combatant.resources.structure_current,
        stress_current=combatant.resources.stress_current,
        statuses=[str(s) for s in combatant.statuses],
        is_destroyed=is_destroyed,
        is_ai_controlled=combatant.ai_controlled,
        speed=combatant.stats.speed,
        evasion=combatant.stats.evasion,
        e_defense=combatant.stats.e_defense,
        armor=combatant.stats.armor,
    )


def build_economy_brief(economy: "ActionEconomyState") -> ActionEconomyBrief:
    """Convert action economy state to brief form.

    Args:
        economy: Full economy state from core

    Returns:
        ActionEconomyBrief with computed remaining values
    """
    return ActionEconomyBrief(
        full_actions_remaining=economy.full_actions_remaining,
        quick_actions_remaining=economy.quick_actions_remaining,
        can_overcharge=economy.can_overcharge,
        reactions_remaining=economy.reactions_remaining_this_turn,
        overcharge_used=economy.overcharge_used,
        move_used=False,  # Move tracking not in economy state
    )


def build_current_actor_state(
    combatant: "CombatantState",
    economy: "ActionEconomyState | None" = None,
) -> CurrentActorState:
    """Build current actor state with economy info.

    Args:
        combatant: The current actor combatant
        economy: Optional economy state (defaults if not provided)

    Returns:
        CurrentActorState with pre-computed info
    """
    from core.mech.action_economy import ActionEconomyState

    if economy is None:
        economy = ActionEconomyState()

    # Player controlled if on players side and not AI controlled
    is_player_controlled = combatant.side == "players" and not combatant.ai_controlled

    return CurrentActorState(
        actor_id=combatant.id,
        actor_name=combatant.name,
        frame_id=combatant.frame_id,
        side=combatant.side,  # type: ignore[arg-type]
        is_player_controlled=is_player_controlled,
        economy=build_economy_brief(economy),
    )


def build_action_feed_entries(
    rounds: list["CombatRound"],
    combatant_names: dict[str, str],
    combatant_sides: dict[str, CombatSide],
    max_entries: int = MAX_RECENT_ACTIONS,
) -> tuple[list[ActionFeedEntry], int]:
    """Flatten rounds/turns/actions into feed entries.

    Args:
        rounds: Combat rounds from scenario
        combatant_names: Pre-built name lookup map
        combatant_sides: Pre-built side lookup map
        max_entries: Maximum entries to return

    Returns:
        Tuple of (recent entries most-recent-first, total action count)
    """
    all_entries: list[ActionFeedEntry] = []

    for round_idx, combat_round in enumerate(rounds):
        round_number = round_idx + 1

        for turn_idx, turn in enumerate(combat_round.turns):
            actor_id = turn.actor_id
            actor_name = combatant_names.get(actor_id, actor_id)
            actor_side = combatant_sides.get(actor_id, "neutral")

            for action_idx, action in enumerate(turn.actions):
                entry_id = f"{round_idx}-{turn_idx}-{action_idx}"
                timestamp = round_number * 1000 + turn_idx * 10 + action_idx

                # Build target names
                target_names: list[str] = []
                if action.target_ids:
                    target_names = [
                        combatant_names.get(tid, tid) for tid in action.target_ids
                    ]
                elif action.target_id:
                    target_names = [combatant_names.get(action.target_id, action.target_id)]

                # Extract damage and effects from log
                damage_dealt: int | None = None
                heat_dealt: int | None = None
                statuses_applied: list[str] = []

                for effect in action.log_effects or []:
                    if effect.type == "damage" and effect.amount:
                        damage_dealt = (damage_dealt or 0) + effect.amount
                    elif effect.type == "heat" and effect.amount:
                        heat_dealt = (heat_dealt or 0) + effect.amount
                    elif effect.type == "status_applied" and effect.status:
                        statuses_applied.append(str(effect.status).replace("_", " "))

                # Generate action name from ID
                action_name = action.action_id.replace("_", " ").title()

                all_entries.append(
                    ActionFeedEntry(
                        id=entry_id,
                        round_number=round_number,
                        actor_id=actor_id,
                        actor_name=actor_name,
                        actor_side=actor_side,
                        action_id=action.action_id,
                        action_name=action_name,
                        target_names=target_names,
                        damage_dealt=damage_dealt if damage_dealt and damage_dealt > 0 else None,
                        heat_dealt=heat_dealt if heat_dealt and heat_dealt > 0 else None,
                        statuses_applied=statuses_applied,
                        timestamp=timestamp,
                    )
                )

    total_count = len(all_entries)

    # Return most recent first, limited to max_entries
    recent_entries = list(reversed(all_entries))[:max_entries]

    return recent_entries, total_count


def build_pending_decision_brief(
    decision: "PendingDecision",
    combatant_names: dict[str, str],
) -> PendingDecisionBrief:
    """Convert pending decision to brief form.

    Args:
        decision: Full pending decision from core
        combatant_names: Name lookup map

    Returns:
        PendingDecisionBrief for UI prompts
    """
    return PendingDecisionBrief(
        decision_id=decision.decision_id,
        decision_type=decision.decision_type,  # type: ignore[arg-type]
        combatant_id=decision.combatant_id,
        combatant_name=combatant_names.get(decision.combatant_id, decision.combatant_id),
        trigger_source=decision.trigger_source,
        save_target=decision.save_target,
        eligible_mounts=list(decision.eligible_mounts),
        eligible_systems=list(decision.eligible_systems),
    )


def build_deployable_brief(
    deployable_id: str,
    deployable: "DeployableState",
) -> DeployableBrief:
    """Convert deployable state to brief form.

    Args:
        deployable_id: ID of the deployable
        deployable: Full deployable state

    Returns:
        DeployableBrief for rendering
    """
    return DeployableBrief(
        id=deployable_id,
        name=deployable.name,
        kind=deployable.kind,  # type: ignore[arg-type]
        owner_id=deployable.owner_id,
        position=deployable.position.coord,
        hp=deployable.hp,
        max_hp=deployable.max_hp,
        is_armed=deployable.is_armed,
        is_destroyed=deployable.is_destroyed,
    )


def build_objective_brief(objective) -> ObjectiveBrief:
    """Convert mission objective to brief form.

    Args:
        objective: Mission objective from scenario

    Returns:
        ObjectiveBrief for UI display
    """
    return ObjectiveBrief(
        objective_id=objective.objective_id,
        name=objective.name,
        description=objective.description,
        status=objective.status,  # type: ignore[arg-type]
        is_optional=objective.is_optional,
        is_primary=objective.is_primary,
    )


def build_terrain_hash(scenario: "MechCombatScenario") -> str | None:
    """Generate hash for terrain cache invalidation.

    Args:
        scenario: Combat scenario with terrain

    Returns:
        Hash string or None if no terrain
    """
    if not scenario.terrain or not scenario.terrain.tiles:
        return None

    # Hash based on tile count and first/last tile coords
    tiles = scenario.terrain.tiles
    tile_data = f"{len(tiles)}"
    if tiles:
        first = tiles[0]
        last = tiles[-1]
        tile_data += f":{first.coord.q},{first.coord.r}:{last.coord.q},{last.coord.r}"

    return hashlib.md5(tile_data.encode()).hexdigest()[:8]


def build_turn_order(scenario: "MechCombatScenario", current_round: int) -> list[str]:
    """Extract turn order for initiative display.

    Args:
        scenario: Combat scenario
        current_round: Current round number (1-indexed)

    Returns:
        List of combatant IDs in turn order
    """
    round_idx = current_round - 1
    if round_idx < 0 or round_idx >= len(scenario.rounds):
        # If no rounds yet, return combatants ordered by side (players first)
        players = [c.id for c in scenario.combatants if c.side == "players"]
        enemies = [c.id for c in scenario.combatants if c.side != "players"]
        return players + enemies

    combat_round = scenario.rounds[round_idx]
    return [turn.actor_id for turn in combat_round.turns]


def build_combat_ui_state(
    scenario: "MechCombatScenario",
    session_id: str,
    current_round: int,
    current_turn_index: int,
    status: SessionStatus,
    economy: "ActionEconomyState | None" = None,
    pending_decisions: list["PendingDecision"] | None = None,
    mission_name: str | None = None,
    tile_set: str | None = None,
    include_movement_range: bool = True,
) -> CombatUIState:
    """Build complete UI state from scenario and session data.

    This is the main entry point for creating CombatUIState.
    It pre-computes all lookups and flattens nested data.

    Args:
        scenario: Full MechCombatScenario
        session_id: Combat session ID
        current_round: Current round number (1-indexed)
        current_turn_index: Current turn index within round (0-indexed)
        status: Session status
        economy: Optional action economy state for current actor
        pending_decisions: Optional list of pending decisions
        mission_name: Optional mission name for display
        tile_set: Optional terrain tileset for rendering
        include_movement_range: Whether to pre-compute movement range for player turns

    Returns:
        CombatUIState ready for API response
    """
    from core.mech.combat_execution import get_current_actor

    # Build lookup maps first (frontend no longer needs to compute these)
    combatant_names: dict[str, str] = {}
    combatant_sides: dict[str, CombatSide] = {}
    for combatant in scenario.combatants:
        combatant_names[combatant.id] = combatant.name
        combatant_sides[combatant.id] = combatant.side  # type: ignore[assignment]

    # Build combatant briefs
    combatants = [build_combatant_brief(c) for c in scenario.combatants]

    # Get current actor
    current_actor_state: CurrentActorState | None = None
    is_player_turn = False
    movement_range: MovementRangeData | None = None

    actor = get_current_actor(scenario, current_round, current_turn_index)
    if actor:
        current_actor_state = build_current_actor_state(actor, economy)
        is_player_turn = actor.side == "players" and not actor.ai_controlled

        # Pre-compute movement range for player turns
        if is_player_turn and include_movement_range and actor.stats.speed > 0:
            movement_range = build_movement_range(
                actor_id=actor.id,
                speed=actor.stats.speed,
                scenario=scenario,
            )

    # Build action feed
    recent_actions, total_action_count = build_action_feed_entries(
        list(scenario.rounds),
        combatant_names,
        combatant_sides,
    )

    # Build pending decisions
    decision_briefs: list[PendingDecisionBrief] = []
    if pending_decisions:
        decision_briefs = [
            build_pending_decision_brief(d, combatant_names)
            for d in pending_decisions
        ]

    # Build deployables
    deployables = [
        build_deployable_brief(did, d)
        for did, d in scenario.deployables.items()
        if not d.is_destroyed
    ]

    # Build objectives
    objectives = [build_objective_brief(o) for o in scenario.objectives]

    # Build turn order
    turn_order = build_turn_order(scenario, current_round)

    # Build terrain hash
    terrain_hash = build_terrain_hash(scenario)

    return CombatUIState(
        session_id=session_id,
        current_round=current_round,
        current_turn_index=current_turn_index,
        status=status,
        combatant_names=combatant_names,
        combatant_sides=combatant_sides,
        current_actor=current_actor_state,
        is_player_turn=is_player_turn,
        pending_decisions=decision_briefs,
        combatants=combatants,
        terrain_hash=terrain_hash,
        deployables=deployables,
        objectives=objectives,
        recent_actions=recent_actions,
        total_action_count=total_action_count,
        turn_order=turn_order,
        movement_range=movement_range,
        mission_name=mission_name,
        tile_set=tile_set,
    )


def build_movement_range(
    actor_id: str,
    speed: int,
    scenario: "MechCombatScenario",
) -> MovementRangeData:
    """Build movement range data for an actor.

    This pre-computes reachable hexes using BFS pathfinding that accounts
    for terrain costs, eliminating the need for frontend hex calculations.

    Args:
        actor_id: ID of the moving combatant
        speed: Movement speed (spaces)
        scenario: Combat scenario for terrain/blocking info

    Returns:
        MovementRangeData with pre-computed hex sets
    """
    from core.mech.grid import HexCoord

    # Find actor position
    actor = next((c for c in scenario.combatants if c.id == actor_id), None)
    if not actor or not actor.position:
        return MovementRangeData(
            actor_id=actor_id,
            max_range=speed,
            reachable_hexes=[],
            blocked_hexes=[],
            difficult_hexes=[],
        )

    origin = actor.position.coord

    # Build blocked hexes (occupied by other combatants, not destroyed)
    blocked_hexes: list[HexCoord] = []
    blocked_set: set[tuple[int, int]] = set()
    for combatant in scenario.combatants:
        if combatant.id != actor_id and combatant.position:
            # Skip destroyed combatants
            if combatant.resources.hp_current <= 0:
                continue
            blocked_hexes.append(combatant.position.coord)
            blocked_set.add((combatant.position.coord.q, combatant.position.coord.r))

    # Build difficult terrain set
    difficult_hexes: list[HexCoord] = []
    difficult_set: set[tuple[int, int]] = set()
    if scenario.terrain and scenario.terrain.tiles:
        for tile in scenario.terrain.tiles:
            if tile.difficult:
                difficult_hexes.append(tile.coord)
                difficult_set.add((tile.coord.q, tile.coord.r))

    # BFS to find reachable hexes with proper movement costs
    # Difficult terrain costs 2 movement instead of 1
    reachable_hexes: list[HexCoord] = []
    visited: dict[tuple[int, int], int] = {}  # (q, r) -> lowest cost to reach
    queue: list[tuple[HexCoord, int]] = [(origin, 0)]

    # Hex neighbor directions (axial coordinates)
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

    while queue:
        coord, cost = queue.pop(0)
        key = (coord.q, coord.r)

        # Skip if we've found a cheaper path
        if key in visited and visited[key] <= cost:
            continue
        visited[key] = cost

        # Add to reachable (excluding origin)
        if cost > 0 and cost <= speed:
            reachable_hexes.append(coord)

        # Explore neighbors
        for dq, dr in directions:
            neighbor = HexCoord(q=coord.q + dq, r=coord.r + dr)
            neighbor_key = (neighbor.q, neighbor.r)

            # Skip blocked hexes
            if neighbor_key in blocked_set:
                continue

            # Calculate movement cost (difficult terrain = 2)
            is_difficult = neighbor_key in difficult_set
            move_cost = 2 if is_difficult else 1
            new_cost = cost + move_cost

            # Only queue if within speed and cheaper than previous visit
            if new_cost <= speed:
                if neighbor_key not in visited or visited[neighbor_key] > new_cost:
                    queue.append((neighbor, new_cost))

    return MovementRangeData(
        actor_id=actor_id,
        max_range=speed,
        reachable_hexes=reachable_hexes,
        blocked_hexes=blocked_hexes,
        difficult_hexes=difficult_hexes,
    )
