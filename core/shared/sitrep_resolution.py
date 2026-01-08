"""Enhanced SITREP resolution models and functions.

Provides mechanical tracking for SITREP victory conditions including:
- Zone control state tracking
- Victory condition resolution
- Reserve/ingress zone management
- Turn limit enforcement
- Point scoring systems
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING, Any
from pydantic import Field, model_validator
from core.shared.models import FrozenModel
from core.shared.scenario import (
    SitrepZone,
    SitrepType,
    SitrepTemplate,
    VictoryConditionType,
)

if TYPE_CHECKING:
    from core.shared.combat_loop import CombatLoopState


SitrepVictoryOutcome = Literal["players_win", "enemies_win", "draw", "ongoing"]

ZoneControlState = Literal[
    "player_controlled", "enemy_controlled", "contested", "neutral"
]

ReserveSpawnCause = Literal["round_start", "turn_end", "event_triggered", "manual"]


class ZoneControlStateTracker(FrozenModel):
    """Tracks the control state of a zone during active resolution.

    Attributes:
        zone_id: Identifier matching a zone in SitrepTemplate.objective_zones
        state: Current control state (player, enemy, contested, or neutral)
        controlling_side: Which side controls the zone ("players" or "enemies")
        last_checked_turn: The turn number when this state was last evaluated
    """

    zone_id: str
    state: ZoneControlState = Field(default="neutral")
    controlling_side: str | None = Field(default=None)
    last_checked_turn: int = Field(default=1, ge=1)


class SitrepVictoryCondition(FrozenModel):
    """Active tracking of a victory condition during mission resolution.

    Attributes:
        condition_type: Type of victory condition being tracked
        target_value: The target value needed to achieve victory (e.g., 3 zones)
        current_value: The current progress toward the target
        is_met: Whether this condition has been satisfied
        description: Human-readable description of the condition
    """

    condition_type: VictoryConditionType
    target_value: int | None = Field(default=None, ge=1)
    current_value: int = Field(default=0, ge=0)
    is_met: bool = Field(default=False)
    description: str = Field(..., description="Human-readable description")


class SitrepDeployment(FrozenModel):
    """Deployment zone configuration for active mission resolution.

    Attributes:
        player_zones: Where player characters deploy
        enemy_zones: Where enemy forces deploy
        ingress_zones: Where reinforcements can enter
        reserve_pool: NPC IDs held in reserve for delayed deployment
        reserves_spawned_per_round: How many reserves enter per round
        last_ingress_zone: The most recently used ingress zone (for rotation rule)
        reserve_pattern: How reserves are managed (none, half, normal, double, increasing)
    """

    player_zones: list[SitrepZone] = Field(default_factory=list)
    enemy_zones: list[SitrepZone] = Field(default_factory=list)
    ingress_zones: list[SitrepZone] = Field(default_factory=list)
    reserve_pool: list[str] = Field(default_factory=list)
    reserves_spawned_per_round: int | None = Field(default=None, ge=1)
    last_ingress_zone: str | None = Field(default=None)
    reserve_pattern: Literal["none", "half", "normal", "double", "increasing"] = Field(
        default="normal"
    )


class SitrepResolution(FrozenModel):
    """Active SITREP resolution state for tracking mission progress.

    Attributes:
        template_type: The SITREP type (escort, control, extract, etc.)
        current_round: Current round number (1-indexed)
        max_rounds: The round limit from the template
        player_score: Cumulative player score
        enemy_score: Cumulative enemy score
        zone_states: Map of zone_id to control state tracker
        victory_conditions: List of active victory conditions being tracked
        extraction_progress: Progress toward extraction (0.0 to 1.0)
        surviving_players: Number of active player characters
        surviving_enemies: Number of active enemy NPCs
        outcome: Current outcome assessment (None = ongoing)
        turn_limit_reached: Whether the time limit has been reached
        reserves_remaining: Count of reserves left in the pool
    """

    template_type: SitrepType
    current_round: int = Field(default=1, ge=1)
    max_rounds: int = Field(default=6, ge=1)
    player_score: int = Field(default=0, ge=0)
    enemy_score: int = Field(default=0, ge=0)
    zone_states: dict[str, ZoneControlStateTracker] = Field(default_factory=dict)
    victory_conditions: list[SitrepVictoryCondition] = Field(default_factory=list)
    extraction_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    surviving_players: int = Field(default=0, ge=0)
    surviving_enemies: int = Field(default=0, ge=0)
    outcome: SitrepVictoryOutcome | None = Field(default=None)
    turn_limit_reached: bool = Field(default=False)
    reserves_remaining: int = Field(default=0, ge=0)
    deployment: SitrepDeployment | None = Field(default=None)


def create_sitrep_resolution(
    template: SitrepTemplate,
    player_count: int,
    reserve_ids: list[str] | None = None,
    enemy_count: int = 0,
) -> SitrepResolution:
    """Create a SitrepResolution from a SitrepTemplate.

    Initializes all tracking structures for an active SITREP mission.

    Args:
        template: The SITREP template to base resolution on
        player_count: Number of player characters in the mission
        reserve_ids: Optional list of NPC IDs held in reserve
        enemy_count: Total number of enemy NPCs at start

    Returns:
        Initialized SitrepResolution ready for active tracking
    """
    reserve_pool = reserve_ids or []
    reserves_per_round = template.reserve_per_round

    if template.reserve_pattern == "none":
        reserve_pool = []
        reserves_per_round = None
    elif template.reserve_pattern == "half":
        half_pool = len(reserve_pool) // 2
        reserve_pool = reserve_pool[half_pool:]
    elif template.reserve_pattern == "increasing" and reserves_per_round is None:
        reserves_per_round = 1

    zone_states: dict[str, ZoneControlStateTracker] = {}
    for zone in template.objective_zones:
        zone_id = zone.location or f"zone_{len(zone_states)}"
        zone_states[zone_id] = ZoneControlStateTracker(
            zone_id=zone_id,
            state="neutral",
            controlling_side=None,
            last_checked_turn=1,
        )

    victory_conditions: list[SitrepVictoryCondition] = []
    for vc in template.victory_conditions:
        if vc.condition_type == "control_zones" and vc.threshold:
            victory_conditions.append(
                SitrepVictoryCondition(
                    condition_type=vc.condition_type,
                    target_value=vc.threshold,
                    current_value=0,
                    is_met=False,
                    description=vc.description,
                )
            )
        elif vc.condition_type in ("extract_objective", "survive_rounds"):
            victory_conditions.append(
                SitrepVictoryCondition(
                    condition_type=vc.condition_type,
                    target_value=1,
                    current_value=0,
                    is_met=False,
                    description=vc.description,
                )
            )
        else:
            victory_conditions.append(
                SitrepVictoryCondition(
                    condition_type=vc.condition_type,
                    target_value=vc.threshold,
                    current_value=0,
                    is_met=False,
                    description=vc.description,
                )
            )

    deployment = SitrepDeployment(
        player_zones=template.deployment_zones,
        enemy_zones=[z for z in template.ingress_zones if z.zone_type == "ingress"],
        ingress_zones=template.ingress_zones,
        reserve_pool=list(reserve_pool),
        reserves_spawned_per_round=reserves_per_round,
        last_ingress_zone=None,
        reserve_pattern=template.reserve_pattern,
    )

    initial_player_score = 0
    initial_enemy_score = 0

    if template.sitrep_type == "hold_out":
        initial_player_score = 4

    return SitrepResolution(
        template_type=template.sitrep_type,
        current_round=1,
        max_rounds=template.duration_rounds,
        player_score=initial_player_score,
        enemy_score=initial_enemy_score,
        zone_states=zone_states,
        victory_conditions=victory_conditions,
        extraction_progress=0.0,
        surviving_players=player_count,
        surviving_enemies=enemy_count,
        outcome=None,
        turn_limit_reached=False,
        reserves_remaining=len(reserve_pool),
        deployment=deployment,
    )


def spawn_reserves(
    resolution: SitrepResolution,
    count: int | None = None,
    seed: int | None = None,
) -> tuple[SitrepResolution, list[str]]:
    """Spawn reserves from the reserve pool into ingress zones.

    Per PR2 rules: Cannot use the same ingress zone twice in a row.

    Args:
        resolution: Current SitrepResolution state
        count: Number of reserves to spawn (defaults to reserves_spawned_per_round)
        seed: Optional seed for reproducible ordering

    Returns:
        Tuple of (updated resolution, list of spawned NPC IDs)
    """
    if resolution.deployment is None:
        return resolution, []

    pool = resolution.deployment.reserve_pool
    spawn_count = (
        count
        if count is not None
        else (resolution.deployment.reserves_spawned_per_round or 1)
    )

    if not pool or spawn_count <= 0:
        return resolution, []

    actual_spawn = min(spawn_count, len(pool), spawn_count)
    remaining = pool[actual_spawn:]
    spawned = pool[:actual_spawn]

    last_zone = resolution.deployment.last_ingress_zone
    available_zones = [
        z.location
        for z in resolution.deployment.ingress_zones
        if z.location != last_zone
    ] or [z.location for z in resolution.deployment.ingress_zones]

    import random

    if seed is not None:
        random.seed(seed)

    if available_zones:
        selected_zone = random.choice(available_zones)
    else:
        selected_zone = None

    updated_deployment = resolution.deployment.model_copy(
        update={
            "reserve_pool": remaining,
            "last_ingress_zone": selected_zone,
        }
    )

    updated_resolution = resolution.model_copy(
        update={
            "deployment": updated_deployment,
            "reserves_remaining": len(remaining),
            "surviving_enemies": resolution.surviving_enemies + actual_spawn,
        }
    )

    return updated_resolution, spawned


def update_zone_control(
    resolution: SitrepResolution,
    zone_id: str,
    new_state: ZoneControlState,
    controlling_side: str | None,
) -> SitrepResolution:
    """Update the control state of a zone and recalculate scores.

    For CONTROL-type missions, scores are updated based on zone control:
    - Player controlled: +1 player score
    - Enemy controlled: +1 enemy score
    - Contested: No score change

    Args:
        resolution: Current SitrepResolution state
        zone_id: The zone to update
        new_state: The new control state
        controlling_side: Which side controls the zone ("players" or "enemies")

    Returns:
        Updated resolution with zone state and scores updated
    """
    if zone_id not in resolution.zone_states:
        return resolution

    current_zone = resolution.zone_states[zone_id]
    updated_zone = current_zone.model_copy(
        update={
            "state": new_state,
            "controlling_side": controlling_side,
            "last_checked_turn": resolution.current_round,
        }
    )

    player_score_delta = 0
    enemy_score_delta = 0

    if new_state == "player_controlled" and controlling_side == "players":
        if current_zone.state != "player_controlled":
            player_score_delta = 1
    if new_state == "enemy_controlled" and controlling_side == "enemies":
        if current_zone.state != "enemy_controlled":
            enemy_score_delta = 1
    if current_zone.state == "player_controlled" and new_state != "player_controlled":
        player_score_delta = -1
    if current_zone.state == "enemy_controlled" and new_state != "enemy_controlled":
        enemy_score_delta = -1

    new_player_score = resolution.player_score + player_score_delta
    new_enemy_score = resolution.enemy_score + enemy_score_delta

    new_zone_states = {**resolution.zone_states, zone_id: updated_zone}

    return resolution.model_copy(
        update={
            "zone_states": new_zone_states,
            "player_score": new_player_score,
            "enemy_score": new_enemy_score,
        }
    )


def check_extraction_progress(
    resolution: SitrepResolution,
    action_type: Literal["free", "quick", "full"],
) -> SitrepResolution:
    """Track extraction progress for ESCORT/EXTRACT missions.

    Extraction completes when progress reaches 1.0.

    Args:
        resolution: Current SitrepResolution state
        action_type: The type of action used for extraction

    Returns:
        Updated resolution with extraction progress incremented
    """
    if resolution.template_type not in ("escort", "extract"):
        return resolution

    progress_increment = {"free": 0.25, "quick": 0.5, "full": 1.0}
    increment = progress_increment.get(action_type, 0.0)

    new_progress = min(1.0, resolution.extraction_progress + increment)

    new_vc = []
    for vc in resolution.victory_conditions:
        if vc.condition_type == "extract_objective":
            updated_vc = vc.model_copy(
                update={
                    "current_value": int(new_progress * 100),
                    "is_met": new_progress >= 1.0,
                }
            )
            new_vc.append(updated_vc)
        else:
            new_vc.append(vc)

    return resolution.model_copy(
        update={
            "extraction_progress": new_progress,
            "victory_conditions": new_vc,
        }
    )


def check_victory_conditions(
    resolution: SitrepResolution,
    combat_state: Any | None = None,
) -> SitrepResolution:
    """Evaluate all victory conditions against current state.

    Args:
        resolution: Current SitrepResolution state
        combat_state: Optional combat state for NPC tracking

    Returns:
        Updated resolution with is_met flags updated for all conditions
    """
    new_vc = []
    players_win = False
    enemies_win = False

    for vc in resolution.victory_conditions:
        updated = vc

        if vc.condition_type == "control_zones" and vc.target_value:
            player_zones = sum(
                1
                for z in resolution.zone_states.values()
                if z.state == "player_controlled"
            )
            current = min(player_zones, vc.target_value)
            updated = vc.model_copy(
                update={
                    "current_value": current,
                    "is_met": player_zones >= vc.target_value,
                }
            )
            if updated.is_met:
                players_win = True

        elif vc.condition_type == "extract_objective":
            updated = vc.model_copy(
                update={
                    "current_value": int(resolution.extraction_progress * 100),
                    "is_met": resolution.extraction_progress >= 1.0,
                }
            )
            if updated.is_met:
                players_win = True

        elif vc.condition_type == "score_above_threshold" and vc.target_value:
            updated = vc.model_copy(
                update={
                    "current_value": resolution.player_score,
                    "is_met": resolution.player_score >= vc.target_value,
                }
            )
            if updated.is_met:
                players_win = True

        elif vc.condition_type == "outnumber_enemies":
            player_power = resolution.surviving_players
            enemy_power = resolution.surviving_enemies
            updated = vc.model_copy(
                update={
                    "current_value": player_power,
                    "is_met": player_power > enemy_power,
                }
            )
            if updated.is_met:
                players_win = True

        elif vc.condition_type == "survive_rounds":
            updated = vc.model_copy(
                update={
                    "current_value": resolution.current_round,
                    "is_met": resolution.current_round
                    >= (vc.target_value or resolution.max_rounds),
                }
            )
            if updated.is_met:
                players_win = True

        new_vc.append(updated)

    enemies_win = (
        resolution.extraction_progress < 1.0
        and resolution.turn_limit_reached
        and resolution.current_round >= resolution.max_rounds
    )

    if players_win:
        outcome: SitrepVictoryOutcome | None = "players_win"
    elif enemies_win:
        outcome = "enemies_win"
    else:
        outcome = None

    return resolution.model_copy(
        update={
            "victory_conditions": new_vc,
            "outcome": outcome,
        }
    )


def advance_sitrep_round(resolution: SitrepResolution) -> SitrepResolution:
    """Advance the SITREP to the next round.

    Handles round-based events including:
    - Incrementing round counter
    - Checking turn limit
    - Spawning reserves for "increasing" reserve pattern

    Args:
        resolution: Current SitrepResolution state

    Returns:
        Updated resolution with round incremented
    """
    new_round = resolution.current_round + 1
    turn_limit_reached = new_round >= resolution.max_rounds

    updated = resolution.model_copy(
        update={
            "current_round": new_round,
            "turn_limit_reached": turn_limit_reached,
        }
    )

    if (
        resolution.template_type == "escort"
        and resolution.deployment
        and resolution.deployment.reserve_pattern == "increasing"
    ):
        updated, _ = spawn_reserves(updated)

    return updated


def resolve_sitrep(resolution: SitrepResolution) -> SitrepResolution:
    """Final victory resolution at mission end.

    Compares scores, checks turn limits, and determines final outcome.

    Args:
        resolution: Final SitrepResolution state at mission end

    Returns:
        Resolution with outcome set to final result
    """
    if resolution.outcome is not None:
        return resolution

    if resolution.template_type == "control":
        if resolution.player_score > resolution.enemy_score:
            outcome: SitrepVictoryOutcome = "players_win"
        elif resolution.enemy_score > resolution.player_score:
            outcome = "enemies_win"
        else:
            outcome = "draw"

    elif resolution.template_type == "hold_out":
        if resolution.player_score >= 1:
            outcome = "players_win"
        else:
            outcome = "enemies_win"

    elif resolution.template_type in ("escort", "extract"):
        if resolution.extraction_progress >= 1.0:
            outcome = "players_win"
        elif resolution.turn_limit_reached:
            outcome = "enemies_win"
        else:
            outcome = "ongoing"

    elif resolution.template_type == "gauntlet":
        player_power = resolution.surviving_players
        enemy_power = resolution.surviving_enemies
        if player_power > enemy_power:
            outcome = "players_win"
        else:
            outcome = "enemies_win"

    elif resolution.template_type == "recon":
        real_zone = None
        for zone_id, state in resolution.zone_states.items():
            if state.controlling_side == "players":
                if resolution.template_type == "recon":
                    real_zone = zone_id
                    break
        if real_zone is not None:
            zone = resolution.zone_states.get(real_zone)
            if zone is not None and zone.state == "player_controlled":
                outcome = "players_win"
            else:
                outcome = "enemies_win"
        else:
            outcome = "enemies_win"

    else:
        outcome = "draw"

    return resolution.model_copy(update={"outcome": outcome})
