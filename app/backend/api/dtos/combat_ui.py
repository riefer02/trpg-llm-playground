"""UI-optimized DTO models for combat state.

These models provide pre-computed, flattened views of MechCombatScenario
for efficient frontend rendering. The frontend becomes a thin rendering
layer while the backend owns all state and computation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.mech.grid import HexCoord


# Type aliases
CombatSide = Literal["players", "hostiles", "neutral"]
SessionStatus = Literal["active", "paused", "completed", "abandoned"]
DecisionType = Literal["hull_save", "engineering_save", "engineering_check", "system_trauma"]
DeployableKind = Literal["drone", "mine", "deployable", "other"]


class CombatantBrief(BaseModel):
    """Minimal combatant data for UI rendering.

    Contains only the fields needed for:
    - Token rendering (position, side, frame)
    - Health bars (hp, heat, structure)
    - Status indicators (statuses, destroyed)
    - AI indicator
    """

    id: str
    name: str
    side: CombatSide
    frame_id: str | None = Field(
        default=None,
        description="Frame identifier for sprite lookup (e.g., 'gms_everest')",
    )
    position: HexCoord | None = Field(default=None, description="Current hex position")
    hp_current: int = Field(..., ge=0)
    hp_max: int = Field(..., ge=1)
    heat_current: int = Field(default=0, ge=0)
    heat_cap: int = Field(default=0, ge=0)
    structure_current: int = Field(default=0, ge=0)
    stress_current: int = Field(default=0, ge=0)
    statuses: list[str] = Field(default_factory=list)
    is_destroyed: bool = False
    is_ai_controlled: bool = False
    speed: int = Field(default=0, ge=0, description="Movement speed for range preview")
    evasion: int = Field(default=0, ge=0)
    e_defense: int = Field(default=0, ge=0)
    armor: int = Field(default=0, ge=0)


class ActionEconomyBrief(BaseModel):
    """Pre-computed action economy state for the current actor."""

    full_actions_remaining: int = Field(default=1, ge=0, le=1)
    quick_actions_remaining: int = Field(default=2, ge=0)
    can_overcharge: bool = True
    reactions_remaining: int = Field(default=1, ge=0)
    overcharge_used: bool = False
    move_used: bool = False


class CurrentActorState(BaseModel):
    """Pre-computed current actor info.

    Eliminates the need for frontend to traverse rounds/turns/combatants
    to find the current actor.
    """

    actor_id: str
    actor_name: str
    frame_id: str | None = None
    side: CombatSide
    is_player_controlled: bool
    economy: ActionEconomyBrief


class ActionFeedEntry(BaseModel):
    """Pre-flattened action for display in the action feed.

    Replaces the frontend's triple-nested loop over rounds/turns/actions.
    """

    id: str = Field(..., description="Unique ID in 'round-turn-action' format")
    round_number: int = Field(..., ge=1)
    actor_id: str
    actor_name: str
    actor_side: CombatSide
    action_id: str
    action_name: str
    target_names: list[str] = Field(default_factory=list)
    damage_dealt: int | None = None
    heat_dealt: int | None = None
    statuses_applied: list[str] = Field(default_factory=list)
    timestamp: int = Field(..., description="Ordering index (round * 1000 + turn * 10 + action)")


class MovementRangeData(BaseModel):
    """Pre-computed movement data for range preview.

    Eliminates the expensive hex generation and pathfinding on the frontend.
    """

    actor_id: str
    max_range: int = Field(..., ge=0, description="From speed stat")
    reachable_hexes: list[HexCoord] = Field(
        default_factory=list,
        description="Valid movement destinations",
    )
    blocked_hexes: list[HexCoord] = Field(
        default_factory=list,
        description="Hexes blocked by other combatants",
    )
    difficult_hexes: list[HexCoord] = Field(
        default_factory=list,
        description="Difficult terrain hexes",
    )


class PendingDecisionBrief(BaseModel):
    """Minimal decision info for UI prompts."""

    decision_id: str
    decision_type: DecisionType
    combatant_id: str
    combatant_name: str
    trigger_source: str
    save_target: int | None = None
    eligible_mounts: list[int] = Field(default_factory=list)
    eligible_systems: list[str] = Field(default_factory=list)


class DeployableBrief(BaseModel):
    """Minimal deployable info for rendering."""

    id: str
    name: str
    kind: DeployableKind
    owner_id: str | None = None
    position: HexCoord
    hp: int
    max_hp: int
    is_armed: bool = False
    is_destroyed: bool = False


class ObjectiveBrief(BaseModel):
    """Minimal objective info for UI display."""

    objective_id: str
    name: str
    description: str
    status: Literal["active", "completed", "failed"]
    is_optional: bool = False
    is_primary: bool = False


class CombatUIState(BaseModel):
    """Top-level UI DTO - replaces raw scenario in responses.

    This model provides everything the frontend needs to render combat
    without additional computation:

    - Pre-computed lookups (combatant_names, combatant_sides)
    - Current turn info (current_actor, is_player_turn)
    - Flattened action history (recent_actions)
    - Movement preview data (movement_range)

    The frontend becomes a thin rendering layer that just maps this
    data to React components.
    """

    # Session identity
    session_id: str
    current_round: int = Field(..., ge=1)
    current_turn_index: int = Field(..., ge=0)
    status: SessionStatus

    # Pre-computed lookups (frontend no longer builds these Maps)
    combatant_names: dict[str, str] = Field(
        default_factory=dict,
        description="Map of combatant ID to display name",
    )
    combatant_sides: dict[str, CombatSide] = Field(
        default_factory=dict,
        description="Map of combatant ID to side",
    )

    # Current turn info
    current_actor: CurrentActorState | None = None
    is_player_turn: bool = False
    pending_decisions: list[PendingDecisionBrief] = Field(default_factory=list)

    # Board state (minimal data for rendering)
    combatants: list[CombatantBrief] = Field(default_factory=list)
    terrain_hash: str | None = Field(
        default=None,
        description="Hash for terrain cache invalidation",
    )
    deployables: list[DeployableBrief] = Field(default_factory=list)

    # Objectives
    objectives: list[ObjectiveBrief] = Field(default_factory=list)

    # Action history (pre-flattened, most recent first)
    recent_actions: list[ActionFeedEntry] = Field(
        default_factory=list,
        description="Last 50 actions, most recent first",
    )
    total_action_count: int = Field(
        default=0,
        description="Total actions for 'X more actions' display",
    )

    # Movement data (populated when in path selection mode)
    movement_range: MovementRangeData | None = None

    # Turn order preview
    turn_order: list[str] = Field(
        default_factory=list,
        description="Ordered list of combatant IDs for initiative display",
    )

    # Mission context
    mission_name: str | None = None
    tile_set: str | None = Field(
        default=None,
        description="Terrain tileset for visual rendering",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "combat_123",
                "current_round": 1,
                "current_turn_index": 0,
                "status": "active",
                "combatant_names": {"c1": "VANGUARD's Everest", "c2": "GMS Grunt"},
                "combatant_sides": {"c1": "players", "c2": "hostiles"},
                "is_player_turn": True,
            }
        }
    }
