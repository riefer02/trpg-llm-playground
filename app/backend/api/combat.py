"""Combat session CRUD endpoints with core model validation.

This module provides API endpoints for managing combat sessions.
It directly uses core.mech.combat_state models - no duplicate schemas.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.db.models import CombatSessionDB, utc_now
from app.backend.dependencies import get_current_user
from app.backend.exceptions import ConflictError, NotFoundError, ValidationError
from app.backend.api.campaigns import record_campaign_session_outcome
from app.backend.api.combat_ws import combat_ws_manager

# Import core combat models - use directly, don't duplicate!
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    CombatTurn,
    CombatRound,
)
from core.mech.grid import HexPosition, HexCoord
from core.shared.campaign.campaign import MissionOutcomeReport
from core.mech.action_economy import ActionEconomyState
from core.mech.combat_execution import (
    start_turn,
    end_turn,
    execute_action,
    execute_reaction,
    get_available_actions,
    get_current_actor,
    ActionExecutionInput,
    ReactionInput,
)
from core.shared.enums import ActionType

router = APIRouter(prefix="/combat", tags=["combat"])


# =============================================================================
# Type Aliases (reuse from core where possible)
# =============================================================================

SessionStatus = Literal["active", "paused", "completed", "abandoned"]


# =============================================================================
# Request Schemas - Minimal wrappers, delegate to core
# =============================================================================


class CombatSessionCreateRequest(BaseModel):
    """Request body for creating a combat session.

    Combatants use the exact same structure as core.mech.combat_state.CombatantState.
    """

    name: str = Field(..., min_length=1, description="Session name")
    environment: Literal["standard", "zero_g", "underwater"] = "standard"
    combatants: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of combatants (validated against CombatantState)",
    )
    notes: str = Field(default="")
    campaign_id: str | None = None


class CombatSessionUpdateRequest(BaseModel):
    """Request body for updating session metadata."""

    name: str | None = None
    status: SessionStatus | None = None
    notes: str | None = None


class AddCombatantRequest(BaseModel):
    """Request body for adding a combatant.

    Combatant dict is validated against core.mech.combat_state.CombatantState.
    """

    combatant: dict[str, Any]


class CombatSessionCompleteRequest(BaseModel):
    """Request body for completing a combat session and logging outcomes."""

    outcome: Literal["success", "partial", "failure", "catastrophic"]
    completion_score: float | None = Field(default=None, ge=0.0, le=1.0)
    debrief_notes: str | None = None
    reserves_spent: list[dict] = Field(default_factory=list)
    reserves_earned: list[dict] = Field(default_factory=list)
    rewards: list[str] = Field(default_factory=list)
    notes: str | None = None


# =============================================================================
# Response Schemas
# =============================================================================


class CombatSessionResponse(BaseModel):
    """Response model for a combat session."""

    # Database metadata
    id: str
    gm_user_id: str
    campaign_id: str | None
    created_at: datetime
    updated_at: datetime

    # Session state
    name: str
    status: SessionStatus
    current_round: int
    current_turn_index: int
    notes: str

    # Full scenario - serialized from core model
    scenario: dict[str, Any]


class CombatSessionListItem(BaseModel):
    """Compact response for listing sessions."""

    id: str
    name: str
    status: SessionStatus
    current_round: int
    combatant_count: int
    environment: str
    campaign_id: str | None
    created_at: datetime
    updated_at: datetime


class CombatSessionListResponse(BaseModel):
    """Response model for listing combat sessions."""

    items: list[CombatSessionListItem]
    total: int


# =============================================================================
# Helper Functions - Use core models directly
# =============================================================================


def _validate_combatant(data: dict[str, Any]) -> CombatantState:
    """Validate combatant dict against core CombatantState.

    Core model handles all validation - we just wrap errors.
    """
    try:
        return CombatantState.model_validate(data)
    except PydanticValidationError as e:
        errors = [
            {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
            for err in e.errors()
        ]
        raise ValidationError("Invalid combatant data", errors=errors)


def _validate_scenario(
    combatants: list[dict[str, Any]],
    environment: str,
) -> MechCombatScenario:
    """Build and validate scenario using core models."""
    validated_combatants = [_validate_combatant(c) for c in combatants]

    try:
        return MechCombatScenario(
            combatants=validated_combatants,
            environment=environment,
            rounds=[],
            grapples=[],
            deployables={},
        )
    except PydanticValidationError as e:
        errors = [
            {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
            for err in e.errors()
        ]
        raise ValidationError("Invalid combat scenario", errors=errors)


def _session_to_response(session_db: CombatSessionDB) -> CombatSessionResponse:
    """Convert DB record to response - scenario already validated on write."""
    return CombatSessionResponse(
        id=session_db.id,
        gm_user_id=session_db.gm_user_id,
        campaign_id=session_db.campaign_id,
        created_at=session_db.created_at,
        updated_at=session_db.updated_at,
        name=session_db.name,
        status=session_db.status,
        current_round=session_db.current_round,
        current_turn_index=session_db.current_turn_index,
        notes=session_db.notes,
        scenario=session_db.scenario,
    )


def _session_to_list_item(session_db: CombatSessionDB) -> CombatSessionListItem:
    """Convert DB record to list item."""
    scenario = session_db.scenario
    return CombatSessionListItem(
        id=session_db.id,
        name=session_db.name,
        status=session_db.status,
        current_round=session_db.current_round,
        combatant_count=len(scenario.get("combatants", [])),
        environment=scenario.get("environment", "standard"),
        campaign_id=session_db.campaign_id,
        created_at=session_db.created_at,
        updated_at=session_db.updated_at,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "", response_model=CombatSessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_combat_session(
    body: CombatSessionCreateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Create a new combat session.

    Combatants are validated against core.mech.combat_state.CombatantState.
    """
    session_id = f"combat_{uuid4().hex[:12]}"

    # Core models handle all validation
    scenario = _validate_scenario(body.combatants, body.environment)

    db_session = CombatSessionDB(
        id=session_id,
        name=body.name,
        status="active",
        current_round=1,
        current_turn_index=0,
        scenario=scenario.model_dump(mode="json"),
        gm_user_id=user["id"],
        campaign_id=body.campaign_id,
        notes=body.notes,
    )

    session.add(db_session)
    await session.commit()
    await session.refresh(db_session)

    return _session_to_response(db_session)


@router.get("", response_model=CombatSessionListResponse)
async def list_combat_sessions(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
    status_filter: SessionStatus | None = None,
    campaign_id: str | None = None,
) -> CombatSessionListResponse:
    """List combat sessions for the current user."""
    query = select(CombatSessionDB).where(CombatSessionDB.gm_user_id == user["id"])

    if status_filter:
        query = query.where(CombatSessionDB.status == status_filter)
    if campaign_id:
        query = query.where(CombatSessionDB.campaign_id == campaign_id)

    result = await session.exec(query)
    sessions = result.all()

    return CombatSessionListResponse(
        items=[_session_to_list_item(s) for s in sessions],
        total=len(sessions),
    )


@router.get("/{session_id}", response_model=CombatSessionResponse)
async def get_combat_session(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Get a combat session by ID with full scenario."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    return _session_to_response(combat_session)


@router.put("/{session_id}", response_model=CombatSessionResponse)
async def update_combat_session(
    session_id: str,
    body: CombatSessionUpdateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Update combat session metadata."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if body.name is not None:
        combat_session.name = body.name
    if body.status is not None:
        combat_session.status = body.status
    if body.notes is not None:
        combat_session.notes = body.notes

    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    return _session_to_response(combat_session)


@router.post("/{session_id}/complete", response_model=CombatSessionResponse)
async def complete_combat_session(
    session_id: str,
    body: CombatSessionCompleteRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Mark a combat session complete and propagate mission outcomes."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)
    if combat_session.status == "completed":
        raise ConflictError("Combat session already completed")

    combat_session.status = "completed"
    if body.notes is not None:
        combat_session.notes = body.notes
    combat_session.updated_at = utc_now()

    payload = body.model_dump(exclude={"notes"})
    if payload.get("completion_score") is None:
        payload.pop("completion_score", None)
    mission_outcome = MissionOutcomeReport(**payload)

    if combat_session.campaign_id and combat_session.campaign_session_id:
        await record_campaign_session_outcome(
            session,
            combat_session.campaign_id,
            combat_session.campaign_session_id,
            mission_outcome,
        )

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    return _session_to_response(combat_session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_combat_session(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> None:
    """Delete a combat session."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    await session.delete(combat_session)
    await session.commit()


@router.post("/{session_id}/combatants", response_model=CombatSessionResponse)
async def add_combatant(
    session_id: str,
    body: AddCombatantRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Add a combatant to an existing combat session."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    # Hydrate and validate
    scenario = MechCombatScenario.model_validate(combat_session.scenario)
    new_combatant = _validate_combatant(body.combatant)

    # Check for duplicate ID
    existing_ids = {c.id for c in scenario.combatants}
    if new_combatant.id in existing_ids:
        raise ValidationError(
            f"Combatant with ID '{new_combatant.id}' already exists",
            errors=[
                {
                    "loc": ["combatant", "id"],
                    "msg": "Duplicate combatant ID",
                    "type": "value_error",
                }
            ],
        )

    # Update scenario with new combatant
    updated_scenario = MechCombatScenario(
        combatants=list(scenario.combatants) + [new_combatant],
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    return _session_to_response(combat_session)


@router.delete(
    "/{session_id}/combatants/{combatant_id}", response_model=CombatSessionResponse
)
async def remove_combatant(
    session_id: str,
    combatant_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Remove a combatant from a combat session."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Find and remove combatant
    updated_combatants = [c for c in scenario.combatants if c.id != combatant_id]
    if len(updated_combatants) == len(scenario.combatants):
        raise NotFoundError("Combatant", combatant_id)

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=[
            g
            for g in scenario.grapples
            if g.grappler_id != combatant_id and g.target_id != combatant_id
        ],
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    return _session_to_response(combat_session)


# =============================================================================
# Combat Turn Execution Request/Response Schemas
# =============================================================================


class TurnStartResponse(BaseModel):
    """Response model for starting a combat turn."""

    actor_id: str
    actor_name: str
    economy: dict[str, Any]
    available_actions: list[str]
    prepared_action_expired: bool
    cooldowns_decremented: list[str]
    scenario: dict[str, Any]


class ActionRequest(BaseModel):
    """Request body for executing a combat action."""

    action_id: str = Field(..., description="Action identifier")
    action_type: Literal["full", "quick", "free", "reaction", "protocol", "move"] = Field(
        ..., description="Type of action"
    )
    target_ids: list[str] = Field(default_factory=list, description="Target combatant IDs")
    target_position: dict[str, Any] | None = Field(default=None, description="Target position")
    weapon_id: str | None = Field(default=None, description="Weapon to use")
    system_id: str | None = Field(default=None, description="System to activate")
    movement_path: list[dict[str, Any]] = Field(default_factory=list, description="Movement path")
    is_overcharge: bool = Field(default=False, description="Whether this uses overcharge")


class ActionResponse(BaseModel):
    """Response model for executing a combat action."""

    success: bool
    error: str | None = None
    action_use: dict[str, Any] | None = None
    effects_applied: list[dict[str, Any]]
    damage_dealt: int
    heat_generated: int
    economy: dict[str, Any]
    scenario: dict[str, Any]


class TurnEndResponse(BaseModel):
    """Response model for ending a combat turn."""

    actor_id: str
    next_actor_id: str | None
    next_actor_name: str | None
    round_advanced: bool
    new_round_number: int | None
    end_of_turn_effects: list[dict[str, Any]]
    scenario: dict[str, Any]


class ReactionRequest(BaseModel):
    """Request body for declaring a reaction."""

    reactor_id: str = Field(..., description="ID of the reacting combatant")
    reaction_type: Literal["brace", "overwatch"] = Field(..., description="Type of reaction")
    trigger_action_id: str | None = Field(default=None, description="Action that triggered this")
    target_ids: list[str] = Field(default_factory=list, description="Targets for the reaction")
    weapon_id: str | None = Field(default=None, description="Weapon for overwatch")


class ReactionResponse(BaseModel):
    """Response model for declaring a reaction."""

    success: bool
    error: str | None = None
    reaction_used: str | None = None
    effects_applied: list[dict[str, Any]]
    damage_dealt: int
    scenario: dict[str, Any]


class AvailableActionItem(BaseModel):
    """Single available action."""

    action_id: str
    action_name: str
    action_type: str
    is_available: bool
    unavailable_reason: str | None = None
    requires_target: bool
    requires_weapon: bool
    requires_system: bool = False
    requires_path: bool = False
    max_targets: int = 1


class AvailableActionsResponse(BaseModel):
    """Response model for available actions."""

    actor_id: str
    economy: dict[str, Any]
    full_actions: list[AvailableActionItem]
    quick_actions: list[AvailableActionItem]
    free_actions: list[AvailableActionItem]
    reactions: list[AvailableActionItem]
    protocols: list[AvailableActionItem]
    can_overcharge: bool
    overcharge_level: int = 0


class ReactionTrigger(BaseModel):
    """A pending reaction trigger for a combatant."""

    trigger_type: Literal["attack_incoming", "enemy_movement"]
    triggering_actor_id: str
    triggering_actor_name: str
    triggering_action_id: str | None = None
    available_reactions: list[Literal["brace", "overwatch"]]


class ReactionOpportunityResponse(BaseModel):
    """Response model for checking reaction opportunities."""

    combatant_id: str
    combatant_name: str
    has_reaction_available: bool
    pending_triggers: list[ReactionTrigger]


# =============================================================================
# Session-scoped Economy Tracking
# =============================================================================

# In-memory economy state per session (for simplicity; production would use Redis/DB)
_session_economy_cache: dict[str, ActionEconomyState] = {}


def _get_session_economy(session_id: str) -> ActionEconomyState:
    """Get or create economy state for a session."""
    if session_id not in _session_economy_cache:
        _session_economy_cache[session_id] = ActionEconomyState()
    return _session_economy_cache[session_id]


def _set_session_economy(session_id: str, economy: ActionEconomyState) -> None:
    """Update economy state for a session."""
    _session_economy_cache[session_id] = economy


def _clear_session_economy(session_id: str) -> None:
    """Clear economy state (on turn end/start)."""
    _session_economy_cache.pop(session_id, None)


# =============================================================================
# Combat Turn Execution Endpoints
# =============================================================================


@router.post("/{session_id}/turns/start", response_model=TurnStartResponse)
async def start_combat_turn(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> TurnStartResponse:
    """Initialize the current actor's turn.

    This resets the action economy, expires prepared actions,
    and decrements turn-start cooldowns.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot start turn: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Get current actor from turn order
    current_actor = get_current_actor(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    if current_actor is None:
        raise ValidationError(
            "No current actor found",
            errors=[{"loc": ["turn"], "msg": "Turn order not initialized", "type": "value_error"}],
        )

    # Start the turn using core helper
    updated_scenario, turn_result = start_turn(scenario, current_actor.id)

    # Persist updated scenario
    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Reset economy for this turn
    _set_session_economy(session_id, turn_result.economy)

    # Broadcast state update to all connected WebSocket clients
    response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": response.model_dump(mode="json")},
    )

    return TurnStartResponse(
        actor_id=turn_result.actor_id,
        actor_name=turn_result.actor_name,
        economy=turn_result.economy.model_dump(),
        available_actions=turn_result.available_actions,
        prepared_action_expired=turn_result.prepared_action_expired,
        cooldowns_decremented=turn_result.cooldowns_decremented,
        scenario=combat_session.scenario,
    )


@router.post("/{session_id}/actions", response_model=ActionResponse)
async def execute_combat_action(
    session_id: str,
    body: ActionRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> ActionResponse:
    """Execute a combat action.

    Validates the action against the current economy, resolves effects,
    and records the action in the combat log.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot execute action: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Get current actor
    current_actor = get_current_actor(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    if current_actor is None:
        raise ValidationError(
            "No current actor found",
            errors=[{"loc": ["turn"], "msg": "Turn not started", "type": "value_error"}],
        )

    # Get or create current turn
    round_idx = combat_session.current_round - 1
    turn_idx = combat_session.current_turn_index
    current_turn: CombatTurn

    if round_idx < len(scenario.rounds) and turn_idx < len(scenario.rounds[round_idx].turns):
        current_turn = scenario.rounds[round_idx].turns[turn_idx]
    else:
        current_turn = CombatTurn(actor_id=current_actor.id)

    # Get current economy
    economy = _get_session_economy(session_id)

    # Parse target position if provided
    target_position: HexPosition | None = None
    if body.target_position:
        try:
            target_position = HexPosition.model_validate(body.target_position)
        except Exception:
            pass

    # Parse movement path
    movement_path: list[HexPosition] = []
    for pos_dict in body.movement_path:
        try:
            movement_path.append(HexPosition.model_validate(pos_dict))
        except Exception:
            pass

    # Build action input
    action_input = ActionExecutionInput(
        actor_id=current_actor.id,
        action_id=body.action_id,
        action_type=body.action_type,
        target_ids=body.target_ids,
        target_position=target_position,
        weapon_id=body.weapon_id,
        system_id=body.system_id,
        movement_path=movement_path,
        is_overcharge=body.is_overcharge,
    )

    # Execute action using core helper
    updated_scenario, updated_turn, updated_economy, action_result = execute_action(
        scenario, current_turn, economy, action_input
    )

    if not action_result.success:
        return ActionResponse(
            success=False,
            error=action_result.error,
            effects_applied=[],
            damage_dealt=0,
            heat_generated=0,
            economy=economy.model_dump(),
            scenario=combat_session.scenario,
        )

    # Update turn in round
    updated_rounds = list(updated_scenario.rounds)
    if round_idx < len(updated_rounds):
        round_turns = list(updated_rounds[round_idx].turns)
        if turn_idx < len(round_turns):
            round_turns[turn_idx] = updated_turn
        else:
            round_turns.append(updated_turn)
        updated_rounds[round_idx] = CombatRound(
            round_index=updated_rounds[round_idx].round_index,
            turns=round_turns,
            reaction_counts_by_actor=dict(updated_rounds[round_idx].reaction_counts_by_actor),
        )

    final_scenario = MechCombatScenario(
        combatants=list(updated_scenario.combatants),
        grapples=list(updated_scenario.grapples),
        rounds=updated_rounds,
        terrain=updated_scenario.terrain,
        environment=updated_scenario.environment,
        deployables=dict(updated_scenario.deployables),
    )

    # Persist
    combat_session.scenario = final_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Update economy cache
    _set_session_economy(session_id, updated_economy)

    # Broadcast state update to all connected WebSocket clients
    ws_response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": ws_response.model_dump(mode="json")},
    )

    return ActionResponse(
        success=True,
        action_use=action_result.action_use.model_dump() if action_result.action_use else None,
        effects_applied=action_result.effects_applied,
        damage_dealt=action_result.damage_dealt,
        heat_generated=action_result.heat_generated,
        economy=updated_economy.model_dump(),
        scenario=combat_session.scenario,
    )


@router.post("/{session_id}/turns/end", response_model=TurnEndResponse)
async def end_combat_turn(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> TurnEndResponse:
    """Finalize the current turn and advance to the next actor.

    Applies end-of-turn effects, decrements cooldowns,
    and advances the turn index (or round if needed).
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot end turn: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Get current turn
    round_idx = combat_session.current_round - 1
    turn_idx = combat_session.current_turn_index

    if round_idx >= len(scenario.rounds):
        raise ValidationError(
            "Invalid round",
            errors=[{"loc": ["round"], "msg": "Round not found", "type": "value_error"}],
        )

    current_round_data = scenario.rounds[round_idx]
    if turn_idx >= len(current_round_data.turns):
        raise ValidationError(
            "Invalid turn index",
            errors=[{"loc": ["turn"], "msg": "Turn not found", "type": "value_error"}],
        )

    current_turn = current_round_data.turns[turn_idx]

    # End turn using core helper
    updated_scenario, turn_end_result, new_round, new_turn_idx = end_turn(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
        current_turn,
    )

    # Persist
    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.current_round = new_round
    combat_session.current_turn_index = new_turn_idx
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Clear economy for this session (next turn will reset it)
    _clear_session_economy(session_id)

    # Broadcast state update to all connected WebSocket clients
    ws_response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": ws_response.model_dump(mode="json")},
    )

    return TurnEndResponse(
        actor_id=turn_end_result.actor_id,
        next_actor_id=turn_end_result.next_actor_id,
        next_actor_name=turn_end_result.next_actor_name,
        round_advanced=turn_end_result.round_advanced,
        new_round_number=turn_end_result.new_round_number,
        end_of_turn_effects=turn_end_result.end_of_turn_effects,
        scenario=combat_session.scenario,
    )


@router.post("/{session_id}/reactions", response_model=ReactionResponse)
async def submit_reaction(
    session_id: str,
    body: ReactionRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> ReactionResponse:
    """Declare a reaction during another combatant's turn.

    Brace grants resistance to the triggering attack.
    Overwatch allows a skirmish attack against a moving enemy.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot react: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Build reaction input
    reaction_input = ReactionInput(
        reactor_id=body.reactor_id,
        reaction_type=body.reaction_type,
        trigger_action_id=body.trigger_action_id,
        target_ids=body.target_ids,
        weapon_id=body.weapon_id,
    )

    # Get economy for the reactor (may be different from current actor)
    reactor_economy = ActionEconomyState()  # Reactions have their own per-round tracking

    # Execute reaction using core helper
    updated_scenario, updated_economy, reaction_result = execute_reaction(
        scenario, reactor_economy, reaction_input
    )

    if not reaction_result.success:
        return ReactionResponse(
            success=False,
            error=reaction_result.error,
            effects_applied=[],
            damage_dealt=0,
            scenario=combat_session.scenario,
        )

    # Persist
    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Broadcast state update to all connected WebSocket clients
    ws_response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": ws_response.model_dump(mode="json")},
    )

    return ReactionResponse(
        success=True,
        reaction_used=reaction_result.reaction_used,
        effects_applied=reaction_result.effects_applied,
        damage_dealt=reaction_result.damage_dealt,
        scenario=combat_session.scenario,
    )


@router.get("/{session_id}/available-actions", response_model=AvailableActionsResponse)
async def get_combat_available_actions(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> AvailableActionsResponse:
    """List valid actions for the current actor given economy and status.

    Returns categorized lists of full, quick, free actions, reactions,
    and protocols with availability status.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Get current actor
    current_actor = get_current_actor(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    if current_actor is None:
        raise ValidationError(
            "No current actor found",
            errors=[{"loc": ["turn"], "msg": "Turn not started", "type": "value_error"}],
        )

    # Get current economy
    economy = _get_session_economy(session_id)

    # Get available actions using core helper
    available = get_available_actions(scenario, current_actor.id, economy)

    def to_item(a) -> AvailableActionItem:
        return AvailableActionItem(
            action_id=a.action_id,
            action_name=a.action_name,
            action_type=a.action_type,
            is_available=a.is_available,
            unavailable_reason=a.unavailable_reason,
            requires_target=a.requires_target,
            requires_weapon=a.requires_weapon,
            requires_system=a.requires_system,
            requires_path=a.requires_path,
            max_targets=a.max_targets,
        )

    # Get overcharge level from combatant state
    overcharge_level = 0
    if current_actor.overcharge_state:
        overcharge_level = current_actor.overcharge_state.level

    return AvailableActionsResponse(
        actor_id=available.actor_id,
        economy=available.economy.model_dump(),
        full_actions=[to_item(a) for a in available.full_actions],
        quick_actions=[to_item(a) for a in available.quick_actions],
        free_actions=[to_item(a) for a in available.free_actions],
        reactions=[to_item(a) for a in available.reactions],
        protocols=[to_item(a) for a in available.protocols],
        can_overcharge=available.can_overcharge,
        overcharge_level=overcharge_level,
    )


@router.get(
    "/{session_id}/reaction-opportunities/{combatant_id}",
    response_model=ReactionOpportunityResponse,
)
async def check_reaction_opportunity(
    session_id: str,
    combatant_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> ReactionOpportunityResponse:
    """Check if a combatant has pending reaction triggers.

    This endpoint is polled by clients during opponent turns to detect
    when reaction opportunities arise.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Find the combatant
    combatant = next(
        (c for c in scenario.combatants if c.id == combatant_id),
        None,
    )
    if combatant is None:
        raise NotFoundError("Combatant", combatant_id)

    # Get current actor (whose turn it is)
    current_actor = get_current_actor(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    # Check if combatant has reaction available (not used this round)
    round_idx = combat_session.current_round - 1
    reaction_counts = {}
    if round_idx < len(scenario.rounds):
        reaction_counts = scenario.rounds[round_idx].reaction_counts_by_actor or {}

    has_reaction = reaction_counts.get(combatant_id, 0) == 0

    # Build pending triggers based on current turn state
    pending_triggers: list[ReactionTrigger] = []

    if has_reaction and current_actor and current_actor.id != combatant_id:
        # Check the current turn for triggering actions
        if round_idx < len(scenario.rounds):
            current_round = scenario.rounds[round_idx]
            turn_idx = combat_session.current_turn_index
            if turn_idx < len(current_round.turns):
                current_turn = current_round.turns[turn_idx]

                # Check each action in the current turn for triggers
                for action in current_turn.actions or []:
                    # Attack targeting this combatant
                    target_ids = action.target_ids or []
                    if action.target_id:
                        target_ids = [action.target_id] + list(target_ids)

                    if combatant_id in target_ids:
                        # This is an attack targeting us - brace opportunity
                        pending_triggers.append(
                            ReactionTrigger(
                                trigger_type="attack_incoming",
                                triggering_actor_id=current_actor.id,
                                triggering_actor_name=current_actor.name,
                                triggering_action_id=action.action_id,
                                available_reactions=["brace"],
                            )
                        )

                # Check for movement that passed near us - overwatch opportunity
                if current_turn.movement_path:
                    # Simplified check: if enemy moved and we have ranged weapons
                    pending_triggers.append(
                        ReactionTrigger(
                            trigger_type="enemy_movement",
                            triggering_actor_id=current_actor.id,
                            triggering_actor_name=current_actor.name,
                            triggering_action_id=None,
                            available_reactions=["overwatch"],
                        )
                    )

    return ReactionOpportunityResponse(
        combatant_id=combatant_id,
        combatant_name=combatant.name,
        has_reaction_available=has_reaction,
        pending_triggers=pending_triggers,
    )


# =============================================================================
# WebSocket Endpoint for Real-time Updates
# =============================================================================


@router.websocket("/{session_id}/ws")
async def combat_websocket(
    websocket: WebSocket,
    session_id: str,
    session: AsyncSession = Depends(get_session),
) -> None:
    """WebSocket endpoint for real-time combat state updates.

    Connects to a combat session and receives state updates whenever
    any player executes an action. Also supports ping/pong for keep-alive.

    Message protocol:
    - Server -> Client: {"type": "state", "data": CombatSessionResponse}
    - Server -> Client: {"type": "pong"}
    - Client -> Server: {"type": "ping"}
    """
    # Verify session exists
    result = await session.exec(
        select(CombatSessionDB).where(CombatSessionDB.id == session_id)
    )
    combat_session = result.first()

    if not combat_session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await combat_ws_manager.connect(session_id, websocket)
    try:
        # Send initial state
        response = _session_to_response(combat_session)
        await websocket.send_json({
            "type": "state",
            "data": response.model_dump(mode="json"),
        })

        # Keep connection alive and handle client messages
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        combat_ws_manager.disconnect(session_id, websocket)
    except Exception:
        # Handle any other exceptions by cleaning up
        combat_ws_manager.disconnect(session_id, websocket)