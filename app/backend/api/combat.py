"""Combat session CRUD endpoints with core model validation.

This module provides API endpoints for managing combat sessions.
It directly uses core.mech.combat_state models - no duplicate schemas.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.db.models import CombatSessionDB, utc_now
from app.backend.dependencies import get_current_user
from app.backend.exceptions import ConflictError, NotFoundError, ValidationError
from app.backend.api.campaigns import record_campaign_session_outcome

# Import core combat models - use directly, don't duplicate!
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
)
from core.mech.grid import HexPosition, HexCoord
from core.shared.campaign.campaign import MissionOutcomeReport

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
