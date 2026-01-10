"""Pilot CRUD endpoints (internal/low-level primitive).

NOTE: For user-facing character management, use /characters endpoints instead.
This module is kept as an internal primitive for direct pilot manipulation.

Design principle: Core is the source of truth. Request bodies accept raw dicts
which are validated by core models via validate_core_model().
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.db.models import PilotDB, utc_now
from app.backend.dependencies import get_current_user
from app.backend.exceptions import NotFoundError
from app.backend.schemas import ListResponse, ValidationIssue, ValidationResponse
from app.backend.utils import validate_core_model

# Import core models - these are the source of truth
from core.pilot import (
    Pilot,
    SkillSet,
    PilotTrigger,
    Talent,
    License,
    CoreBonus,
    Background,
)

router = APIRouter(prefix="/pilots", tags=["pilots"])


# =============================================================================
# Request Schemas - Thin wrappers, core handles validation
# =============================================================================


class PilotCreateRequest(BaseModel):
    """Request body for creating a pilot.

    Nested objects (skills, triggers, etc.) are passed as dicts.
    Core models handle all validation via validate_core_model().
    """

    callsign: str = Field(..., min_length=1, description="Pilot callsign (required)")
    name: str = Field(default="", description="Pilot's real name")
    level: int = Field(default=0, ge=0, le=12, description="License level (0-12)")

    # Nested objects as dicts - core validates
    skills: dict[str, int] | None = Field(
        default=None,
        description="HASE skills as {hull, agility, systems, engineering}",
    )
    triggers: list[dict[str, Any]] | None = Field(
        default=None,
        description="Triggers as [{trigger_id, rank}, ...]",
    )
    talents: list[dict[str, Any]] | None = Field(
        default=None,
        description="Talents as [{talent_id, rank}, ...]",
    )
    licenses: list[dict[str, Any]] | None = Field(
        default=None,
        description="Licenses as [{license_id, rank}, ...]",
    )
    core_bonuses: list[dict[str, Any]] | None = Field(
        default=None,
        description="Core bonuses as [{core_bonus_id}, ...]",
    )
    background: dict[str, Any] | None = Field(
        default=None,
        description="Background as {id, name, triggers: [...]}",
    )
    notes: str = Field(default="")


class PilotUpdateRequest(BaseModel):
    """Request body for updating a pilot.

    All fields optional - only provided fields are updated.
    """

    callsign: str | None = None
    name: str | None = None
    level: int | None = Field(default=None, ge=0, le=12)
    skills: dict[str, int] | None = None
    triggers: list[dict[str, Any]] | None = None
    talents: list[dict[str, Any]] | None = None
    licenses: list[dict[str, Any]] | None = None
    core_bonuses: list[dict[str, Any]] | None = None
    background: dict[str, Any] | None = None
    notes: str | None = None


# =============================================================================
# Response Schemas
# =============================================================================


class PilotResponse(BaseModel):
    """Response model for pilot data.

    Includes database metadata and hydrated Pilot with computed fields.
    """

    # Database metadata
    id: str
    user_id: str
    campaign_id: str | None
    created_at: datetime
    updated_at: datetime

    # Core pilot data (hydrated from JSON)
    callsign: str
    name: str
    level: int
    skills: dict[str, int]
    triggers: list[dict[str, Any]]
    talents: list[dict[str, Any]]
    licenses: list[dict[str, Any]]
    core_bonuses: list[dict[str, Any]]
    background: dict[str, Any] | None
    notes: str

    # Computed fields from core Pilot
    grit: int
    hp: int
    armor: int
    evasion: int
    e_defense: int
    speed: int
    save_target: int
    attack_bonus: int


# Use shared ListResponse[PilotResponse] for list endpoint
# Use shared ValidationResponse for validation endpoint


# =============================================================================
# Helper Functions
# =============================================================================


def _build_core_pilot(
    pilot_id: str,
    request: PilotCreateRequest | PilotUpdateRequest,
    existing_data: dict[str, Any] | None = None,
) -> Pilot:
    """Build a core Pilot model from request data.

    Uses validate_core_model() for nested objects - core handles validation.
    """
    # Start with existing data or empty dict
    data: dict[str, Any] = dict(existing_data) if existing_data else {}

    # Update simple fields
    if request.callsign is not None:
        data["callsign"] = request.callsign
    if request.name is not None:
        data["name"] = request.name
    if request.level is not None:
        data["level"] = request.level
    if request.notes is not None:
        data["notes"] = request.notes

    # Validate nested objects through core models
    if request.skills is not None:
        data["skills"] = validate_core_model(SkillSet, request.skills, "skills")

    if request.triggers is not None:
        data["triggers"] = [
            validate_core_model(PilotTrigger, t, "trigger") for t in request.triggers
        ]

    if request.talents is not None:
        data["talents"] = [
            validate_core_model(Talent, t, "talent") for t in request.talents
        ]

    if request.licenses is not None:
        data["licenses"] = [
            validate_core_model(License, lic, "license") for lic in request.licenses
        ]

    if request.core_bonuses is not None:
        data["core_bonuses"] = [
            validate_core_model(CoreBonus, cb, "core_bonus")
            for cb in request.core_bonuses
        ]

    if request.background is not None:
        data["background"] = validate_core_model(
            Background, request.background, "background"
        )
    elif isinstance(request, PilotCreateRequest) and "background" not in data:
        data["background"] = None

    # Set the pilot ID
    data["id"] = pilot_id

    # Final validation through core Pilot model
    return validate_core_model(Pilot, data, "pilot")


def _pilot_to_response(pilot_db: PilotDB) -> PilotResponse:
    """Convert a PilotDB record to a PilotResponse.

    Hydrates the core Pilot model to get computed fields.
    """
    # Hydrate core Pilot from stored JSON
    core_pilot = Pilot.model_validate(pilot_db.data)

    return PilotResponse(
        id=pilot_db.id,
        user_id=pilot_db.user_id,
        campaign_id=pilot_db.campaign_id,
        created_at=pilot_db.created_at,
        updated_at=pilot_db.updated_at,
        callsign=core_pilot.callsign,
        name=core_pilot.name,
        level=core_pilot.level,
        skills=core_pilot.skills.as_dict(),
        triggers=[t.model_dump() for t in core_pilot.triggers],
        talents=[t.model_dump() for t in core_pilot.talents],
        licenses=[lic.model_dump() for lic in core_pilot.licenses],
        core_bonuses=[cb.model_dump() for cb in core_pilot.core_bonuses],
        background=core_pilot.background.model_dump()
        if core_pilot.background
        else None,
        notes=core_pilot.notes,
        grit=core_pilot.grit,
        hp=core_pilot.hp,
        armor=core_pilot.armor,
        evasion=core_pilot.evasion,
        e_defense=core_pilot.e_defense,
        speed=core_pilot.speed,
        save_target=core_pilot.save_target,
        attack_bonus=core_pilot.attack_bonus,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.post("", response_model=PilotResponse, status_code=status.HTTP_201_CREATED)
async def create_pilot(
    body: PilotCreateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> PilotResponse:
    """Create a new pilot.

    NOTE: For user-facing character creation, use POST /characters instead.
    """
    pilot_id = f"pilot_{uuid4().hex[:12]}"

    # Build and validate core Pilot model
    core_pilot = _build_core_pilot(pilot_id, body)

    # Serialize to JSON for storage
    pilot_data = core_pilot.model_dump(mode="json")

    db_pilot = PilotDB(
        id=pilot_id,
        name=core_pilot.callsign,  # Use callsign as indexed name
        data=pilot_data,
        user_id=user["id"],
    )

    session.add(db_pilot)
    await session.commit()
    await session.refresh(db_pilot)

    return _pilot_to_response(db_pilot)


@router.get("", response_model=ListResponse[PilotResponse])
async def list_pilots(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
    campaign_id: str | None = None,
) -> ListResponse[PilotResponse]:
    """List pilots for the current user.

    NOTE: For user-facing character list, use GET /characters instead.
    """
    query = select(PilotDB).where(PilotDB.user_id == user["id"])

    if campaign_id:
        query = query.where(PilotDB.campaign_id == campaign_id)

    result = await session.exec(query)
    pilots = result.all()

    return ListResponse(
        items=[_pilot_to_response(p) for p in pilots],
        total=len(pilots),
    )


@router.get("/{pilot_id}", response_model=PilotResponse)
async def get_pilot(
    pilot_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> PilotResponse:
    """Get a pilot by ID with hydrated computed fields."""
    result = await session.exec(
        select(PilotDB).where(
            PilotDB.id == pilot_id,
            PilotDB.user_id == user["id"],
        )
    )
    pilot = result.first()

    if not pilot:
        raise NotFoundError("Pilot", pilot_id)

    return _pilot_to_response(pilot)


@router.put("/{pilot_id}", response_model=PilotResponse)
async def update_pilot(
    pilot_id: str,
    body: PilotUpdateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> PilotResponse:
    """Update a pilot with validation."""
    result = await session.exec(
        select(PilotDB).where(
            PilotDB.id == pilot_id,
            PilotDB.user_id == user["id"],
        )
    )
    pilot = result.first()

    if not pilot:
        raise NotFoundError("Pilot", pilot_id)

    # Build updated Pilot with validation
    core_pilot = _build_core_pilot(pilot_id, body, pilot.data)

    # Update database record
    pilot.name = core_pilot.callsign
    pilot.data = core_pilot.model_dump(mode="json")
    pilot.updated_at = utc_now()

    session.add(pilot)
    await session.commit()
    await session.refresh(pilot)

    return _pilot_to_response(pilot)


@router.delete("/{pilot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pilot(
    pilot_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> None:
    """Delete a pilot."""
    result = await session.exec(
        select(PilotDB).where(
            PilotDB.id == pilot_id,
            PilotDB.user_id == user["id"],
        )
    )
    pilot = result.first()

    if not pilot:
        raise NotFoundError("Pilot", pilot_id)

    await session.delete(pilot)
    await session.commit()


@router.get("/{pilot_id}/validate", response_model=ValidationResponse)
async def validate_pilot(
    pilot_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> ValidationResponse:
    """Validate a pilot against progression rules."""
    result = await session.exec(
        select(PilotDB).where(
            PilotDB.id == pilot_id,
            PilotDB.user_id == user["id"],
        )
    )
    pilot = result.first()

    if not pilot:
        raise NotFoundError("Pilot", pilot_id)

    # Hydrate and validate
    core_pilot = Pilot.model_validate(pilot.data)
    validation = core_pilot.validate_progression()

    return ValidationResponse(
        valid=validation.valid,
        issues=[
            ValidationIssue(
                severity=issue.severity,
                field=None,  # Core validation doesn't track field paths
                message=issue.message,
                code=issue.code,
            )
            for issue in validation.issues
        ],
    )
