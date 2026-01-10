"""Pilot CRUD endpoints with core model validation.

This module integrates the core.pilot.Pilot model with the API layer:
1. Request bodies are validated against core Pilot schemas
2. Responses include hydrated Pilot data with computed fields
3. Validation errors return structured error responses
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.db.models import PilotDB, utc_now
from app.backend.dependencies import get_current_user
from app.backend.exceptions import NotFoundError, ValidationError

# Import core models for validation
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
# Request Schemas
# =============================================================================


class SkillSetInput(BaseModel):
    """Input schema for pilot skills."""

    hull: int = Field(default=0, ge=0, le=6)
    agility: int = Field(default=0, ge=0, le=6)
    systems: int = Field(default=0, ge=0, le=6)
    engineering: int = Field(default=0, ge=0, le=6)


class TriggerInput(BaseModel):
    """Input schema for pilot triggers."""

    trigger_id: str
    rank: int = Field(default=2, ge=2, le=6)


class TalentInput(BaseModel):
    """Input schema for talents."""

    talent_id: str
    rank: int = Field(default=1, ge=1, le=3)


class LicenseInput(BaseModel):
    """Input schema for licenses."""

    license_id: str
    rank: int = Field(default=1, ge=1, le=3)


class CoreBonusInput(BaseModel):
    """Input schema for core bonuses."""

    core_bonus_id: str


class BackgroundInput(BaseModel):
    """Input schema for background."""

    id: str
    name: str


class PilotCreateRequest(BaseModel):
    """Request body for creating a pilot.

    Creates a pilot with validation against core.pilot.Pilot.
    All progression-related fields default to LL0 starting values.
    """

    callsign: str = Field(..., min_length=1, description="Pilot callsign (required)")
    name: str = Field(default="", description="Pilot's real name")
    level: int = Field(default=0, ge=0, le=12, description="License level (0-12)")
    skills: SkillSetInput = Field(default_factory=SkillSetInput)
    triggers: list[TriggerInput] = Field(default_factory=list)
    talents: list[TalentInput] = Field(default_factory=list)
    licenses: list[LicenseInput] = Field(default_factory=list)
    core_bonuses: list[CoreBonusInput] = Field(default_factory=list)
    background: BackgroundInput | None = None
    notes: str = Field(default="")


class PilotUpdateRequest(BaseModel):
    """Request body for updating a pilot.

    All fields are optional - only provided fields are updated.
    """

    callsign: str | None = None
    name: str | None = None
    level: int | None = Field(default=None, ge=0, le=12)
    skills: SkillSetInput | None = None
    triggers: list[TriggerInput] | None = None
    talents: list[TalentInput] | None = None
    licenses: list[LicenseInput] | None = None
    core_bonuses: list[CoreBonusInput] | None = None
    background: BackgroundInput | None = None
    notes: str | None = None


# =============================================================================
# Response Schemas
# =============================================================================


class PilotResponse(BaseModel):
    """Response model for pilot data.

    Includes both database metadata and the full hydrated Pilot model.
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


class PilotListResponse(BaseModel):
    """Response model for listing pilots."""

    items: list[PilotResponse]
    total: int


# =============================================================================
# Helper Functions
# =============================================================================


def _build_core_pilot(
    pilot_id: str,
    request: PilotCreateRequest | PilotUpdateRequest,
    existing_data: dict[str, Any] | None = None,
) -> Pilot:
    """Build a core Pilot model from request data.

    For updates, merges request data with existing data.
    Raises ValidationError if the resulting Pilot is invalid.
    """
    # Start with existing data or empty dict
    data: dict[str, Any] = dict(existing_data) if existing_data else {}

    # Update with request fields (only if provided)
    if hasattr(request, "callsign") and request.callsign is not None:
        data["callsign"] = request.callsign
    if hasattr(request, "name") and request.name is not None:
        data["name"] = request.name
    if hasattr(request, "level") and request.level is not None:
        data["level"] = request.level
    if hasattr(request, "notes") and request.notes is not None:
        data["notes"] = request.notes

    # Handle nested objects
    if hasattr(request, "skills") and request.skills is not None:
        data["skills"] = SkillSet(
            hull=request.skills.hull,
            agility=request.skills.agility,
            systems=request.skills.systems,
            engineering=request.skills.engineering,
        )

    if hasattr(request, "triggers") and request.triggers is not None:
        data["triggers"] = [
            PilotTrigger(trigger_id=t.trigger_id, rank=t.rank)
            for t in request.triggers
        ]

    if hasattr(request, "talents") and request.talents is not None:
        data["talents"] = [
            Talent(talent_id=t.talent_id, rank=t.rank) for t in request.talents
        ]

    if hasattr(request, "licenses") and request.licenses is not None:
        data["licenses"] = [
            License(license_id=lic.license_id, rank=lic.rank)
            for lic in request.licenses
        ]

    if hasattr(request, "core_bonuses") and request.core_bonuses is not None:
        data["core_bonuses"] = [
            CoreBonus(core_bonus_id=cb.core_bonus_id) for cb in request.core_bonuses
        ]

    if hasattr(request, "background"):
        if request.background is not None:
            data["background"] = Background(
                id=request.background.id, name=request.background.name
            )
        elif isinstance(request, PilotCreateRequest):
            data["background"] = None

    # Set the pilot ID
    data["id"] = pilot_id

    try:
        return Pilot.model_validate(data)
    except PydanticValidationError as e:
        errors = [
            {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
            for err in e.errors()
        ]
        raise ValidationError("Invalid pilot data", errors=errors)


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
        background=core_pilot.background.model_dump() if core_pilot.background else None,
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
    """Create a new pilot with validation against core.pilot.Pilot.

    The pilot data is validated against the core Pilot model, ensuring
    all game rules are enforced (skill limits, trigger constraints, etc.).
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


@router.get("", response_model=PilotListResponse)
async def list_pilots(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
    campaign_id: str | None = None,
) -> PilotListResponse:
    """List pilots for the current user.

    Optionally filter by campaign_id.
    """
    query = select(PilotDB).where(PilotDB.user_id == user["id"])

    if campaign_id:
        query = query.where(PilotDB.campaign_id == campaign_id)

    result = await session.exec(query)
    pilots = result.all()

    return PilotListResponse(
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
    """Update a pilot with validation.

    Only provided fields are updated. The resulting pilot is validated
    against the core Pilot model.
    """
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


@router.get("/{pilot_id}/validate")
async def validate_pilot(
    pilot_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Validate a pilot against progression rules.

    Returns validation results including any issues with skill points,
    talent ranks, license levels, or core bonuses.
    """
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

    return {
        "valid": validation.valid,
        "issues": [issue.model_dump() for issue in validation.issues],
    }
