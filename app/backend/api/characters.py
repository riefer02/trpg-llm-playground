"""Character CRUD endpoints with unified pilot + mech validation.

This module provides the primary user-facing API for character management.
A Character is the unified abstraction combining a Pilot with their Mechs.

Key features:
1. Full validation against core.character.Character model
2. Computed fields (active_mech_stats, core_bonus_effects) included in responses
3. LL0 character factory for easy character creation
4. Mech management endpoints (add/remove mechs)

Design principle: Core is the source of truth. This layer is a thin wrapper
that passes data to core models via model_validate(). Never duplicate
validation logic - let core handle it.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.db.models import CharacterDB, utc_now
from app.backend.dependencies import get_current_user
from app.backend.exceptions import NotFoundError, ValidationError
from app.backend.utils import validate_core_model, core_validation_error_to_api

# Import core models - these are the source of truth
from core.character import (
    Character,
    MechConfiguration,
    validate_character,
    create_ll0_character,
)
from core.pilot import (
    Pilot,
    SkillSet,
    PilotTrigger,
    Talent,
    License,
    CoreBonus,
    Background,
)
from core.mech import MechBuild

router = APIRouter(prefix="/characters", tags=["characters"])


# =============================================================================
# Request Schemas - Thin wrappers, core handles validation
# =============================================================================


class CharacterCreateRequest(BaseModel):
    """Request body for creating a character.

    For LL0 characters (use_ll0_defaults=True), only callsign is required.
    The create_ll0_character factory provides game-accurate defaults.

    For manual creation, provide pilot/mech data as dicts - core validates.
    """

    callsign: str = Field(..., min_length=1, description="Pilot callsign (required)")
    name: str = Field(default="", description="Pilot's real name")

    # Use factory for LL0 defaults (recommended for new characters)
    use_ll0_defaults: bool = Field(
        default=True,
        description="Use create_ll0_character factory for game-accurate LL0 defaults",
    )

    # Optional overrides for LL0 factory (validated by core)
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
    background: dict[str, Any] | None = Field(
        default=None,
        description="Background as {id, name, triggers: [...]}",
    )

    # Mech configuration
    mech_name: str | None = Field(
        default=None, description="Mech name (defaults to callsign)"
    )
    mech_frame_id: str = Field(default="gms_everest", description="Frame ID")

    # For non-LL0 / manual creation
    level: int = Field(default=0, description="License level (0-12)")
    licenses: list[dict[str, Any]] | None = None
    core_bonuses: list[dict[str, Any]] | None = None
    notes: str = Field(default="")


class CharacterUpdateRequest(BaseModel):
    """Request body for updating a character.

    All fields optional - only provided fields are updated.
    Nested objects are dicts validated by core models.
    """

    callsign: str | None = None
    name: str | None = None
    level: int | None = None
    skills: dict[str, int] | None = None
    triggers: list[dict[str, Any]] | None = None
    talents: list[dict[str, Any]] | None = None
    licenses: list[dict[str, Any]] | None = None
    core_bonuses: list[dict[str, Any]] | None = None
    background: dict[str, Any] | None = None
    notes: str | None = None
    active_mech_id: str | None = None


class MechAddRequest(BaseModel):
    """Request body for adding a mech to a character."""

    name: str = Field(..., min_length=1)
    frame_id: str = Field(default="gms_everest")
    build: dict[str, Any] = Field(default_factory=dict, description="MechBuild data")


# =============================================================================
# Response Schemas
# =============================================================================


class MechStatsResponse(BaseModel):
    """Computed mech stats."""

    hp: int
    armor: int
    evasion: int
    e_defense: int
    speed: int
    sensor_range: int
    tech_attack: int
    heat_cap: int
    repair_cap: int
    system_points: int
    save_target: int
    size: str


class MechConfigResponse(BaseModel):
    """Response model for a mech configuration."""

    id: str
    name: str
    frame_id: str
    build: dict[str, Any]


class CharacterResponse(BaseModel):
    """Response model for character data.

    Includes database metadata, full character data, and computed fields.
    """

    # Database metadata
    id: str
    user_id: str
    campaign_id: str | None
    created_at: datetime
    updated_at: datetime

    # Pilot data
    pilot_id: str
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

    # Pilot computed fields
    grit: int
    pilot_hp: int

    # Mech data
    mechs: list[MechConfigResponse]
    active_mech_id: str | None

    # Computed fields from active mech
    active_mech_stats: MechStatsResponse | None
    core_bonus_effects: list[dict[str, Any]]


class CharacterListResponse(BaseModel):
    """Response model for listing characters."""

    items: list[CharacterResponse]
    total: int


class ValidationIssueResponse(BaseModel):
    """A single validation issue."""

    code: str
    message: str
    severity: str = "error"


class CharacterValidationResponse(BaseModel):
    """Validation results for a character."""

    valid: bool
    issues: list[ValidationIssueResponse]


# =============================================================================
# Helper Functions
# =============================================================================


def _generate_character_id() -> str:
    """Generate a unique character ID."""
    return f"char_{uuid4().hex[:12]}"


def _generate_mech_id() -> str:
    """Generate a unique mech configuration ID."""
    return f"mech_{uuid4().hex[:12]}"


def _build_character_from_request(
    character_id: str,
    request: CharacterCreateRequest,
) -> Character:
    """Build a core Character model from request data.

    Uses create_ll0_character factory if use_ll0_defaults is True and level is 0.
    Core models handle all validation - we just pass through.
    """
    if request.use_ll0_defaults and request.level == 0:
        # Use factory for LL0 characters - it handles defaults and validation
        try:
            # Convert optional dict inputs to core models
            skills = None
            if request.skills:
                skills = validate_core_model(SkillSet, request.skills, "skills")

            triggers = None
            if request.triggers:
                triggers = [
                    validate_core_model(PilotTrigger, t, "trigger")
                    for t in request.triggers
                ]

            talents = None
            if request.talents:
                talents = [
                    validate_core_model(Talent, t, "talent") for t in request.talents
                ]

            background = None
            if request.background:
                background = validate_core_model(
                    Background, request.background, "background"
                )

            mech_build = MechBuild(frame_id=request.mech_frame_id)

            return create_ll0_character(
                callsign=request.callsign,
                name=request.name,
                background=background,
                skills=skills,
                triggers=triggers,
                talents=talents,
                mech_name=request.mech_name,
                mech_build=mech_build,
                character_id=character_id,
            )
        except ValueError as e:
            raise ValidationError(str(e))

    # Manual character creation (non-LL0 or use_ll0_defaults=False)
    pilot_id = f"pilot_{uuid4().hex[:12]}"

    # Build pilot data dict, let core validate
    pilot_data: dict[str, Any] = {
        "id": pilot_id,
        "callsign": request.callsign,
        "name": request.name,
        "level": request.level,
        "notes": request.notes,
    }

    if request.skills:
        pilot_data["skills"] = request.skills
    if request.triggers:
        pilot_data["triggers"] = request.triggers
    if request.talents:
        pilot_data["talents"] = request.talents
    if request.licenses:
        pilot_data["licenses"] = request.licenses
    if request.core_bonuses:
        pilot_data["core_bonuses"] = request.core_bonuses
    if request.background:
        pilot_data["background"] = request.background

    pilot = validate_core_model(Pilot, pilot_data, "pilot")

    # Build mech if frame specified
    mechs = []
    mech_build = MechBuild(frame_id=request.mech_frame_id)
    mech = MechConfiguration(
        id=_generate_mech_id(),
        name=request.mech_name or request.callsign,
        frame_id=request.mech_frame_id,
        build=mech_build,
    )
    mechs.append(mech)

    return validate_core_model(
        Character,
        {
            "id": character_id,
            "pilot": pilot,
            "mechs": mechs,
            "active_mech_id": mechs[0].id if mechs else None,
        },
        "character",
    )


def _update_character_from_request(
    character: Character,
    request: CharacterUpdateRequest,
) -> Character:
    """Apply updates from request to an existing character.

    Core models validate all updates via model_validate().
    """
    pilot_updates: dict[str, Any] = {}

    if request.callsign is not None:
        pilot_updates["callsign"] = request.callsign
    if request.name is not None:
        pilot_updates["name"] = request.name
    if request.level is not None:
        pilot_updates["level"] = request.level
    if request.notes is not None:
        pilot_updates["notes"] = request.notes

    # Pass dicts directly - core validates via model_validate()
    if request.skills is not None:
        pilot_updates["skills"] = validate_core_model(
            SkillSet, request.skills, "skills"
        )

    if request.triggers is not None:
        pilot_updates["triggers"] = [
            validate_core_model(PilotTrigger, t, "trigger") for t in request.triggers
        ]

    if request.talents is not None:
        pilot_updates["talents"] = [
            validate_core_model(Talent, t, "talent") for t in request.talents
        ]

    if request.licenses is not None:
        pilot_updates["licenses"] = [
            validate_core_model(License, lic, "license") for lic in request.licenses
        ]

    if request.core_bonuses is not None:
        pilot_updates["core_bonuses"] = [
            validate_core_model(CoreBonus, cb, "core_bonus")
            for cb in request.core_bonuses
        ]

    if request.background is not None:
        pilot_updates["background"] = validate_core_model(
            Background, request.background, "background"
        )

    # Apply pilot updates if any
    if pilot_updates:
        try:
            character = character.update_pilot(**pilot_updates)
        except PydanticValidationError as e:
            raise core_validation_error_to_api(e, "pilot data")

    # Update active mech if specified
    if request.active_mech_id is not None:
        try:
            character = character.set_active_mech(
                request.active_mech_id if request.active_mech_id else None
            )
        except ValueError as e:
            raise ValidationError(str(e))

    return character


def _character_to_response(char_db: CharacterDB) -> CharacterResponse:
    """Convert a CharacterDB record to a CharacterResponse.

    Hydrates the core Character model to get computed fields.
    """
    core_char = Character.model_validate(char_db.data)

    # Build mech responses
    mech_responses = [
        MechConfigResponse(
            id=m.id,
            name=m.name,
            frame_id=m.frame_id,
            build=m.build.model_dump(mode="json"),
        )
        for m in core_char.mechs
    ]

    # Get active mech stats if available
    active_stats = None
    if core_char.active_mech_stats:
        stats = core_char.active_mech_stats
        active_stats = MechStatsResponse(
            hp=stats.hp,
            armor=stats.armor,
            evasion=stats.evasion,
            e_defense=stats.e_defense,
            speed=stats.speed,
            sensor_range=stats.sensor_range,
            tech_attack=stats.tech_attack,
            heat_cap=stats.heat_cap,
            repair_cap=stats.repair_cap,
            system_points=stats.system_points,
            save_target=stats.save_target,
            size=stats.size,
        )

    return CharacterResponse(
        id=char_db.id,
        user_id=char_db.user_id,
        campaign_id=char_db.campaign_id,
        created_at=char_db.created_at,
        updated_at=char_db.updated_at,
        pilot_id=core_char.pilot.id,
        callsign=core_char.pilot.callsign,
        name=core_char.pilot.name,
        level=core_char.pilot.level,
        skills=core_char.pilot.skills.as_dict(),
        triggers=[t.model_dump() for t in core_char.pilot.triggers],
        talents=[t.model_dump() for t in core_char.pilot.talents],
        licenses=[lic.model_dump() for lic in core_char.pilot.licenses],
        core_bonuses=[cb.model_dump() for cb in core_char.pilot.core_bonuses],
        background=core_char.pilot.background.model_dump()
        if core_char.pilot.background
        else None,
        notes=core_char.pilot.notes,
        grit=core_char.pilot.grit,
        pilot_hp=core_char.pilot.hp,
        mechs=mech_responses,
        active_mech_id=core_char.active_mech_id,
        active_mech_stats=active_stats,
        core_bonus_effects=[e.model_dump() for e in core_char.core_bonus_effects],
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.post("", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED)
async def create_character(
    body: CharacterCreateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CharacterResponse:
    """Create a new character with validation.

    For LL0 characters (level 0 with use_ll0_defaults=True), sensible defaults
    are applied including:
    - 2 mech skill points (Hull +2 by default)
    - 4 triggers at +2 each
    - 3 rank I talents
    - GMS Everest frame with empty loadout

    Custom values can override any defaults.
    """
    character_id = _generate_character_id()

    # Build and validate character
    core_char = _build_character_from_request(character_id, body)

    # Serialize to JSON for storage
    char_data = core_char.model_dump(mode="json")

    db_char = CharacterDB(
        id=character_id,
        callsign=core_char.pilot.callsign,
        data=char_data,
        user_id=user["id"],
    )

    session.add(db_char)
    await session.commit()
    await session.refresh(db_char)

    return _character_to_response(db_char)


@router.get("", response_model=CharacterListResponse)
async def list_characters(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
    campaign_id: str | None = None,
) -> CharacterListResponse:
    """List characters for the current user.

    Optionally filter by campaign_id.
    """
    query = select(CharacterDB).where(CharacterDB.user_id == user["id"])

    if campaign_id:
        query = query.where(CharacterDB.campaign_id == campaign_id)

    result = await session.exec(query)
    characters = result.all()

    return CharacterListResponse(
        items=[_character_to_response(c) for c in characters],
        total=len(characters),
    )


@router.get("/{character_id}", response_model=CharacterResponse)
async def get_character(
    character_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CharacterResponse:
    """Get a character by ID with computed fields."""
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    character = result.first()

    if not character:
        raise NotFoundError("Character", character_id)

    return _character_to_response(character)


@router.put("/{character_id}", response_model=CharacterResponse)
async def update_character(
    character_id: str,
    body: CharacterUpdateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CharacterResponse:
    """Update a character with validation.

    Only provided fields are updated. The resulting character is validated
    against the core Character model.
    """
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    char_db = result.first()

    if not char_db:
        raise NotFoundError("Character", character_id)

    # Hydrate existing character
    core_char = Character.model_validate(char_db.data)

    # Apply updates
    updated_char = _update_character_from_request(core_char, body)

    # Update database record
    char_db.callsign = updated_char.pilot.callsign
    char_db.data = updated_char.model_dump(mode="json")
    char_db.updated_at = utc_now()

    session.add(char_db)
    await session.commit()
    await session.refresh(char_db)

    return _character_to_response(char_db)


@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(
    character_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> None:
    """Delete a character."""
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    character = result.first()

    if not character:
        raise NotFoundError("Character", character_id)

    await session.delete(character)
    await session.commit()


@router.get("/{character_id}/validate", response_model=CharacterValidationResponse)
async def validate_character_endpoint(
    character_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CharacterValidationResponse:
    """Validate a character against game rules.

    Returns validation results including any issues with:
    - Pilot progression (skill points, triggers, talents, licenses)
    - Mech builds (SP limits, mount compatibility, license gating)
    - LL0-specific rules (GMS-only gear)
    """
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    char_db = result.first()

    if not char_db:
        raise NotFoundError("Character", character_id)

    # Hydrate and validate
    core_char = Character.model_validate(char_db.data)
    validation = validate_character(core_char)

    return CharacterValidationResponse(
        valid=validation.valid,
        issues=[
            ValidationIssueResponse(
                code=issue.code,
                message=issue.message,
                severity=issue.severity,
            )
            for issue in validation.issues
        ],
    )


# =============================================================================
# Mech Management Endpoints
# =============================================================================


@router.post("/{character_id}/mechs", response_model=CharacterResponse)
async def add_mech(
    character_id: str,
    body: MechAddRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CharacterResponse:
    """Add a new mech configuration to a character."""
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    char_db = result.first()

    if not char_db:
        raise NotFoundError("Character", character_id)

    # Hydrate existing character
    core_char = Character.model_validate(char_db.data)

    # Build new mech - let core validate
    build_data = {"frame_id": body.frame_id, **body.build}
    mech_build = validate_core_model(MechBuild, build_data, "mech_build")

    new_mech = MechConfiguration(
        id=_generate_mech_id(),
        name=body.name,
        frame_id=body.frame_id,
        build=mech_build,
    )

    # Add mech to character
    try:
        updated_char = core_char.add_mech(new_mech)
    except ValueError as e:
        raise ValidationError(str(e))

    # Update database
    char_db.data = updated_char.model_dump(mode="json")
    char_db.updated_at = utc_now()

    session.add(char_db)
    await session.commit()
    await session.refresh(char_db)

    return _character_to_response(char_db)


@router.delete("/{character_id}/mechs/{mech_id}", response_model=CharacterResponse)
async def remove_mech(
    character_id: str,
    mech_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CharacterResponse:
    """Remove a mech configuration from a character."""
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    char_db = result.first()

    if not char_db:
        raise NotFoundError("Character", character_id)

    # Hydrate existing character
    core_char = Character.model_validate(char_db.data)

    # Remove mech
    try:
        updated_char = core_char.remove_mech(mech_id)
    except ValueError as e:
        raise ValidationError(str(e))

    # Update database
    char_db.data = updated_char.model_dump(mode="json")
    char_db.updated_at = utc_now()

    session.add(char_db)
    await session.commit()
    await session.refresh(char_db)

    return _character_to_response(char_db)


@router.put(
    "/{character_id}/mechs/{mech_id}/activate", response_model=CharacterResponse
)
async def set_active_mech(
    character_id: str,
    mech_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CharacterResponse:
    """Set the active mech for a character."""
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    char_db = result.first()

    if not char_db:
        raise NotFoundError("Character", character_id)

    # Hydrate existing character
    core_char = Character.model_validate(char_db.data)

    # Set active mech
    try:
        updated_char = core_char.set_active_mech(mech_id)
    except ValueError as e:
        raise ValidationError(str(e))

    # Update database
    char_db.data = updated_char.model_dump(mode="json")
    char_db.updated_at = utc_now()

    session.add(char_db)
    await session.commit()
    await session.refresh(char_db)

    return _character_to_response(char_db)
