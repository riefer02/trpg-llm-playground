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

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.db.models import CharacterDB, CampaignCharacterDB, utc_now
from app.backend.dependencies import get_current_user
from app.backend.exceptions import NotFoundError, ValidationError
from app.backend.pdf import render_character_sheet_pdf
from app.backend.schemas import (
    DatabaseMetadata,
    ListResponse,
    ValidationIssue,
    ValidationResponse,
)
from app.backend.serializers import serialize_character_response_fields
from app.backend.utils import (
    core_validation_error_to_api,
    validate_core_model,
    validate_core_model_list,
)

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
    PilotLoadout,
)
from core.mech import MechBuild
from core.mech.build_validation import validate_mech_build

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
    pilot_gear: dict[str, Any] | None = Field(
        default=None,
        description="Pilot gear loadout (clothing, armor, weapons, gear)",
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


class PilotGearUpdateRequest(BaseModel):
    """Request body for updating pilot gear loadout."""

    pilot_gear: dict[str, Any] = Field(
        ..., description="PilotLoadout data for the mission"
    )


class MechBuildUpdateRequest(BaseModel):
    """Request body for updating a mech build."""

    build: dict[str, Any] = Field(
        default_factory=dict, description="MechBuild data (weapons, systems)"
    )


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


class CharacterResponse(DatabaseMetadata):
    """Response model for character data.

    Includes database metadata, full character data, and computed fields.
    """

    # Linkage metadata
    campaign_ids: list[str]

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
    pilot_gear: dict[str, Any] | None
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


# Use shared ListResponse[CharacterResponse] for list endpoint
# Use shared ValidationResponse for validation endpoint


# =============================================================================
# Helper Functions
# =============================================================================


def _generate_character_id() -> str:
    """Generate a unique character ID."""
    return f"char_{uuid4().hex[:12]}"


def _generate_mech_id() -> str:
    """Generate a unique mech configuration ID."""
    return f"mech_{uuid4().hex[:12]}"


def _sanitize_filename(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    safe = safe.strip("_")
    return safe or "character"


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
                triggers = validate_core_model_list(
                    PilotTrigger, request.triggers, "trigger"
                )

            talents = None
            if request.talents:
                talents = validate_core_model_list(Talent, request.talents, "talent")

            background = None
            if request.background:
                background = validate_core_model(
                    Background, request.background, "background"
                )

            pilot_gear = None
            if request.pilot_gear:
                pilot_gear = validate_core_model(
                    PilotLoadout, request.pilot_gear, "pilot_gear"
                )

            mech_build = MechBuild(frame_id=request.mech_frame_id)

            return create_ll0_character(
                callsign=request.callsign,
                name=request.name,
                background=background,
                skills=skills,
                triggers=triggers,
                talents=talents,
                pilot_gear=pilot_gear,
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
    if request.pilot_gear:
        pilot_data["pilot_gear"] = request.pilot_gear

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
        pilot_updates["triggers"] = validate_core_model_list(
            PilotTrigger, request.triggers, "trigger"
        )

    if request.talents is not None:
        pilot_updates["talents"] = validate_core_model_list(
            Talent, request.talents, "talent"
        )

    if request.licenses is not None:
        pilot_updates["licenses"] = validate_core_model_list(
            License, request.licenses, "license"
        )

    if request.core_bonuses is not None:
        pilot_updates["core_bonuses"] = validate_core_model_list(
            CoreBonus, request.core_bonuses, "core_bonus"
        )

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


async def _get_character_campaign_ids(
    session: AsyncSession, character_id: str
) -> list[str]:
    result = await session.exec(
        select(CampaignCharacterDB.campaign_id).where(
            CampaignCharacterDB.character_id == character_id
        )
    )
    return [campaign_id for campaign_id in result.all() if campaign_id]


async def _character_to_response(
    session: AsyncSession, char_db: CharacterDB
) -> CharacterResponse:
    """Convert a CharacterDB record to a CharacterResponse.

    Hydrates the core Character model to get computed fields.
    """
    core_char = Character.model_validate(char_db.data)
    campaign_ids = set(await _get_character_campaign_ids(session, char_db.id))
    if char_db.campaign_id:
        campaign_ids.add(char_db.campaign_id)

    return CharacterResponse(
        id=char_db.id,
        user_id=char_db.user_id,
        campaign_id=char_db.campaign_id,
        campaign_ids=sorted(campaign_ids),
        created_at=char_db.created_at,
        updated_at=char_db.updated_at,
        **serialize_character_response_fields(core_char),
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

    return await _character_to_response(session, db_char)


@router.get("", response_model=ListResponse[CharacterResponse])
async def list_characters(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
    campaign_id: str | None = None,
) -> ListResponse[CharacterResponse]:
    """List characters for the current user.

    Optionally filter by campaign_id.
    """
    query = select(CharacterDB).where(CharacterDB.user_id == user["id"])

    linked_ids: list[str] = []
    if campaign_id:
        link_result = await session.exec(
            select(CampaignCharacterDB.character_id).where(
                CampaignCharacterDB.campaign_id == campaign_id
            )
        )
        linked_ids = [char_id for char_id in link_result.all() if char_id]

    result = await session.exec(query)
    characters = result.all()

    if campaign_id:
        characters = [
            c for c in characters if c.campaign_id == campaign_id or c.id in linked_ids
        ]

    items: list[CharacterResponse] = []
    for char_db in characters:
        items.append(await _character_to_response(session, char_db))

    return ListResponse(items=items, total=len(items))


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

    return await _character_to_response(session, character)


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

    return await _character_to_response(session, char_db)


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


@router.get("/{character_id}/validate", response_model=ValidationResponse)
async def validate_character_endpoint(
    character_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> ValidationResponse:
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

    return ValidationResponse(
        valid=validation.valid,
        issues=[
            ValidationIssue(
                code=issue.code,
                message=issue.message,
                severity=issue.severity,
            )
            for issue in validation.issues
        ],
    )


@router.get("/{character_id}/export.pdf")
async def export_character_pdf(
    character_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> Response:
    """Render a character sheet PDF using a server-side template."""
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    char_db = result.first()

    if not char_db:
        raise NotFoundError("Character", character_id)

    core_char = Character.model_validate(char_db.data)
    pdf_bytes = render_character_sheet_pdf(core_char)
    filename = f"{_sanitize_filename(core_char.pilot.callsign)}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# =============================================================================
# Loadout Endpoints
# =============================================================================


@router.put("/{character_id}/pilot-gear", response_model=CharacterResponse)
async def update_pilot_gear(
    character_id: str,
    body: PilotGearUpdateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CharacterResponse:
    """Update a character's pilot gear loadout."""
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    char_db = result.first()

    if not char_db:
        raise NotFoundError("Character", character_id)

    core_char = Character.model_validate(char_db.data)
    pilot_gear = validate_core_model(PilotLoadout, body.pilot_gear, "pilot_gear")

    pilot_data = core_char.pilot.model_dump(mode="json")
    pilot_data["pilot_gear"] = pilot_gear.model_dump(mode="json")
    updated_pilot = validate_core_model(Pilot, pilot_data, "pilot")

    updated_char = Character(
        id=core_char.id,
        pilot=updated_pilot,
        mechs=core_char.mechs,
        active_mech_id=core_char.active_mech_id,
    )

    char_db.data = updated_char.model_dump(mode="json")
    char_db.updated_at = utc_now()

    session.add(char_db)
    await session.commit()
    await session.refresh(char_db)

    return await _character_to_response(session, char_db)


@router.put("/{character_id}/mechs/{mech_id}/build", response_model=CharacterResponse)
async def update_mech_build(
    character_id: str,
    mech_id: str,
    body: MechBuildUpdateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CharacterResponse:
    """Update a mech build (weapons + systems) for a character."""
    result = await session.exec(
        select(CharacterDB).where(
            CharacterDB.id == character_id,
            CharacterDB.user_id == user["id"],
        )
    )
    char_db = result.first()

    if not char_db:
        raise NotFoundError("Character", character_id)

    core_char = Character.model_validate(char_db.data)
    mech = core_char.get_mech(mech_id)
    if not mech:
        raise NotFoundError("Mech", mech_id)

    build_data = {"frame_id": mech.frame_id, **body.build}
    mech_build = validate_core_model(MechBuild, build_data, "mech_build")

    frame = mech.get_frame()
    if frame is None:
        raise ValidationError("Invalid mech frame")

    validation = validate_mech_build(
        frame=frame,
        build=mech_build,
        skills=core_char.pilot.skills,
        grit=core_char.pilot.grit,
        licenses=core_char.pilot.licenses,
        bonus_effects=core_char.core_bonus_effects,
    )

    if not validation.valid:
        raise ValidationError(
            "Invalid mech build",
            errors=[
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                }
                for issue in validation.issues
            ],
        )

    updated_char = core_char.update_mech(mech_id, build=mech_build)

    if updated_char.pilot.level == 0:
        ll0_validation = validate_character(updated_char)
        ll0_errors = [
            issue
            for issue in ll0_validation.issues
            if issue.severity == "error" and issue.code.startswith("ll0_")
        ]
        if ll0_errors:
            raise ValidationError(
                "Invalid LL0 mech build",
                errors=[
                    {
                        "code": issue.code,
                        "message": issue.message,
                        "severity": issue.severity,
                    }
                    for issue in ll0_errors
                ],
            )

    char_db.data = updated_char.model_dump(mode="json")
    char_db.updated_at = utc_now()

    session.add(char_db)
    await session.commit()
    await session.refresh(char_db)

    return await _character_to_response(session, char_db)


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

    return await _character_to_response(session, char_db)


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

    return await _character_to_response(session, char_db)


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

    return await _character_to_response(session, char_db)
