"""Compendium endpoints for reference data.

Read-only endpoints for game reference data (backgrounds, triggers, talents).
This data comes from core definitions and is used by the frontend for
character creation forms, validation, and display.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.backend.schemas import ListResponse

# Import core reference data
from core.pilot.background import PILOT_BACKGROUNDS, Background
from core.pilot.skill import TRIGGER_DEFINITIONS, TriggerDefinition
from core.pilot.talent import EXAMPLE_TALENTS, TalentDefinition

router = APIRouter(prefix="/compendium", tags=["compendium"])


# =============================================================================
# Response Schemas
# =============================================================================


class BackgroundResponse(BaseModel):
    """Background reference data for character creation."""

    id: str
    name: str
    triggers: list[str] = Field(description="Suggested trigger IDs")


class TriggerResponse(BaseModel):
    """Trigger reference data."""

    id: str
    name: str


class TalentResponse(BaseModel):
    """Talent reference data for character creation."""

    id: str
    name: str
    ranks: int = Field(default=3, description="Max ranks (always 3)")


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/backgrounds", response_model=ListResponse[BackgroundResponse])
async def list_backgrounds() -> ListResponse[BackgroundResponse]:
    """List all available pilot backgrounds.

    Each background includes its 4 suggested triggers.
    """
    items = [
        BackgroundResponse(id=bg.id, name=bg.name, triggers=bg.triggers)
        for bg in PILOT_BACKGROUNDS
    ]
    return ListResponse(items=items, total=len(items))


@router.get("/triggers", response_model=ListResponse[TriggerResponse])
async def list_triggers() -> ListResponse[TriggerResponse]:
    """List all available pilot triggers.

    Triggers are used for pilot skill checks. At LL0, pilots have 4 triggers
    at +2 each, typically based on their background's suggestions.
    """
    items = [TriggerResponse(id=t.id, name=t.name) for t in TRIGGER_DEFINITIONS]
    return ListResponse(items=items, total=len(items))


@router.get("/talents", response_model=ListResponse[TalentResponse])
async def list_talents() -> ListResponse[TalentResponse]:
    """List all available pilot talents.

    At LL0, pilots choose 3 talents at rank I.
    """
    items = [TalentResponse(id=t.id, name=t.name, ranks=3) for t in EXAMPLE_TALENTS]
    return ListResponse(items=items, total=len(items))
