"""Pilot CRUD endpoints.

This module demonstrates the pattern for resource endpoints:
1. Use core.* models for validation
2. Store as JSON blob in database
3. Support standard CRUD operations
"""

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.db.models import PilotDB
from app.backend.dependencies import get_current_user
from app.backend.exceptions import NotFoundError

# Import core Pilot model for validation
# Note: Importing here to keep module loosely coupled
# In production, consider a service layer

router = APIRouter(prefix="/pilots", tags=["pilots"])


class PilotCreate(BaseModel):
    """Request body for creating a pilot."""

    name: str
    callsign: str | None = None
    data: dict[str, Any] = {}


class PilotUpdate(BaseModel):
    """Request body for updating a pilot."""

    name: str | None = None
    callsign: str | None = None
    data: dict[str, Any] | None = None


class PilotResponse(BaseModel):
    """Response model for pilot data."""

    id: str
    name: str
    data: dict[str, Any]
    user_id: str
    campaign_id: str | None


class PilotListResponse(BaseModel):
    """Response model for listing pilots."""

    items: list[PilotResponse]
    total: int


@router.post("", response_model=PilotResponse, status_code=status.HTTP_201_CREATED)
async def create_pilot(
    body: PilotCreate,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> PilotResponse:
    """Create a new pilot.
    
    The pilot data is validated and stored as JSON.
    In production, this would validate against core.pilot.Pilot.
    """
    pilot_id = f"pilot_{uuid4().hex[:12]}"
    
    # Merge provided data with defaults
    pilot_data = {
        "name": body.name,
        "callsign": body.callsign or body.name.upper()[:8],
        **body.data,
    }
    
    db_pilot = PilotDB(
        id=pilot_id,
        name=body.name,
        data=pilot_data,
        user_id=user["id"],
    )
    
    session.add(db_pilot)
    await session.commit()
    await session.refresh(db_pilot)
    
    return PilotResponse(
        id=db_pilot.id,
        name=db_pilot.name,
        data=db_pilot.data,
        user_id=db_pilot.user_id,
        campaign_id=db_pilot.campaign_id,
    )


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
        items=[
            PilotResponse(
                id=p.id,
                name=p.name,
                data=p.data,
                user_id=p.user_id,
                campaign_id=p.campaign_id,
            )
            for p in pilots
        ],
        total=len(pilots),
    )


@router.get("/{pilot_id}", response_model=PilotResponse)
async def get_pilot(
    pilot_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> PilotResponse:
    """Get a pilot by ID."""
    result = await session.exec(
        select(PilotDB).where(
            PilotDB.id == pilot_id,
            PilotDB.user_id == user["id"],
        )
    )
    pilot = result.first()
    
    if not pilot:
        raise NotFoundError("Pilot", pilot_id)
    
    return PilotResponse(
        id=pilot.id,
        name=pilot.name,
        data=pilot.data,
        user_id=pilot.user_id,
        campaign_id=pilot.campaign_id,
    )


@router.put("/{pilot_id}", response_model=PilotResponse)
async def update_pilot(
    pilot_id: str,
    body: PilotUpdate,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> PilotResponse:
    """Update a pilot.
    
    Only provided fields are updated.
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
    
    # Create a copy of the data to trigger SQLAlchemy change detection
    updated_data = dict(pilot.data)
    
    if body.name is not None:
        pilot.name = body.name
        updated_data["name"] = body.name
    
    if body.callsign is not None:
        updated_data["callsign"] = body.callsign
    
    if body.data is not None:
        updated_data.update(body.data)
    
    # Reassign to trigger change detection for JSON column
    pilot.data = updated_data
    
    session.add(pilot)
    await session.commit()
    await session.refresh(pilot)
    
    return PilotResponse(
        id=pilot.id,
        name=pilot.name,
        data=pilot.data,
        user_id=pilot.user_id,
        campaign_id=pilot.campaign_id,
    )


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
