"""SQLModel table definitions.

Database models use the JSON blob pattern - store full Pydantic model
as JSON in a single column for flexibility, with indexed columns for
common query patterns.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class TimestampMixin(SQLModel):
    """Mixin providing created_at and updated_at timestamps."""

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class PilotDB(TimestampMixin, table=True):
    """Pilot storage with full JSON blob.
    
    The `data` column contains the full core.pilot.Pilot model serialized
    as JSON. This allows the frontend to work with the complete pilot data
    while the database handles persistence.
    
    Indexed columns (id, name, user_id, campaign_id) support common queries
    without needing to parse the JSON.
    """

    __tablename__ = "pilots"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    user_id: str = Field(index=True, default="default_user")
    campaign_id: str | None = Field(default=None, index=True)


class MechDB(TimestampMixin, table=True):
    """Mech storage with loadout data.
    
    Similar to PilotDB, stores the full mech configuration as JSON
    with indexed columns for common queries.
    """

    __tablename__ = "mechs"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    frame_id: str = Field(index=True)
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    pilot_id: str | None = Field(default=None, index=True)
    campaign_id: str | None = Field(default=None, index=True)


class CampaignDB(TimestampMixin, table=True):
    """Campaign container for organizing pilots, mechs, and sessions."""

    __tablename__ = "campaigns"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    description: str = ""
    gm_user_id: str = Field(index=True)
    settings: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
