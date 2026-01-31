"""SQLModel table definitions.

Database models use the JSON blob pattern - store full Pydantic model
as JSON in a single column for flexibility, with indexed columns for
common query patterns.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, JSON, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    """Get current UTC timestamp (naive, for PostgreSQL TIMESTAMP WITHOUT TIME ZONE)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    description: str = Field(default="")
    user_id: str = Field(index=True)
    status: str = Field(default="active", index=True)
    visibility: str = Field(default="private")
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    settings: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class CampaignMembershipDB(TimestampMixin, table=True):
    """Campaign membership linking users to campaigns with roles."""

    __tablename__ = "campaign_memberships"
    __table_args__ = (
        UniqueConstraint("campaign_id", "user_id", name="uq_campaign_member"),
    )

    id: str = Field(primary_key=True)
    campaign_id: str = Field(index=True)
    user_id: str = Field(index=True)
    role: str = Field(default="player")  # owner, co_gm, player
    status: str = Field(default="active", index=True)
    ready_state: str = Field(default="not_ready")  # ready, not_ready
    assigned_character_id: str | None = Field(default=None, index=True)


class CampaignInviteDB(TimestampMixin, table=True):
    """Shareable invite links for campaigns."""

    __tablename__ = "campaign_invites"
    __table_args__ = (UniqueConstraint("token", name="uq_campaign_invite_token"),)

    id: str = Field(primary_key=True)
    campaign_id: str = Field(index=True)
    invited_by_user_id: str = Field(index=True)
    role: str = Field(default="player")
    token: str = Field(index=True)
    status: str = Field(default="pending")  # pending, accepted, revoked, expired
    invited_email: str | None = None
    invite_note: str | None = Field(default=None, sa_column=Column(Text))
    expires_at: datetime | None = None
    redeemed_by_user_id: str | None = Field(default=None, index=True)


class CampaignCharacterDB(TimestampMixin, table=True):
    """Associates characters with campaigns (many-to-many)."""

    __tablename__ = "campaign_characters"
    __table_args__ = (
        UniqueConstraint("campaign_id", "character_id", name="uq_campaign_character"),
    )

    id: str = Field(primary_key=True)
    campaign_id: str = Field(index=True)
    character_id: str = Field(index=True)
    added_by_user_id: str = Field(index=True)
    role: str = Field(default="player")  # player, npc
    notes: str = Field(default="")


class CharacterDB(TimestampMixin, table=True):
    """Character storage with full JSON blob (pilot + mechs).

    The `data` column contains the full core.character.Character model
    serialized as JSON. This includes the pilot, all mech configurations,
    and the active mech ID.

    Indexed columns support common queries without parsing JSON.
    """

    __tablename__ = "characters"

    id: str = Field(primary_key=True)
    callsign: str = Field(index=True)  # Pilot callsign for search
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    user_id: str = Field(index=True, default="default_user")
    campaign_id: str | None = Field(default=None, index=True)


CombatSessionStatus = str  # "active", "paused", "completed", "abandoned"


class CombatSessionDB(TimestampMixin, table=True):
    """Combat session storage with full scenario as JSON blob.

    The `scenario` column contains the full core.mech.combat_state.MechCombatScenario
    model serialized as JSON. This includes all combatants, rounds, terrain, and
    deployables.

    Indexed columns support queries for active sessions and session history.
    """

    __tablename__ = "combat_sessions"

    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    status: CombatSessionStatus = Field(default="active", index=True)
    current_round: int = Field(default=1, ge=1)
    current_turn_index: int = Field(default=0, ge=0)

    # Full combat scenario as JSON blob
    scenario: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Ownership and organization
    gm_user_id: str = Field(index=True, default="default_user")
    campaign_id: str | None = Field(default=None, index=True)
    campaign_session_id: str | None = Field(default=None, index=True)

    # Optional metadata
    notes: str = Field(default="")
    mission_id: str | None = Field(default=None, index=True)
    mission_difficulty: int | None = Field(default=None, ge=1, le=3)
