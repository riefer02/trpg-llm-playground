"""Campaign management API endpoints.

Provides CRUD operations for campaigns, membership management, invite flows,
character attachments, and lobby readiness tracking.

Design principles:
- Store campaign state as core.shared.campaign.Campaign JSON blobs
- Keep API layer thin: core models handle mechanical validation
- Support future real-auth by keeping user/role metadata explicit
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field, constr
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.db.models import (
    CampaignDB,
    CampaignMembershipDB,
    CampaignInviteDB,
    CampaignCharacterDB,
    CharacterDB,
)
from app.backend.dependencies import get_current_user
from app.backend.exceptions import ConflictError, ForbiddenError, NotFoundError
from app.backend.schemas import DatabaseMetadata, ListResponse
from core.shared.campaign.campaign import Campaign

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


# =============================================================================
# Helpers
# =============================================================================


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _load_campaign_model(record: CampaignDB) -> Campaign:
    if record.data:
        try:
            return Campaign.model_validate(record.data)
        except Exception:  # pragma: no cover - guard against stale blobs
            pass
    return Campaign(
        id=record.id, name=record.name, description=record.description or ""
    )


def _save_campaign_model(record: CampaignDB, core_campaign: Campaign) -> None:
    record.data = core_campaign.model_dump(mode="json")
    record.updated_at = datetime.utcnow()


async def _get_campaign_or_404(session: AsyncSession, campaign_id: str) -> CampaignDB:
    result = await session.exec(select(CampaignDB).where(CampaignDB.id == campaign_id))
    campaign = result.one_or_none()
    if campaign is None:
        raise NotFoundError("Campaign", campaign_id)
    return campaign


async def _get_membership(
    session: AsyncSession,
    campaign_id: str,
    user_id: str,
) -> CampaignMembershipDB | None:
    result = await session.exec(
        select(CampaignMembershipDB)
        .where(CampaignMembershipDB.campaign_id == campaign_id)
        .where(CampaignMembershipDB.user_id == user_id)
    )
    return result.one_or_none()


async def _require_membership(
    session: AsyncSession,
    campaign_id: str,
    user_id: str,
    *,
    allowed_roles: tuple[str, ...] | None = None,
) -> CampaignMembershipDB:
    membership = await _get_membership(session, campaign_id, user_id)
    if membership is None or membership.status != "active":
        raise ForbiddenError("Campaign membership required")
    if allowed_roles and membership.role not in allowed_roles:
        raise ForbiddenError("Insufficient role for this campaign")
    return membership


async def _count_rows(session: AsyncSession, model, **filters) -> int:
    stmt = select(func.count()).select_from(model)
    for column, value in filters.items():
        stmt = stmt.where(getattr(model, column) == value)
    result = await session.exec(stmt)
    return int(result.one())


# =============================================================================
# Request Models
# =============================================================================


class CampaignCreateRequest(BaseModel):
    name: constr(strip_whitespace=True, min_length=1)
    description: str | None = Field(default="", description="Campaign pitch")
    notes: str | None = Field(default="", description="GM notes")


class CampaignInviteCreateRequest(BaseModel):
    role: str = Field(default="player", description="player or co_gm")
    invited_email: str | None = Field(default=None, description="Optional email memo")
    expires_in_hours: int | None = Field(
        default=168, ge=1, le=24 * 30, description="Validity window in hours"
    )


class CampaignCharacterAttachRequest(BaseModel):
    character_id: str = Field(..., description="Character ID to attach")
    role: str = Field(default="player", description="player or npc")
    notes: str | None = Field(default="")


class CampaignMemberSettingsRequest(BaseModel):
    ready: bool | None = Field(default=None, description="Toggle ready state")
    assigned_character_id: str | None = Field(
        default=None, description="Set active character for this member"
    )


# =============================================================================
# Response Models
# =============================================================================


class CampaignMemberResponse(BaseModel):
    id: str
    user_id: str
    role: str
    status: str
    ready_state: str
    assigned_character_id: str | None
    created_at: datetime
    updated_at: datetime


class CampaignInviteResponse(BaseModel):
    id: str
    token: str
    role: str
    status: str
    invited_email: str | None
    expires_at: datetime | None
    invited_by_user_id: str
    created_at: datetime
    updated_at: datetime


class CampaignCharacterResponse(BaseModel):
    id: str
    campaign_id: str
    character_id: str
    callsign: str
    user_id: str
    role: str
    notes: str
    created_at: datetime
    updated_at: datetime


class CampaignSummaryResponse(DatabaseMetadata):
    name: str
    description: str
    status: str
    visibility: str
    membership_role: str
    membership_status: str
    ready_state: str | None
    member_count: int
    character_count: int


class CampaignDetailResponse(DatabaseMetadata):
    name: str
    description: str
    status: str
    visibility: str
    data: dict[str, Any]
    members: list[CampaignMemberResponse]
    invites: list[CampaignInviteResponse]
    characters: list[CampaignCharacterResponse]


# =============================================================================
# Routes
# =============================================================================


@router.get("", response_model=ListResponse[CampaignSummaryResponse])
async def list_campaigns(
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> ListResponse[CampaignSummaryResponse]:
    result = await session.exec(
        select(CampaignDB, CampaignMembershipDB)
        .join(
            CampaignMembershipDB,
            CampaignMembershipDB.campaign_id == CampaignDB.id,
        )
        .where(CampaignMembershipDB.user_id == user["id"])
    )
    rows = result.all()

    summaries: list[CampaignSummaryResponse] = []
    for campaign_db, membership in rows:
        member_count = await _count_rows(
            session, CampaignMembershipDB, campaign_id=campaign_db.id
        )
        character_count = await _count_rows(
            session, CampaignCharacterDB, campaign_id=campaign_db.id
        )
        summaries.append(
            CampaignSummaryResponse(
                id=campaign_db.id,
                user_id=campaign_db.user_id,
                campaign_id=None,
                created_at=campaign_db.created_at,
                updated_at=campaign_db.updated_at,
                name=campaign_db.name,
                description=campaign_db.description,
                status=campaign_db.status,
                visibility=campaign_db.visibility,
                membership_role=membership.role,
                membership_status=membership.status,
                ready_state=membership.ready_state,
                member_count=member_count,
                character_count=character_count,
            )
        )

    return ListResponse(items=summaries, total=len(summaries))


@router.post(
    "", response_model=CampaignDetailResponse, status_code=status.HTTP_201_CREATED
)
async def create_campaign(
    body: CampaignCreateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    campaign_id = _generate_id("camp")
    core_campaign = Campaign(
        id=campaign_id,
        name=body.name,
        description=body.description or "",
        campaign_notes=body.notes or "",
    )

    campaign_db = CampaignDB(
        id=campaign_id,
        name=body.name,
        description=body.description or "",
        user_id=user["id"],
        status="active",
        visibility="private",
        data=core_campaign.model_dump(mode="json"),
        settings={},
    )
    session.add(campaign_db)

    membership = CampaignMembershipDB(
        id=_generate_id("campmem"),
        campaign_id=campaign_id,
        user_id=user["id"],
        role="owner",
        status="active",
        ready_state="not_ready",
    )
    session.add(membership)

    await session.commit()
    await session.refresh(campaign_db)
    await session.refresh(membership)

    return await _build_campaign_detail_response(session, campaign_db, user["id"])


@router.get("/{campaign_id}", response_model=CampaignDetailResponse)
async def get_campaign(
    campaign_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    await _require_membership(session, campaign_id, user["id"])
    return await _build_campaign_detail_response(session, campaign_db, user["id"])


@router.post("/{campaign_id}/invites", response_model=CampaignInviteResponse)
async def create_invite(
    campaign_id: str,
    body: CampaignInviteCreateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignInviteResponse:
    await _get_campaign_or_404(session, campaign_id)
    await _require_membership(
        session,
        campaign_id,
        user["id"],
        allowed_roles=("owner", "co_gm"),
    )

    invite_id = _generate_id("campinv")
    token = uuid4().hex
    expires_at = None
    if body.expires_in_hours:
        expires_at = datetime.utcnow() + timedelta(hours=body.expires_in_hours)

    invite = CampaignInviteDB(
        id=invite_id,
        campaign_id=campaign_id,
        invited_by_user_id=user["id"],
        role=body.role,
        token=token,
        status="pending",
        invited_email=body.invited_email,
        expires_at=expires_at,
    )
    session.add(invite)
    await session.commit()
    await session.refresh(invite)

    return CampaignInviteResponse(**invite.model_dump())


@router.post("/invites/{token}/accept", response_model=CampaignDetailResponse)
async def accept_invite(
    token: str,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    invite_result = await session.exec(
        select(CampaignInviteDB).where(CampaignInviteDB.token == token)
    )
    invite = invite_result.one_or_none()
    if invite is None:
        raise NotFoundError("Invite", token)
    if invite.status != "pending":
        raise ConflictError("Invite already processed")
    if invite.expires_at and invite.expires_at < datetime.utcnow():
        invite.status = "expired"
        await session.commit()
        raise ConflictError("Invite has expired")

    campaign = await _get_campaign_or_404(session, invite.campaign_id)
    membership = await _get_membership(session, invite.campaign_id, user["id"])
    if membership is None:
        membership = CampaignMembershipDB(
            id=_generate_id("campmem"),
            campaign_id=invite.campaign_id,
            user_id=user["id"],
            role=invite.role,
            status="active",
            ready_state="not_ready",
        )
        session.add(membership)
    else:
        membership.status = "active"
        membership.role = invite.role

    invite.status = "accepted"
    invite.redeemed_by_user_id = user["id"]

    await session.commit()

    return await _build_campaign_detail_response(session, campaign, user["id"])


@router.post("/{campaign_id}/characters", response_model=CampaignDetailResponse)
async def attach_character(
    campaign_id: str,
    body: CampaignCharacterAttachRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    membership = await _require_membership(session, campaign_id, user["id"])

    character_result = await session.exec(
        select(CharacterDB).where(CharacterDB.id == body.character_id)
    )
    character = character_result.one_or_none()
    if character is None:
        raise NotFoundError("Character", body.character_id)

    if character.user_id != user["id"] and membership.role not in {"owner", "co_gm"}:
        raise ForbiddenError("Cannot attach another user's character")

    existing = await session.exec(
        select(CampaignCharacterDB)
        .where(CampaignCharacterDB.campaign_id == campaign_id)
        .where(CampaignCharacterDB.character_id == body.character_id)
    )
    if existing.one_or_none() is not None:
        raise ConflictError("Character already attached to this campaign")

    link = CampaignCharacterDB(
        id=_generate_id("campchar"),
        campaign_id=campaign_id,
        character_id=body.character_id,
        added_by_user_id=user["id"],
        role=body.role,
        notes=body.notes or "",
    )
    session.add(link)
    await session.commit()

    return await _build_campaign_detail_response(session, campaign_db, user["id"])


@router.delete(
    "/{campaign_id}/characters/{character_id}",
    response_model=CampaignDetailResponse,
)
async def detach_character(
    campaign_id: str,
    character_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    membership = await _require_membership(session, campaign_id, user["id"])

    link_result = await session.exec(
        select(CampaignCharacterDB)
        .where(CampaignCharacterDB.campaign_id == campaign_id)
        .where(CampaignCharacterDB.character_id == character_id)
    )
    link = link_result.one_or_none()
    if link is None:
        raise NotFoundError("CampaignCharacter", character_id)

    character_result = await session.exec(
        select(CharacterDB).where(CharacterDB.id == character_id)
    )
    character = character_result.one_or_none()

    if (
        character
        and character.user_id != user["id"]
        and membership.role not in {"owner", "co_gm"}
    ):
        raise ForbiddenError("Cannot detach another user's character")

    await session.delete(link)
    await session.commit()

    return await _build_campaign_detail_response(session, campaign_db, user["id"])


@router.post(
    "/{campaign_id}/members/{member_id}/settings",
    response_model=CampaignMemberResponse,
)
async def update_member_settings(
    campaign_id: str,
    member_id: str,
    body: CampaignMemberSettingsRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignMemberResponse:
    membership_result = await session.exec(
        select(CampaignMembershipDB)
        .where(CampaignMembershipDB.id == member_id)
        .where(CampaignMembershipDB.campaign_id == campaign_id)
    )
    membership = membership_result.one_or_none()
    if membership is None:
        raise NotFoundError("CampaignMembership", member_id)

    requester = await _require_membership(session, campaign_id, user["id"])
    if membership.user_id != user["id"] and requester.role not in {"owner", "co_gm"}:
        raise ForbiddenError("Cannot update other members")

    if body.ready is not None:
        membership.ready_state = "ready" if body.ready else "not_ready"
    if body.assigned_character_id is not None:
        membership.assigned_character_id = body.assigned_character_id

    await session.commit()
    await session.refresh(membership)

    return CampaignMemberResponse(**membership.model_dump())


# =============================================================================
# Internal builders
# =============================================================================


async def _build_campaign_detail_response(
    session: AsyncSession,
    campaign_db: CampaignDB,
    requester_id: str,
) -> CampaignDetailResponse:
    members_result = await session.exec(
        select(CampaignMembershipDB)
        .where(CampaignMembershipDB.campaign_id == campaign_db.id)
        .order_by(CampaignMembershipDB.created_at)
    )
    members = members_result.all()

    invites_result = await session.exec(
        select(CampaignInviteDB)
        .where(CampaignInviteDB.campaign_id == campaign_db.id)
        .order_by(CampaignInviteDB.created_at.desc())
    )
    invites = invites_result.all()

    characters_result = await session.exec(
        select(CampaignCharacterDB, CharacterDB)
        .join(CharacterDB, CharacterDB.id == CampaignCharacterDB.character_id)
        .where(CampaignCharacterDB.campaign_id == campaign_db.id)
    )
    character_rows = characters_result.all()

    member_payload = [
        CampaignMemberResponse(**member.model_dump()) for member in members
    ]
    invite_payload = [
        CampaignInviteResponse(**invite.model_dump()) for invite in invites
    ]

    character_payload: list[CampaignCharacterResponse] = []
    for link, character in character_rows:
        character_payload.append(
            CampaignCharacterResponse(
                id=link.id,
                campaign_id=link.campaign_id,
                character_id=link.character_id,
                callsign=character.callsign,
                user_id=character.user_id,
                role=link.role,
                notes=link.notes,
                created_at=link.created_at,
                updated_at=link.updated_at,
            )
        )

    return CampaignDetailResponse(
        id=campaign_db.id,
        user_id=campaign_db.user_id,
        campaign_id=None,
        created_at=campaign_db.created_at,
        updated_at=campaign_db.updated_at,
        name=campaign_db.name,
        description=campaign_db.description,
        status=campaign_db.status,
        visibility=campaign_db.visibility,
        data=_load_campaign_model(campaign_db).model_dump(mode="json"),
        members=member_payload,
        invites=invite_payload,
        characters=character_payload,
    )
