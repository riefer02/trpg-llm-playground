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
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Response, status
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
    CombatSessionDB,
)
from app.backend.dependencies import get_current_user
from app.backend.exceptions import ConflictError, ForbiddenError, NotFoundError
from app.backend.pdf.campaign_brief import render_campaign_brief_pdf
from app.backend.schemas import DatabaseMetadata, ListResponse
from core.character import Character
from core.mech.combat_state import (
    CombatantState,
    CombatResources,
    CombatStats,
    MechCombatScenario,
)
from core.shared.campaign.campaign import (
    Campaign,
    CampaignIdentity,
    CampaignLobbyState,
    CampaignMissionRecord,
    MissionObjectiveBrief,
    MissionPrepPlan,
    MissionOutcomeReport,
    MissionStakesBrief,
    ReservePlanEntry,
    Session,
    SessionLifecycleCheckpoint,
)
from core.shared.campaign.serialization import get_campaign_summary

router = APIRouter(prefix="/campaigns", tags=["campaigns"])

DEFAULT_MIN_PILOTS = 3
DEFAULT_MAX_PILOTS = 5

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


async def _get_invite_by_id(
    session: AsyncSession,
    campaign_id: str,
    invite_id: str,
) -> CampaignInviteDB:
    result = await session.exec(
        select(CampaignInviteDB)
        .where(CampaignInviteDB.campaign_id == campaign_id)
        .where(CampaignInviteDB.id == invite_id)
    )
    invite = result.one_or_none()
    if invite is None:
        raise NotFoundError("Campaign invite", invite_id)
    return invite


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


def _get_seat_limits(core_campaign: Campaign) -> tuple[int, int]:
    lobby = core_campaign.lobby_state
    min_required = lobby.min_pilot_count if lobby else DEFAULT_MIN_PILOTS
    preferred = lobby.preferred_pilot_count if lobby else DEFAULT_MAX_PILOTS
    return min_required, preferred


def _build_readiness_summary(
    core_campaign: Campaign, members: list[CampaignMembershipDB]
) -> dict[str, Any]:
    lobby = core_campaign.lobby_state
    ready_members = [m for m in members if m.ready_state == "ready"]
    ready_players = [m for m in ready_members if m.role == "player"]
    members_by_id = {m.id: m for m in members}
    issues: list[str] = []
    member_issues: dict[str, list[str]] = {m.id: [] for m in members}

    if lobby:
        assigned_ready_players = [
            m for m in ready_players if m.id in lobby.assigned_member_ids
        ]
        min_required, preferred = _get_seat_limits(core_campaign)
        lobby_status = lobby.status
        for member_id in lobby.assigned_member_ids:
            member = members_by_id.get(member_id)
            if member is None:
                issues.append(f"Member {member_id} no longer in campaign")
                continue
            if member.ready_state != "ready":
                issues.append(f"{member.user_id} is not ready")
                member_issues[member.id].append("Not ready")
            if member.assigned_character_id is None:
                issues.append(f"{member.user_id} missing character assignment")
                member_issues[member.id].append("Character missing")
        if len(assigned_ready_players) < min_required:
            issues.append(
                f"Need {min_required} ready pilots, currently {len(assigned_ready_players)}"
            )
    else:
        assigned_ready_players = ready_players
        min_required, preferred = DEFAULT_MIN_PILOTS, DEFAULT_MAX_PILOTS
        lobby_status = None
        issues.append("No mission lobby configured")

    can_launch = (
        lobby_status is not None and len(assigned_ready_players) >= min_required
    )
    if lobby_status == "launched":
        issues.append("Mission already launched")
    return {
        "ready_members": len(ready_members),
        "ready_players": len(ready_players),
        "assigned_ready_players": len(assigned_ready_players),
        "total_members": len(members),
        "min_pilots": min_required,
        "preferred_pilots": preferred,
        "can_launch": can_launch,
        "lobby_status": lobby_status,
        "issues": issues,
        "member_issues": member_issues,
    }


def _build_mission_summary(core_campaign: Campaign) -> CampaignOutcomeSummary:
    summary = get_campaign_summary(core_campaign)
    return CampaignOutcomeSummary(
        total_missions=summary["total_missions"],
        successful_missions=summary["successful_missions"],
        partial_missions=summary["partial_missions"],
        failed_missions=summary["failed_missions"],
        average_completion=summary["average_completion"],
        last_outcome=summary.get("last_outcome"),
        last_mission_name=summary.get("last_mission_name"),
        last_mission_date=summary.get("last_mission_date"),
    )


def _compute_seat_warning(
    core_campaign: Campaign,
    members: list[CampaignMembershipDB],
    invites: list[CampaignInviteDB],
) -> str | None:
    preferred = _get_seat_limits(core_campaign)[1]
    active_players = len(
        [m for m in members if m.role == "player" and m.status == "active"]
    )
    pending_player_invites = len(
        [i for i in invites if i.role == "player" and i.status == "pending"]
    )
    if active_players + pending_player_invites > preferred:
        return (
            "GM + 3-5 pilots recommended (PR2 ~12052-12097); "
            "current player seats exceed preferred cap"
        )
    return None


def _character_to_combatant(character_db: CharacterDB) -> CombatantState:
    character = Character.model_validate(character_db.data)
    stats = character.active_mech_stats
    if stats is None:
        raise ConflictError(
            "Character requires an active mech before launching a mission"
        )
    return CombatantState(
        id=f"combat_{character_db.id}",
        name=character_db.callsign,
        side="players",
        kind="mech",
        stats=CombatStats(
            size=stats.size,
            hp_max=stats.hp,
            evasion=stats.evasion,
            e_defense=stats.e_defense,
            armor=stats.armor,
            speed=stats.speed,
            sensor_range=stats.sensor_range,
            tech_attack=stats.tech_attack,
        ),
        resources=CombatResources(
            hp_current=stats.hp,
            heat_current=0,
            heat_cap=stats.heat_cap,
            structure_current=stats.structure,
            stress_current=stats.structure,
            repairs_remaining=stats.repair_cap,
        ),
    )


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


class CampaignIdentityUpdateRequest(BaseModel):
    squad_name: str | None = Field(default=None)
    patron: str | None = Field(default=None)
    who_we_are: str | None = Field(default=None)
    relationships: list[str] | None = Field(default=None)
    themes: list[str] | None = Field(default=None)
    gm_prompts: list[str] | None = Field(default=None)


class CampaignLobbyObjectiveInput(BaseModel):
    id: str
    title: str
    success_condition: str
    priority: Literal["primary", "secondary", "optional"] = "primary"
    related_objective_id: str | None = None


class CampaignLobbyStakesInput(BaseModel):
    stakes_type: Literal["personal", "faction", "immediate", "gradual", "custom"]
    summary: str
    consequences_success: str | None = None
    consequences_failure: str | None = None
    consequences_partial: str | None = None


class CampaignLobbyReserveInput(BaseModel):
    reserve_id: str
    assigned_character_id: str | None = None
    usage_notes: str | None = None
    status: Literal["planned", "spent", "earned"] = "planned"


class CampaignLobbyUpdateRequest(BaseModel):
    mission_name: constr(strip_whitespace=True, min_length=1)
    operation_code: str | None = None
    theater: str | None = None
    objectives: list[CampaignLobbyObjectiveInput] = Field(default_factory=list)
    stakes: CampaignLobbyStakesInput | None = None
    reserves: list[CampaignLobbyReserveInput] = Field(default_factory=list)
    briefing_notes: str | None = None
    support_assets: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)
    assigned_member_ids: list[str] = Field(default_factory=list)
    preferred_pilot_count: int = Field(default=4, ge=1, le=6)
    min_pilot_count: int = Field(default=3, ge=1, le=5)
    gm_notes: str | None = None
    status: Literal["draft", "ready", "launched", "cooldown"] | None = None


class CampaignInviteResendRequest(BaseModel):
    expires_in_hours: int | None = Field(default=168, ge=1, le=24 * 30)


class CampaignMissionLaunchRequest(BaseModel):
    environment: Literal["standard", "zero_g", "underwater"] = "standard"
    notes: str | None = None


class SessionLifecycleUpdateRequest(BaseModel):
    phase: Literal["downtime", "brief", "prep", "mission", "debrief"]
    status: Literal["pending", "in_progress", "complete"]
    summary: str | None = None
    gm_notes: str | None = None


class CampaignSessionOutcomeRequest(BaseModel):
    outcome: Literal["success", "partial", "failure", "catastrophic"]
    completion_score: float | None = Field(default=None, ge=0.0, le=1.0)
    debrief_notes: str | None = None
    reserves_spent: list[dict] = Field(default_factory=list)
    reserves_earned: list[dict] = Field(default_factory=list)
    rewards: list[str] = Field(default_factory=list)


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
    redeemed_by_user_id: str | None
    created_at: datetime
    updated_at: datetime


class CampaignInvitePreviewResponse(BaseModel):
    campaign_id: str
    campaign_name: str
    squad_name: str | None
    patron: str | None
    role: str
    status: str
    expires_at: datetime | None
    seat_warning: str | None
    ready_players: int
    preferred_pilots: int
    assigned_ready_players: int
    min_pilots: int
    lobby_status: str | None
    readiness_issues: list[str]
    can_join: bool


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


class CampaignReadinessSummary(BaseModel):
    ready_members: int
    ready_players: int
    assigned_ready_players: int
    total_members: int
    min_pilots: int
    preferred_pilots: int
    can_launch: bool
    lobby_status: str | None
    issues: list[str]
    member_issues: dict[str, list[str]]


class CampaignOutcomeSummary(BaseModel):
    total_missions: int
    successful_missions: int
    partial_missions: int
    failed_missions: int
    average_completion: float
    last_outcome: str | None
    last_mission_name: str | None
    last_mission_date: str | None


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
    lobby_status: str | None
    mission_summary: CampaignOutcomeSummary


class CampaignDetailResponse(DatabaseMetadata):
    name: str
    description: str
    status: str
    visibility: str
    data: dict[str, Any]
    members: list[CampaignMemberResponse]
    invites: list[CampaignInviteResponse]
    characters: list[CampaignCharacterResponse]
    readiness_summary: CampaignReadinessSummary
    seat_warning: str | None
    mission_summary: CampaignOutcomeSummary


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
        core_campaign = _load_campaign_model(campaign_db)
        mission_summary = _build_mission_summary(core_campaign)
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
                lobby_status=core_campaign.lobby_state.status
                if core_campaign.lobby_state
                else None,
                mission_summary=mission_summary,
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


@router.get("/{campaign_id}/export.pdf")
async def export_campaign_pdf(
    campaign_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    """Render a campaign briefing PDF using a server-side template."""
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    await _require_membership(session, campaign_id, user["id"])

    core_campaign = _load_campaign_model(campaign_db)
    pdf_bytes = render_campaign_brief_pdf(core_campaign)
    filename = f"campaign_{campaign_db.id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{campaign_id}/invites", response_model=CampaignInviteResponse)
async def create_invite(
    campaign_id: str,
    body: CampaignInviteCreateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignInviteResponse:
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    await _require_membership(
        session,
        campaign_id,
        user["id"],
        allowed_roles=("owner", "co_gm"),
    )
    core_campaign = _load_campaign_model(campaign_db)

    if body.role == "player":
        active_players = await _count_rows(
            session,
            CampaignMembershipDB,
            campaign_id=campaign_id,
            role="player",
            status="active",
        )
        pending_player_invites = await _count_rows(
            session,
            CampaignInviteDB,
            campaign_id=campaign_id,
            role="player",
            status="pending",
        )
        preferred = _get_seat_limits(core_campaign)[1]
        if active_players + pending_player_invites >= preferred:
            raise ConflictError(
                f"Player seats exceed recommended cap of {preferred} (PR2 GM + 3-5 pilots)"
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


@router.get("/invites/{token}/preview", response_model=CampaignInvitePreviewResponse)
async def preview_invite(
    token: str,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignInvitePreviewResponse:
    invite_result = await session.exec(
        select(CampaignInviteDB).where(CampaignInviteDB.token == token)
    )
    invite = invite_result.one_or_none()
    if invite is None:
        raise NotFoundError("Invite", token)

    campaign_db = await _get_campaign_or_404(session, invite.campaign_id)
    members_result = await session.exec(
        select(CampaignMembershipDB).where(
            CampaignMembershipDB.campaign_id == invite.campaign_id
        )
    )
    members = list(members_result.all())
    invites_result = await session.exec(
        select(CampaignInviteDB).where(
            CampaignInviteDB.campaign_id == invite.campaign_id
        )
    )
    invites = list(invites_result.all())
    core_campaign = _load_campaign_model(campaign_db)
    readiness = _build_readiness_summary(core_campaign, members)
    seat_warning = _compute_seat_warning(core_campaign, members, invites)
    identity = core_campaign.identity
    can_join = (
        invite.status == "pending"
        and readiness["ready_players"] < readiness["preferred_pilots"]
    )

    return CampaignInvitePreviewResponse(
        campaign_id=campaign_db.id,
        campaign_name=campaign_db.name,
        squad_name=identity.squad_name if identity else None,
        patron=identity.patron if identity else None,
        role=invite.role,
        status=invite.status,
        expires_at=invite.expires_at,
        seat_warning=seat_warning,
        ready_players=readiness["ready_players"],
        preferred_pilots=readiness["preferred_pilots"],
        assigned_ready_players=readiness["assigned_ready_players"],
        min_pilots=readiness["min_pilots"],
        lobby_status=readiness["lobby_status"],
        readiness_issues=readiness["issues"],
        can_join=can_join,
    )


@router.post(
    "/{campaign_id}/invites/{invite_id}/revoke", response_model=CampaignInviteResponse
)
async def revoke_invite(
    campaign_id: str,
    invite_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignInviteResponse:
    await _require_membership(
        session, campaign_id, user["id"], allowed_roles=("owner", "co_gm")
    )
    invite = await _get_invite_by_id(session, campaign_id, invite_id)
    invite.status = "revoked"
    invite.updated_at = datetime.utcnow()
    session.add(invite)
    await session.commit()
    await session.refresh(invite)
    return CampaignInviteResponse(**invite.model_dump())


@router.post(
    "/{campaign_id}/invites/{invite_id}/resend", response_model=CampaignInviteResponse
)
async def resend_invite(
    campaign_id: str,
    invite_id: str,
    body: CampaignInviteResendRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignInviteResponse:
    await _require_membership(
        session, campaign_id, user["id"], allowed_roles=("owner", "co_gm")
    )
    invite = await _get_invite_by_id(session, campaign_id, invite_id)
    invite.status = "pending"
    if body.expires_in_hours:
        invite.expires_at = datetime.utcnow() + timedelta(hours=body.expires_in_hours)
    else:
        invite.expires_at = None
    invite.updated_at = datetime.utcnow()
    session.add(invite)
    await session.commit()
    await session.refresh(invite)
    return CampaignInviteResponse(**invite.model_dump())


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


@router.post("/{campaign_id}/identity", response_model=CampaignDetailResponse)
async def update_campaign_identity(
    campaign_id: str,
    body: CampaignIdentityUpdateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    await _require_membership(
        session,
        campaign_id,
        user["id"],
        allowed_roles=("owner", "co_gm"),
    )
    core_campaign = _load_campaign_model(campaign_db)
    existing_identity = (
        core_campaign.identity.model_dump() if core_campaign.identity else {}
    )
    update_payload = existing_identity | {
        key: value for key, value in body.model_dump().items() if value is not None
    }
    core_campaign.identity = CampaignIdentity(**update_payload)
    _save_campaign_model(campaign_db, core_campaign)
    session.add(campaign_db)
    await session.commit()

    return await _build_campaign_detail_response(session, campaign_db, user["id"])


@router.post("/{campaign_id}/lobby", response_model=CampaignDetailResponse)
async def upsert_campaign_lobby(
    campaign_id: str,
    body: CampaignLobbyUpdateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    await _require_membership(
        session,
        campaign_id,
        user["id"],
        allowed_roles=("owner", "co_gm"),
    )
    members_result = await session.exec(
        select(CampaignMembershipDB).where(
            CampaignMembershipDB.campaign_id == campaign_id
        )
    )
    members = members_result.all()
    valid_member_ids = {member.id for member in members}
    invalid_assignments = [
        member_id
        for member_id in body.assigned_member_ids
        if member_id not in valid_member_ids
    ]
    if invalid_assignments:
        raise ConflictError(
            "Cannot assign unknown members to lobby: " + ", ".join(invalid_assignments)
        )
    if body.min_pilot_count > body.preferred_pilot_count:
        raise ConflictError("Minimum pilots cannot exceed preferred pilot count")

    mission_plan = MissionPrepPlan(
        mission_name=body.mission_name,
        operation_code=body.operation_code,
        theater=body.theater,
        objectives=[
            MissionObjectiveBrief(**obj.model_dump()) for obj in body.objectives
        ],
        stakes=MissionStakesBrief(**body.stakes.model_dump()) if body.stakes else None,
        reserves=[
            ReservePlanEntry(**reserve.model_dump()) for reserve in body.reserves
        ],
        briefing_notes=body.briefing_notes or "",
        support_assets=body.support_assets,
        threats=body.threats,
    )

    core_campaign = _load_campaign_model(campaign_db)
    lobby_state = core_campaign.lobby_state or CampaignLobbyState(
        mission_plan=mission_plan
    )
    lobby_state.mission_plan = mission_plan
    lobby_state.assigned_member_ids = body.assigned_member_ids
    lobby_state.preferred_pilot_count = body.preferred_pilot_count
    lobby_state.min_pilot_count = body.min_pilot_count
    lobby_state.gm_notes = body.gm_notes or ""
    lobby_state.last_ready_check = datetime.utcnow()
    if body.status is not None:
        lobby_state.status = body.status
    core_campaign.lobby_state = lobby_state

    _save_campaign_model(campaign_db, core_campaign)
    session.add(campaign_db)
    await session.commit()

    return await _build_campaign_detail_response(session, campaign_db, user["id"])


@router.post("/{campaign_id}/launch", response_model=CampaignDetailResponse)
async def launch_campaign_mission(
    campaign_id: str,
    body: CampaignMissionLaunchRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    await _require_membership(
        session,
        campaign_id,
        user["id"],
        allowed_roles=("owner", "co_gm"),
    )
    core_campaign = _load_campaign_model(campaign_db)
    lobby_state = core_campaign.lobby_state
    if lobby_state is None:
        raise ConflictError("No lobby data available to launch a mission")
    if not lobby_state.assigned_member_ids:
        raise ConflictError("Assign members to the lobby before launching")

    members_result = await session.exec(
        select(CampaignMembershipDB).where(
            CampaignMembershipDB.campaign_id == campaign_id
        )
    )
    members = members_result.all()
    members_by_id = {member.id: member for member in members}
    assigned_members = []
    for member_id in lobby_state.assigned_member_ids:
        member = members_by_id.get(member_id)
        if member is None:
            raise ConflictError(
                f"Assigned member {member_id} is no longer part of campaign"
            )
        if member.ready_state != "ready":
            raise ConflictError("All assigned members must be marked ready to launch")
        if member.assigned_character_id is None:
            raise ConflictError(
                f"Member {member.user_id} must assign a character before launch"
            )
        assigned_members.append(member)

    min_required, preferred = _get_seat_limits(core_campaign)
    player_assignments = [m for m in assigned_members if m.role == "player"]
    if len(player_assignments) < min_required:
        raise ConflictError(
            f"Launch requires at least {min_required} ready pilots (PR2 cadence guidance)"
        )
    if len(player_assignments) > preferred:
        raise ConflictError(
            f"Launch exceeds preferred pilot cap of {preferred} (PR2 GM + 3-5 pilots)"
        )

    character_ids = {
        m.assigned_character_id for m in assigned_members if m.assigned_character_id
    }
    if not character_ids:
        raise ConflictError("No characters assigned to launch")
    character_column = getattr(CharacterDB, "id")
    characters_result = await session.exec(
        select(CharacterDB).where(character_column.in_(character_ids))
    )
    character_map = {character.id: character for character in characters_result}
    combatants: list[CombatantState] = []
    for member in assigned_members:
        character_db = character_map.get(member.assigned_character_id)
        if character_db is None:
            raise ConflictError(
                f"Character {member.assigned_character_id} is no longer attached to campaign"
            )
        combatants.append(_character_to_combatant(character_db))

    scenario = MechCombatScenario(
        combatants=combatants,
        environment=body.environment,
        rounds=[],
        grapples=[],
        deployables={},
    )

    combat_session_id = _generate_id("combat")
    session_id = _generate_id("campsession")
    combat_session = CombatSessionDB(
        id=combat_session_id,
        name=f"{core_campaign.name} • {lobby_state.mission_plan.mission_name}",
        status="active",
        current_round=1,
        current_turn_index=0,
        scenario=scenario.model_dump(mode="json"),
        gm_user_id=user["id"],
        campaign_id=campaign_id,
        campaign_session_id=session_id,
        notes=body.notes or "",
    )
    session.add(combat_session)

    new_session = Session(
        id=session_id,
        session_number=len(core_campaign.sessions) + 1,
        mission_plan=lobby_state.mission_plan,
    )
    now = datetime.utcnow()
    for checkpoint in new_session.lifecycle_checkpoints:
        if checkpoint.phase in {"downtime", "brief", "prep"}:
            checkpoint.status = "complete"
            checkpoint.completed_at = now
        elif checkpoint.phase == "mission":
            checkpoint.status = "in_progress"
    core_campaign.sessions.append(new_session)

    lobby_state.status = "launched"
    lobby_state.combat_session_id = combat_session_id
    lobby_state.last_ready_check = now
    core_campaign.lobby_state = lobby_state

    _save_campaign_model(campaign_db, core_campaign)
    session.add(campaign_db)
    await session.commit()

    return await _build_campaign_detail_response(session, campaign_db, user["id"])


@router.post(
    "/{campaign_id}/sessions/{session_id}/lifecycle",
    response_model=CampaignDetailResponse,
)
async def update_session_lifecycle(
    campaign_id: str,
    session_id: str,
    body: SessionLifecycleUpdateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    await _require_membership(
        session,
        campaign_id,
        user["id"],
        allowed_roles=("owner", "co_gm"),
    )
    core_campaign = _load_campaign_model(campaign_db)
    session_obj = next((s for s in core_campaign.sessions if s.id == session_id), None)
    if session_obj is None:
        raise NotFoundError("Campaign session", session_id)
    for checkpoint in session_obj.lifecycle_checkpoints:
        if checkpoint.phase == body.phase:
            checkpoint.status = body.status
            if body.summary is not None:
                checkpoint.summary = body.summary
            if body.gm_notes is not None:
                checkpoint.gm_notes = body.gm_notes
            if body.status == "complete" and checkpoint.completed_at is None:
                checkpoint.completed_at = datetime.utcnow()
            break
    else:
        raise ConflictError(f"Unknown lifecycle phase: {body.phase}")

    _save_campaign_model(campaign_db, core_campaign)
    session.add(campaign_db)
    await session.commit()

    return await _build_campaign_detail_response(session, campaign_db, user["id"])


@router.post(
    "/{campaign_id}/sessions/{session_id}/outcome",
    response_model=CampaignDetailResponse,
)
async def record_session_outcome_endpoint(
    campaign_id: str,
    session_id: str,
    body: CampaignSessionOutcomeRequest,
    session: AsyncSession = Depends(get_session),
    user: dict[str, Any] = Depends(get_current_user),
) -> CampaignDetailResponse:
    await _require_membership(
        session,
        campaign_id,
        user["id"],
        allowed_roles=("owner", "co_gm"),
    )
    mission_outcome = MissionOutcomeReport(**body.model_dump())
    await record_campaign_session_outcome(
        session,
        campaign_id,
        session_id,
        mission_outcome,
    )
    await session.commit()
    campaign_db = await _get_campaign_or_404(session, campaign_id)
    return await _build_campaign_detail_response(session, campaign_db, user["id"])


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
    members = list(members_result.all())

    invites_result = await session.exec(
        select(CampaignInviteDB)
        .where(CampaignInviteDB.campaign_id == campaign_db.id)
        .order_by(CampaignInviteDB.created_at.desc())
    )
    invites = list(invites_result.all())

    characters_result = await session.exec(
        select(CampaignCharacterDB, CharacterDB)
        .join(CharacterDB, CharacterDB.id == CampaignCharacterDB.character_id)
        .where(CampaignCharacterDB.campaign_id == campaign_db.id)
    )
    character_rows = list(characters_result.all())

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

    core_campaign = _load_campaign_model(campaign_db)
    readiness_summary = CampaignReadinessSummary(
        **_build_readiness_summary(core_campaign, members)
    )
    mission_summary = _build_mission_summary(core_campaign)
    seat_warning = _compute_seat_warning(core_campaign, members, invites)

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
        data=core_campaign.model_dump(mode="json"),
        members=member_payload,
        invites=invite_payload,
        characters=character_payload,
        readiness_summary=readiness_summary,
        seat_warning=seat_warning,
        mission_summary=mission_summary,
    )


async def record_campaign_session_outcome(
    session: AsyncSession,
    campaign_id: str,
    session_id: str,
    outcome: MissionOutcomeReport,
    *,
    clear_lobby: bool = True,
) -> CampaignDB:
    """Persist a mission outcome onto the campaign blob."""

    campaign_db = await _get_campaign_or_404(session, campaign_id)
    core_campaign = _load_campaign_model(campaign_db)
    session_entry = next(
        (s for s in core_campaign.sessions if s.id == session_id), None
    )
    if session_entry is None:
        raise NotFoundError("Campaign session", session_id)

    session_entry.mission_outcome = outcome
    if outcome.debrief_notes:
        session_entry.debrief = outcome.debrief_notes

    for checkpoint in session_entry.lifecycle_checkpoints:
        if checkpoint.phase == "mission":
            checkpoint.status = "complete"
            if checkpoint.completed_at is None:
                checkpoint.completed_at = datetime.utcnow()
            break

    mission_plan = session_entry.mission_plan
    if mission_plan is None and core_campaign.lobby_state:
        mission_plan = core_campaign.lobby_state.mission_plan
    mission_name = (
        mission_plan.mission_name
        if mission_plan
        else f"Session {session_entry.session_number}"
    )
    mission_id = (
        mission_plan.operation_code
        if mission_plan and mission_plan.operation_code
        else session_id
    )

    record_payload = {
        "mission_id": mission_id,
        "session_id": session_id,
        "mission_name": mission_name,
        "outcome": outcome.outcome,
        "completion_score": outcome.completion_score,
        "participating_character_ids": [],
        "debrief_notes": outcome.debrief_notes,
        "reserves_spent": outcome.reserves_spent,
        "reserves_earned": outcome.reserves_earned,
        "rewards": outcome.rewards,
    }
    existing_record = next(
        (
            record
            for record in core_campaign.mission_history
            if record.session_id == session_id
        ),
        None,
    )
    if existing_record:
        for field, value in record_payload.items():
            setattr(existing_record, field, value)
    else:
        core_campaign.mission_history.append(CampaignMissionRecord(**record_payload))

    if core_campaign.lobby_state and clear_lobby:
        lobby_state = core_campaign.lobby_state
        if lobby_state.status == "launched":
            lobby_state.status = "cooldown"
        lobby_state.combat_session_id = None
        lobby_state.assigned_member_ids = []
        core_campaign.lobby_state = lobby_state

    core_campaign.modified_at = datetime.utcnow()
    _save_campaign_model(campaign_db, core_campaign)
    session.add(campaign_db)
    return campaign_db
