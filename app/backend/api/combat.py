"""Combat session CRUD endpoints with core model validation.

This module provides API endpoints for managing combat sessions.
It directly uses core.mech.combat_state models - no duplicate schemas.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session
from app.backend.db.models import CombatSessionDB, utc_now
from app.backend.dependencies import get_current_user
from app.backend.exceptions import ConflictError, NotFoundError, ValidationError
from app.backend.api.campaigns import record_campaign_session_outcome
from app.backend.api.combat_ws import combat_ws_manager

# Import core combat models - use directly, don't duplicate!
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    CombatTurn,
    CombatRound,
)
from core.mech.grid import HexPosition, HexCoord
from core.shared.campaign.campaign import MissionOutcomeReport
from core.mech.action_economy import ActionEconomyState
from core.mech.combat_execution import (
    start_turn,
    end_turn,
    execute_action,
    execute_reaction,
    get_available_actions,
    get_current_actor,
    apply_damage,
    ActionExecutionInput,
    ReactionInput,
)
from core.mech.combat_rules import DEFAULT_MECH_COMBAT_RULES
from core.shared.overwatch import check_overwatch_triggers_for_movement
from core.shared.full_tech import FullTechOptionSelection
from core.shared.decisions import (
    DecisionResolution,
    resolve_save_decision,
    resolve_trauma_decision,
    get_pending_decisions_for_combatant,
    remove_decision_from_scenario,
    apply_system_trauma_selection,
    apply_failed_hull_save,
)
from core.mech.npc_turn_execution import execute_npc_turn

router = APIRouter(prefix="/combat", tags=["combat"])


# =============================================================================
# Demo Combat Scenario Types
# =============================================================================


DemoScenarioType = Literal["skirmish", "control", "boss"]


# =============================================================================
# Type Aliases (reuse from core where possible)
# =============================================================================

SessionStatus = Literal["active", "paused", "completed", "abandoned"]


# =============================================================================
# Request Schemas - Minimal wrappers, delegate to core
# =============================================================================


class CombatSessionCreateRequest(BaseModel):
    """Request body for creating a combat session.

    Combatants use the exact same structure as core.mech.combat_state.CombatantState.
    """

    name: str = Field(..., min_length=1, description="Session name")
    environment: Literal["standard", "zero_g", "underwater"] = "standard"
    combatants: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of combatants (validated against CombatantState)",
    )
    notes: str = Field(default="")
    campaign_id: str | None = None


class CombatSessionUpdateRequest(BaseModel):
    """Request body for updating session metadata."""

    name: str | None = None
    status: SessionStatus | None = None
    notes: str | None = None


class AddCombatantRequest(BaseModel):
    """Request body for adding a combatant.

    Combatant dict is validated against core.mech.combat_state.CombatantState.
    """

    combatant: dict[str, Any]


class CombatSessionCompleteRequest(BaseModel):
    """Request body for completing a combat session and logging outcomes."""

    outcome: Literal["success", "partial", "failure", "catastrophic"]
    completion_score: float | None = Field(default=None, ge=0.0, le=1.0)
    debrief_notes: str | None = None
    reserves_spent: list[dict] = Field(default_factory=list)
    reserves_earned: list[dict] = Field(default_factory=list)
    rewards: list[str] = Field(default_factory=list)
    notes: str | None = None


class SpendReserveRequest(BaseModel):
    """Request body for spending a mission reserve."""

    reserve_id: str = Field(..., description="ID of the reserve to spend")


# =============================================================================
# Response Schemas
# =============================================================================


class CombatSessionResponse(BaseModel):
    """Response model for a combat session."""

    # Database metadata
    id: str
    gm_user_id: str
    campaign_id: str | None
    created_at: datetime
    updated_at: datetime

    # Session state
    name: str
    status: SessionStatus
    current_round: int
    current_turn_index: int
    notes: str

    # Full scenario - serialized from core model
    scenario: dict[str, Any]


class CombatSessionListItem(BaseModel):
    """Compact response for listing sessions."""

    id: str
    name: str
    status: SessionStatus
    current_round: int
    combatant_count: int
    environment: str
    campaign_id: str | None
    created_at: datetime
    updated_at: datetime


class CombatSessionListResponse(BaseModel):
    """Response model for listing combat sessions."""

    items: list[CombatSessionListItem]
    total: int


# =============================================================================
# Helper Functions - Use core models directly
# =============================================================================


def _validate_combatant(data: dict[str, Any]) -> CombatantState:
    """Validate combatant dict against core CombatantState.

    Core model handles all validation - we just wrap errors.
    """
    try:
        return CombatantState.model_validate(data)
    except PydanticValidationError as e:
        errors = [
            {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
            for err in e.errors()
        ]
        raise ValidationError("Invalid combatant data", errors=errors)


def _validate_scenario(
    combatants: list[dict[str, Any]],
    environment: str,
) -> MechCombatScenario:
    """Build and validate scenario using core models."""
    validated_combatants = [_validate_combatant(c) for c in combatants]

    try:
        return MechCombatScenario(
            combatants=validated_combatants,
            environment=environment,
            rounds=[],
            grapples=[],
            deployables={},
        )
    except PydanticValidationError as e:
        errors = [
            {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
            for err in e.errors()
        ]
        raise ValidationError("Invalid combat scenario", errors=errors)


def _session_to_response(session_db: CombatSessionDB) -> CombatSessionResponse:
    """Convert DB record to response - scenario already validated on write."""
    return CombatSessionResponse(
        id=session_db.id,
        gm_user_id=session_db.gm_user_id,
        campaign_id=session_db.campaign_id,
        created_at=session_db.created_at,
        updated_at=session_db.updated_at,
        name=session_db.name,
        status=session_db.status,
        current_round=session_db.current_round,
        current_turn_index=session_db.current_turn_index,
        notes=session_db.notes,
        scenario=session_db.scenario,
    )


def _session_to_list_item(session_db: CombatSessionDB) -> CombatSessionListItem:
    """Convert DB record to list item."""
    scenario = session_db.scenario
    return CombatSessionListItem(
        id=session_db.id,
        name=session_db.name,
        status=session_db.status,
        current_round=session_db.current_round,
        combatant_count=len(scenario.get("combatants", [])),
        environment=scenario.get("environment", "standard"),
        campaign_id=session_db.campaign_id,
        created_at=session_db.created_at,
        updated_at=session_db.updated_at,
    )


# =============================================================================
# Demo Combat Helpers
# =============================================================================


def _build_demo_player_squad() -> list[CombatantState]:
    """Build 4 LL0 pilots with GMS Everest mechs for demo combat.

    Squad composition:
    - VANGUARD: Melee focus
    - SENTINEL: Ranged focus
    - WARDEN: Support/tank
    - GHOST: Tech/mobility
    """
    players = []
    positions = [(0, -3), (-2, -2), (2, -2), (0, -1)]  # Back row deployment
    pilot_configs = [
        ("VANGUARD", "Alpha"),
        ("SENTINEL", "Bravo"),
        ("WARDEN", "Charlie"),
        ("GHOST", "Delta"),
    ]

    for idx, (callsign, name) in enumerate(pilot_configs):
        q, r = positions[idx]
        combatant = CombatantState(
            id=f"demo_player_{idx + 1}",
            name=f"{callsign}'s Everest",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,  # GMS Everest base HP
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
                tech_attack=0,
                grit=0,  # LL0 grit
            ),
            resources=CombatResources(
                hp_current=10,
                heat_current=0,
                heat_cap=6,
                structure_current=4,
                stress_current=4,
                repairs_remaining=4,
            ),
            position=HexPosition(coord=HexCoord(q=q, r=r), elevation=0),
        )
        players.append(combatant)

    return players


def _build_demo_enemy_squad(scenario_type: DemoScenarioType) -> list[CombatantState]:
    """Build enemy squads based on scenario type.

    Scenario types:
    - skirmish: 4× GMS Grunt T1 (easy)
    - control: 4× GMS Grunt T1 + 2× GMS Elite T2 (standard)
    - boss: 1× IPS-N Boss T3 + 2× GMS Grunt T1 (hard)
    """
    enemies: list[CombatantState] = []

    if scenario_type == "skirmish":
        # 4 grunts spread across front row
        positions = [(0, 3), (-2, 2), (2, 2), (0, 1)]
        for idx, (q, r) in enumerate(positions):
            enemies.append(_make_grunt(f"grunt_{idx + 1}", q, r))

    elif scenario_type == "control":
        # 4 grunts + 2 elites
        grunt_positions = [(0, 3), (-2, 2), (2, 2), (0, 1)]
        for idx, (q, r) in enumerate(grunt_positions):
            enemies.append(_make_grunt(f"grunt_{idx + 1}", q, r))

        elite_positions = [(-1, 2), (1, 2)]
        for idx, (q, r) in enumerate(elite_positions):
            enemies.append(_make_elite(f"elite_{idx + 1}", q, r))

    elif scenario_type == "boss":
        # 1 boss + 2 grunts
        enemies.append(_make_boss("boss_1", 0, 3))
        enemies.append(_make_grunt("grunt_1", -2, 2))
        enemies.append(_make_grunt("grunt_2", 2, 2))

    return enemies


def _make_grunt(id: str, q: int, r: int) -> CombatantState:
    """Create a GMS Grunt T1 NPC."""
    return CombatantState(
        id=f"demo_{id}",
        name=f"GMS Grunt ({id.replace('_', ' ').title()})",
        side="hostiles",
        kind="npc",
        stats=CombatStats(
            size="size_1",
            hp_max=8,
            evasion=8,
            e_defense=6,
            armor=0,
            speed=4,
            sensor_range=10,
            tech_attack=0,
        ),
        resources=CombatResources(
            hp_current=8,
            heat_current=0,
            heat_cap=4,
            structure_current=1,
            stress_current=1,
            repairs_remaining=0,
        ),
        position=HexPosition(coord=HexCoord(q=q, r=r), elevation=0),
        ai_controlled=True,
        npc_role="striker",  # Grunts are aggressive
    )


def _make_elite(id: str, q: int, r: int) -> CombatantState:
    """Create a GMS Elite T2 NPC."""
    return CombatantState(
        id=f"demo_{id}",
        name=f"GMS Elite ({id.replace('_', ' ').title()})",
        side="hostiles",
        kind="npc",
        stats=CombatStats(
            size="size_1",
            hp_max=15,
            evasion=10,
            e_defense=8,
            armor=1,
            speed=5,
            sensor_range=10,
            tech_attack=2,
        ),
        resources=CombatResources(
            hp_current=15,
            heat_current=0,
            heat_cap=6,
            structure_current=2,
            stress_current=2,
            repairs_remaining=0,
        ),
        position=HexPosition(coord=HexCoord(q=q, r=r), elevation=0),
        ai_controlled=True,
        npc_role="defender",  # Elites hold ground
    )


def _make_boss(id: str, q: int, r: int) -> CombatantState:
    """Create an IPS-N Boss T3 NPC."""
    return CombatantState(
        id=f"demo_{id}",
        name="IPS-N Commander",
        side="hostiles",
        kind="npc",
        stats=CombatStats(
            size="size_2",
            hp_max=25,
            evasion=8,
            e_defense=10,
            armor=2,
            speed=3,
            sensor_range=8,
            tech_attack=0,
        ),
        resources=CombatResources(
            hp_current=25,
            heat_current=0,
            heat_cap=8,
            structure_current=3,
            stress_current=3,
            repairs_remaining=0,
        ),
        position=HexPosition(coord=HexCoord(q=q, r=r), elevation=0),
        ai_controlled=True,
        npc_role="striker",  # Bosses deal damage
    )


def _build_demo_terrain() -> None:
    """Build simple demo terrain.

    For demo purposes, we use no terrain (open field).
    This keeps the demo simple and focused on combat mechanics.
    """
    # Return None for a simple open battlefield
    # Future enhancement: add basic terrain with cover
    return None


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/demo", response_model=CombatSessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_demo_combat(
    scenario_type: DemoScenarioType = "skirmish",
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Create a demo combat session with pre-built combatants.

    This endpoint provides a quick way to start a combat session without
    requiring campaign setup or character creation. Perfect for testing
    and demonstrations.

    Scenario types:
    - skirmish: 4 players vs 4 GMS grunts (easy)
    - control: 4 players vs 6 enemies (4 grunts + 2 elites, standard)
    - boss: 4 players vs 1 boss + 2 grunts (hard)
    """
    session_id = f"combat_{uuid4().hex[:12]}"

    # Generate demo squads
    player_combatants = _build_demo_player_squad()
    enemy_combatants = _build_demo_enemy_squad(scenario_type)
    terrain = _build_demo_terrain()

    # Build scenario
    scenario = MechCombatScenario(
        combatants=player_combatants + enemy_combatants,
        terrain=terrain,
        environment="standard",
        grapples=[],
        rounds=[],
        deployables={},
    )

    # Create session name based on scenario type
    scenario_names = {
        "skirmish": "Demo Skirmish",
        "control": "Demo Control",
        "boss": "Demo Boss Fight",
    }

    db_session = CombatSessionDB(
        id=session_id,
        name=scenario_names.get(scenario_type, "Demo Combat"),
        status="active",
        current_round=1,
        current_turn_index=0,
        scenario=scenario.model_dump(mode="json"),
        gm_user_id=user["id"],
        campaign_id=None,  # Demo sessions have no campaign
        notes=f"Quick Battle demo ({scenario_type} scenario)",
    )

    session.add(db_session)
    await session.commit()
    await session.refresh(db_session)

    return _session_to_response(db_session)


@router.post(
    "", response_model=CombatSessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_combat_session(
    body: CombatSessionCreateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Create a new combat session.

    Combatants are validated against core.mech.combat_state.CombatantState.
    """
    session_id = f"combat_{uuid4().hex[:12]}"

    # Core models handle all validation
    scenario = _validate_scenario(body.combatants, body.environment)

    db_session = CombatSessionDB(
        id=session_id,
        name=body.name,
        status="active",
        current_round=1,
        current_turn_index=0,
        scenario=scenario.model_dump(mode="json"),
        gm_user_id=user["id"],
        campaign_id=body.campaign_id,
        notes=body.notes,
    )

    session.add(db_session)
    await session.commit()
    await session.refresh(db_session)

    return _session_to_response(db_session)


@router.get("", response_model=CombatSessionListResponse)
async def list_combat_sessions(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
    status_filter: SessionStatus | None = None,
    campaign_id: str | None = None,
) -> CombatSessionListResponse:
    """List combat sessions for the current user."""
    query = select(CombatSessionDB).where(CombatSessionDB.gm_user_id == user["id"])

    if status_filter:
        query = query.where(CombatSessionDB.status == status_filter)
    if campaign_id:
        query = query.where(CombatSessionDB.campaign_id == campaign_id)

    result = await session.exec(query)
    sessions = result.all()

    return CombatSessionListResponse(
        items=[_session_to_list_item(s) for s in sessions],
        total=len(sessions),
    )


@router.get("/{session_id}", response_model=CombatSessionResponse)
async def get_combat_session(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Get a combat session by ID with full scenario."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    return _session_to_response(combat_session)


@router.put("/{session_id}", response_model=CombatSessionResponse)
async def update_combat_session(
    session_id: str,
    body: CombatSessionUpdateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Update combat session metadata."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if body.name is not None:
        combat_session.name = body.name
    if body.status is not None:
        combat_session.status = body.status
    if body.notes is not None:
        combat_session.notes = body.notes

    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    return _session_to_response(combat_session)


@router.post("/{session_id}/complete", response_model=CombatSessionResponse)
async def complete_combat_session(
    session_id: str,
    body: CombatSessionCompleteRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Mark a combat session complete and propagate mission outcomes."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)
    if combat_session.status == "completed":
        raise ConflictError("Combat session already completed")

    combat_session.status = "completed"
    if body.notes is not None:
        combat_session.notes = body.notes
    combat_session.updated_at = utc_now()

    payload = body.model_dump(exclude={"notes"})
    if payload.get("completion_score") is None:
        payload.pop("completion_score", None)
    mission_outcome = MissionOutcomeReport(**payload)

    if combat_session.campaign_id and combat_session.campaign_session_id:
        await record_campaign_session_outcome(
            session,
            combat_session.campaign_id,
            combat_session.campaign_session_id,
            mission_outcome,
            combat_session=combat_session,
        )

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    return _session_to_response(combat_session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_combat_session(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> None:
    """Delete a combat session."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    await session.delete(combat_session)
    await session.commit()


@router.post("/{session_id}/combatants", response_model=CombatSessionResponse)
async def add_combatant(
    session_id: str,
    body: AddCombatantRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Add a combatant to an existing combat session."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    # Hydrate and validate
    scenario = MechCombatScenario.model_validate(combat_session.scenario)
    new_combatant = _validate_combatant(body.combatant)

    # Check for duplicate ID
    existing_ids = {c.id for c in scenario.combatants}
    if new_combatant.id in existing_ids:
        raise ValidationError(
            f"Combatant with ID '{new_combatant.id}' already exists",
            errors=[
                {
                    "loc": ["combatant", "id"],
                    "msg": "Duplicate combatant ID",
                    "type": "value_error",
                }
            ],
        )

    # Update scenario with new combatant
    updated_scenario = MechCombatScenario(
        combatants=list(scenario.combatants) + [new_combatant],
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    return _session_to_response(combat_session)


@router.delete(
    "/{session_id}/combatants/{combatant_id}", response_model=CombatSessionResponse
)
async def remove_combatant(
    session_id: str,
    combatant_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Remove a combatant from a combat session."""
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Find and remove combatant
    updated_combatants = [c for c in scenario.combatants if c.id != combatant_id]
    if len(updated_combatants) == len(scenario.combatants):
        raise NotFoundError("Combatant", combatant_id)

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=[
            g
            for g in scenario.grapples
            if g.grappler_id != combatant_id and g.target_id != combatant_id
        ],
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    return _session_to_response(combat_session)


# =============================================================================
# Combat Turn Execution Request/Response Schemas
# =============================================================================


class TurnStartResponse(BaseModel):
    """Response model for starting a combat turn."""

    actor_id: str
    actor_name: str
    economy: dict[str, Any]
    available_actions: list[str]
    prepared_action_expired: bool
    cooldowns_decremented: list[str]
    scenario: dict[str, Any]


class ActionRequest(BaseModel):
    """Request body for executing a combat action."""

    action_id: str = Field(..., description="Action identifier")
    action_type: Literal["full", "quick", "free", "reaction", "protocol", "move"] = Field(
        ..., description="Type of action"
    )
    target_ids: list[str] = Field(default_factory=list, description="Target combatant IDs")
    target_position: dict[str, Any] | None = Field(default=None, description="Target position")
    weapon_id: str | None = Field(default=None, description="Weapon to use")
    weapon_profile_id: str | None = Field(
        default=None, description="Weapon profile for weapons with multiple profiles"
    )
    system_id: str | None = Field(default=None, description="System to activate")
    full_tech_first: FullTechOptionSelection | None = Field(
        default=None, description="First Full Tech option"
    )
    full_tech_second: FullTechOptionSelection | None = Field(
        default=None, description="Second Full Tech option"
    )
    movement_path: list[dict[str, Any]] = Field(default_factory=list, description="Movement path")
    prompt_dangerous_terrain: bool = Field(
        default=False,
        description="Whether to prompt for dangerous terrain checks (players only)",
    )
    is_overcharge: bool = Field(default=False, description="Whether this uses overcharge")
    use_thrown: bool = Field(
        default=False,
        description="Whether to treat a melee weapon attack as thrown",
    )
    # Mount/Deployable targeting (Phase 60)
    target_mount_id: int | None = Field(
        default=None,
        ge=0,
        description="Mount index to target for called shot (adds +1 difficulty)",
    )
    target_deployable_id: str | None = Field(
        default=None,
        description="ID of deployable to target instead of combatant",
    )


class ActionResponse(BaseModel):
    """Response model for executing a combat action."""

    success: bool
    error: str | None = None
    action_use: dict[str, Any] | None = None
    effects_applied: list[dict[str, Any]]
    damage_dealt: int
    heat_generated: int
    economy: dict[str, Any]
    scenario: dict[str, Any]


class TurnEndResponse(BaseModel):
    """Response model for ending a combat turn."""

    actor_id: str
    next_actor_id: str | None
    next_actor_name: str | None
    round_advanced: bool
    new_round_number: int | None
    end_of_turn_effects: list[dict[str, Any]]
    scenario: dict[str, Any]


class ReactionRequest(BaseModel):
    """Request body for declaring a reaction."""

    reactor_id: str = Field(..., description="ID of the reacting combatant")
    reaction_type: Literal["brace", "overwatch"] = Field(..., description="Type of reaction")
    trigger_action_id: str | None = Field(default=None, description="Action that triggered this")
    target_ids: list[str] = Field(default_factory=list, description="Targets for the reaction")
    weapon_id: str | None = Field(default=None, description="Weapon for overwatch")


class ReactionResponse(BaseModel):
    """Response model for declaring a reaction."""

    success: bool
    error: str | None = None
    reaction_used: str | None = None
    effects_applied: list[dict[str, Any]]
    damage_dealt: int
    scenario: dict[str, Any]


class AvailableActionItem(BaseModel):
    """Single available action."""

    action_id: str
    action_name: str
    action_type: str
    is_available: bool
    unavailable_reason: str | None = None
    requires_target: bool
    requires_weapon: bool
    requires_system: bool = False
    requires_path: bool = False
    max_targets: int = 1


class AvailableActionsResponse(BaseModel):
    """Response model for available actions."""

    actor_id: str
    economy: dict[str, Any]
    full_actions: list[AvailableActionItem]
    quick_actions: list[AvailableActionItem]
    free_actions: list[AvailableActionItem]
    reactions: list[AvailableActionItem]
    protocols: list[AvailableActionItem]
    can_overcharge: bool
    overcharge_level: int = 0


class ReactionTrigger(BaseModel):
    """A pending reaction trigger for a combatant."""

    trigger_type: Literal["attack_incoming", "enemy_movement"]
    triggering_actor_id: str
    triggering_actor_name: str
    triggering_action_id: str | None = None
    available_reactions: list[Literal["brace", "overwatch"]]


class ReactionOpportunityResponse(BaseModel):
    """Response model for checking reaction opportunities."""

    combatant_id: str
    combatant_name: str
    has_reaction_available: bool
    pending_triggers: list[ReactionTrigger]


# =============================================================================
# Decision Request/Response Schemas
# =============================================================================


class PendingDecisionItem(BaseModel):
    """A pending decision for a combatant."""

    decision_id: str
    decision_type: Literal["hull_save", "engineering_save", "engineering_check", "system_trauma"]
    trigger_source: str
    trigger_round: int

    # Save-specific
    save_type: Literal["hull", "agility", "systems", "engineering"] | None = None
    save_target: int | None = None
    save_bonus: int = 0

    # Trauma-specific
    trauma_target: Literal["mount", "system"] | None = None
    eligible_mounts: list[int] = Field(default_factory=list)
    eligible_systems: list[str] = Field(default_factory=list)

    # Reroll availability
    reroll_available: bool = False
    reroll_source: str | None = None


class PendingDecisionsResponse(BaseModel):
    """Response model for pending decisions."""

    combatant_id: str
    combatant_name: str
    pending_decisions: list[PendingDecisionItem]
    has_pending: bool


class DecisionSubmitRequest(BaseModel):
    """Request body for submitting a decision."""

    decision_id: str = Field(..., description="ID of the decision to resolve")
    combatant_id: str = Field(..., description="ID of the combatant making the decision")
    choice: Literal["roll", "voluntary_fail", "use_reroll"] = Field(
        ..., description="Player's chosen action"
    )
    selected_mount_index: int | None = Field(
        default=None, description="Mount index for system trauma"
    )
    selected_system_id: str | None = Field(
        default=None, description="System ID for system trauma"
    )


class DecisionResultResponse(BaseModel):
    """Response model for decision submission."""

    success: bool
    error: str | None = None
    roll_result: int | None = None
    save_succeeded: bool | None = None
    effects_applied: list[dict[str, Any]]
    scenario: dict[str, Any]


# =============================================================================
# Session-scoped Economy Tracking
# =============================================================================

# In-memory economy state per session (for simplicity; production would use Redis/DB)
_session_economy_cache: dict[str, ActionEconomyState] = {}


def _get_session_economy(session_id: str) -> ActionEconomyState:
    """Get or create economy state for a session."""
    if session_id not in _session_economy_cache:
        _session_economy_cache[session_id] = ActionEconomyState()
    return _session_economy_cache[session_id]


def _set_session_economy(session_id: str, economy: ActionEconomyState) -> None:
    """Update economy state for a session."""
    _session_economy_cache[session_id] = economy


def _clear_session_economy(session_id: str) -> None:
    """Clear economy state (on turn end/start)."""
    _session_economy_cache.pop(session_id, None)


# =============================================================================
# Combat Turn Execution Endpoints
# =============================================================================


@router.post("/{session_id}/turns/start", response_model=TurnStartResponse)
async def start_combat_turn(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> TurnStartResponse:
    """Initialize the current actor's turn.

    This resets the action economy, expires prepared actions,
    and decrements turn-start cooldowns.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot start turn: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Initialize turn order if empty (first turn of combat)
    if not scenario.rounds and scenario.combatants:
        turns = [CombatTurn(actor_id=c.id) for c in scenario.combatants]
        round_1 = CombatRound(round_index=1, turns=turns)
        scenario = MechCombatScenario(
            combatants=scenario.combatants,
            rounds=[round_1],
            grapples=list(scenario.grapples),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
        )
        # Reset turn index to start of round
        combat_session.current_round = 1
        combat_session.current_turn_index = 0

    # Get current actor from turn order
    current_actor = get_current_actor(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    if current_actor is None:
        raise ValidationError(
            "No current actor found",
            errors=[{"loc": ["turn"], "msg": "Turn order not initialized", "type": "value_error"}],
        )

    # Start the turn using core helper
    updated_scenario, turn_result = start_turn(scenario, current_actor.id)

    # Persist updated scenario
    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Reset economy for this turn
    _set_session_economy(session_id, turn_result.economy)

    # Broadcast state update to all connected WebSocket clients
    response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": response.model_dump(mode="json")},
    )

    return TurnStartResponse(
        actor_id=turn_result.actor_id,
        actor_name=turn_result.actor_name,
        economy=turn_result.economy.model_dump(),
        available_actions=turn_result.available_actions,
        prepared_action_expired=turn_result.prepared_action_expired,
        cooldowns_decremented=turn_result.cooldowns_decremented,
        scenario=combat_session.scenario,
    )


@router.post("/{session_id}/actions", response_model=ActionResponse)
async def execute_combat_action(
    session_id: str,
    body: ActionRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> ActionResponse:
    """Execute a combat action.

    Validates the action against the current economy, resolves effects,
    and records the action in the combat log.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot execute action: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Get current actor
    current_actor = get_current_actor(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    if current_actor is None:
        raise ValidationError(
            "No current actor found",
            errors=[{"loc": ["turn"], "msg": "Turn not started", "type": "value_error"}],
        )

    # Get or create current turn
    round_idx = combat_session.current_round - 1
    turn_idx = combat_session.current_turn_index
    current_turn: CombatTurn

    if round_idx < len(scenario.rounds) and turn_idx < len(scenario.rounds[round_idx].turns):
        current_turn = scenario.rounds[round_idx].turns[turn_idx]
    else:
        current_turn = CombatTurn(actor_id=current_actor.id)

    # Get current economy
    economy = _get_session_economy(session_id)

    # Parse target position if provided
    target_position: HexPosition | None = None
    if body.target_position:
        try:
            target_position = HexPosition.model_validate(body.target_position)
        except Exception:
            pass

    # Parse movement path
    movement_path: list[HexPosition] = []
    for pos_dict in body.movement_path:
        try:
            movement_path.append(HexPosition.model_validate(pos_dict))
        except Exception:
            pass

    action_target_ids = body.target_ids
    if body.action_id == "full_tech" and body.full_tech_first and body.full_tech_second:
        action_target_ids = [
            body.full_tech_first.target_id,
            body.full_tech_second.target_id,
        ]

    # Build action input
    action_input = ActionExecutionInput(
        actor_id=current_actor.id,
        action_id=body.action_id,
        action_type=body.action_type,
        target_ids=action_target_ids,
        target_position=target_position,
        weapon_id=body.weapon_id,
        weapon_profile_id=body.weapon_profile_id,
        system_id=body.system_id,
        full_tech_first=body.full_tech_first,
        full_tech_second=body.full_tech_second,
        movement_path=movement_path,
        prompt_dangerous_terrain=body.prompt_dangerous_terrain,
        is_overcharge=body.is_overcharge,
        use_thrown=body.use_thrown,
        target_mount_id=body.target_mount_id,
        target_deployable_id=body.target_deployable_id,
    )

    # Execute action using core helper
    updated_scenario, updated_turn, updated_economy, action_result = execute_action(
        scenario, current_turn, economy, action_input
    )

    if not action_result.success:
        return ActionResponse(
            success=False,
            error=action_result.error,
            effects_applied=[],
            damage_dealt=0,
            heat_generated=0,
            economy=economy.model_dump(),
            scenario=combat_session.scenario,
        )

    # Update turn in round
    updated_rounds = list(updated_scenario.rounds)
    if round_idx < len(updated_rounds):
        round_turns = list(updated_rounds[round_idx].turns)
        if turn_idx < len(round_turns):
            round_turns[turn_idx] = updated_turn
        else:
            round_turns.append(updated_turn)
        updated_rounds[round_idx] = CombatRound(
            round_index=updated_rounds[round_idx].round_index,
            turns=round_turns,
            reaction_counts_by_actor=dict(updated_rounds[round_idx].reaction_counts_by_actor),
        )

    final_scenario = MechCombatScenario(
        combatants=list(updated_scenario.combatants),
        grapples=list(updated_scenario.grapples),
        rounds=updated_rounds,
        terrain=updated_scenario.terrain,
        environment=updated_scenario.environment,
        deployables=dict(updated_scenario.deployables),
    )

    # Persist
    combat_session.scenario = final_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Update economy cache
    _set_session_economy(session_id, updated_economy)

    # Broadcast state update to all connected WebSocket clients
    ws_response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": ws_response.model_dump(mode="json")},
    )

    return ActionResponse(
        success=True,
        action_use=action_result.action_use.model_dump() if action_result.action_use else None,
        effects_applied=action_result.effects_applied,
        damage_dealt=action_result.damage_dealt,
        heat_generated=action_result.heat_generated,
        economy=updated_economy.model_dump(),
        scenario=combat_session.scenario,
    )


@router.post("/{session_id}/turns/end", response_model=TurnEndResponse)
async def end_combat_turn(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> TurnEndResponse:
    """Finalize the current turn and advance to the next actor.

    Applies end-of-turn effects, decrements cooldowns,
    and advances the turn index (or round if needed).
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot end turn: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Get current turn
    round_idx = combat_session.current_round - 1
    turn_idx = combat_session.current_turn_index

    if round_idx >= len(scenario.rounds):
        raise ValidationError(
            "Invalid round",
            errors=[{"loc": ["round"], "msg": "Round not found", "type": "value_error"}],
        )

    current_round_data = scenario.rounds[round_idx]
    if turn_idx >= len(current_round_data.turns):
        raise ValidationError(
            "Invalid turn index",
            errors=[{"loc": ["turn"], "msg": "Turn not found", "type": "value_error"}],
        )

    current_turn = current_round_data.turns[turn_idx]

    # End turn using core helper
    updated_scenario, turn_end_result, new_round, new_turn_idx = end_turn(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
        current_turn,
    )

    # Persist
    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.current_round = new_round
    combat_session.current_turn_index = new_turn_idx
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Clear economy for this session (next turn will reset it)
    _clear_session_economy(session_id)

    # Broadcast state update to all connected WebSocket clients
    ws_response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": ws_response.model_dump(mode="json")},
    )

    return TurnEndResponse(
        actor_id=turn_end_result.actor_id,
        next_actor_id=turn_end_result.next_actor_id,
        next_actor_name=turn_end_result.next_actor_name,
        round_advanced=turn_end_result.round_advanced,
        new_round_number=turn_end_result.new_round_number,
        end_of_turn_effects=turn_end_result.end_of_turn_effects,
        scenario=combat_session.scenario,
    )


@router.post("/{session_id}/reactions", response_model=ReactionResponse)
async def submit_reaction(
    session_id: str,
    body: ReactionRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> ReactionResponse:
    """Declare a reaction during another combatant's turn.

    Brace grants resistance to the triggering attack.
    Overwatch allows a skirmish attack against a moving enemy.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot react: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Build reaction input
    reaction_input = ReactionInput(
        reactor_id=body.reactor_id,
        reaction_type=body.reaction_type,
        trigger_action_id=body.trigger_action_id,
        target_ids=body.target_ids,
        weapon_id=body.weapon_id,
    )

    # Get economy for the reactor (may be different from current actor)
    reactor_economy = ActionEconomyState()  # Reactions have their own per-round tracking

    # Execute reaction using core helper
    updated_scenario, updated_economy, reaction_result = execute_reaction(
        scenario, reactor_economy, reaction_input
    )

    if not reaction_result.success:
        return ReactionResponse(
            success=False,
            error=reaction_result.error,
            effects_applied=[],
            damage_dealt=0,
            scenario=combat_session.scenario,
        )

    # Persist
    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Broadcast state update to all connected WebSocket clients
    ws_response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": ws_response.model_dump(mode="json")},
    )

    return ReactionResponse(
        success=True,
        reaction_used=reaction_result.reaction_used,
        effects_applied=reaction_result.effects_applied,
        damage_dealt=reaction_result.damage_dealt,
        scenario=combat_session.scenario,
    )


@router.get("/{session_id}/available-actions", response_model=AvailableActionsResponse)
async def get_combat_available_actions(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> AvailableActionsResponse:
    """List valid actions for the current actor given economy and status.

    Returns categorized lists of full, quick, free actions, reactions,
    and protocols with availability status.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Get current actor
    current_actor = get_current_actor(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    if current_actor is None:
        raise ValidationError(
            "No current actor found",
            errors=[{"loc": ["turn"], "msg": "Turn not started", "type": "value_error"}],
        )

    # Get current economy
    economy = _get_session_economy(session_id)

    # Get available actions using core helper
    available = get_available_actions(scenario, current_actor.id, economy)

    def to_item(a) -> AvailableActionItem:
        return AvailableActionItem(
            action_id=a.action_id,
            action_name=a.action_name,
            action_type=a.action_type,
            is_available=a.is_available,
            unavailable_reason=a.unavailable_reason,
            requires_target=a.requires_target,
            requires_weapon=a.requires_weapon,
            requires_system=a.requires_system,
            requires_path=a.requires_path,
            max_targets=a.max_targets,
        )

    # Get overcharge level from combatant state
    overcharge_level = 0
    if current_actor.overcharge_state:
        overcharge_level = current_actor.overcharge_state.level

    return AvailableActionsResponse(
        actor_id=available.actor_id,
        economy=available.economy.model_dump(),
        full_actions=[to_item(a) for a in available.full_actions],
        quick_actions=[to_item(a) for a in available.quick_actions],
        free_actions=[to_item(a) for a in available.free_actions],
        reactions=[to_item(a) for a in available.reactions],
        protocols=[to_item(a) for a in available.protocols],
        can_overcharge=available.can_overcharge,
        overcharge_level=overcharge_level,
    )


@router.get(
    "/{session_id}/reaction-opportunities/{combatant_id}",
    response_model=ReactionOpportunityResponse,
)
async def check_reaction_opportunity(
    session_id: str,
    combatant_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> ReactionOpportunityResponse:
    """Check if a combatant has pending reaction triggers.

    This endpoint is polled by clients during opponent turns to detect
    when reaction opportunities arise.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Find the combatant
    combatant = next(
        (c for c in scenario.combatants if c.id == combatant_id),
        None,
    )
    if combatant is None:
        raise NotFoundError("Combatant", combatant_id)

    # Get current actor (whose turn it is)
    current_actor = get_current_actor(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    # Check if combatant has reaction available (per-round usage tracked in combatant state)
    round_idx = combat_session.current_round - 1
    brace_used = combatant.per_round_reactions.get("brace", 0)
    overwatch_used = combatant.per_round_reactions.get("overwatch", 0)
    has_reaction = brace_used < 1 or overwatch_used < 1

    # Build pending triggers based on current turn state
    pending_triggers: list[ReactionTrigger] = []

    if has_reaction and current_actor and current_actor.id != combatant_id:
        # Check the current turn for triggering actions
        if round_idx < len(scenario.rounds):
            current_round = scenario.rounds[round_idx]
            turn_idx = combat_session.current_turn_index
            if turn_idx < len(current_round.turns):
                current_turn = current_round.turns[turn_idx]

                # Check each action in the current turn for triggers
                for action in current_turn.actions or []:
                    # Attack targeting this combatant
                    target_ids = action.target_ids or []
                    if action.target_id:
                        target_ids = [action.target_id] + list(target_ids)

                    if combatant_id in target_ids:
                        # This is an attack targeting us - brace opportunity
                        pending_triggers.append(
                            ReactionTrigger(
                                trigger_type="attack_incoming",
                                triggering_actor_id=current_actor.id,
                                triggering_actor_name=current_actor.name,
                                triggering_action_id=action.action_id,
                                available_reactions=["brace"],
                            )
                        )

                # Check for movement-based overwatch triggers
                if current_turn.movement_path:
                    disengage_active = any(
                        a.action_id == "disengage" for a in current_turn.actions or []
                    )
                    overwatch_result = check_overwatch_triggers_for_movement(
                        scenario=scenario,
                        mover=current_actor,
                        movement_path=current_turn.movement_path,
                        is_disengaging=disengage_active,
                        is_hidden="hidden" in current_actor.statuses,
                        is_invisible="invisible" in current_actor.statuses,
                    )
                    for opp in overwatch_result.opportunities:
                        if opp.reactor_id != combatant_id:
                            continue
                        if not opp.can_react:
                            continue
                        pending_triggers.append(
                            ReactionTrigger(
                                trigger_type="enemy_movement",
                                triggering_actor_id=current_actor.id,
                                triggering_actor_name=current_actor.name,
                                triggering_action_id=None,
                                available_reactions=["overwatch"],
                            )
                        )
                        break

    return ReactionOpportunityResponse(
        combatant_id=combatant_id,
        combatant_name=combatant.name,
        has_reaction_available=has_reaction,
        pending_triggers=pending_triggers,
    )


# =============================================================================
# WebSocket Endpoint for Real-time Updates
# =============================================================================


@router.websocket("/{session_id}/ws")
async def combat_websocket(
    websocket: WebSocket,
    session_id: str,
    session: AsyncSession = Depends(get_session),
) -> None:
    """WebSocket endpoint for real-time combat state updates.

    Connects to a combat session and receives state updates whenever
    any player executes an action. Also supports ping/pong for keep-alive.

    Message protocol:
    - Server -> Client: {"type": "state", "data": CombatSessionResponse}
    - Server -> Client: {"type": "pong"}
    - Client -> Server: {"type": "ping"}
    """
    # Verify session exists
    result = await session.exec(
        select(CombatSessionDB).where(CombatSessionDB.id == session_id)
    )
    combat_session = result.first()

    if not combat_session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await combat_ws_manager.connect(session_id, websocket)
    try:
        # Send initial state
        response = _session_to_response(combat_session)
        await websocket.send_json({
            "type": "state",
            "data": response.model_dump(mode="json"),
        })

        # Keep connection alive and handle client messages
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        combat_ws_manager.disconnect(session_id, websocket)
    except Exception:
        # Handle any other exceptions by cleaning up
        combat_ws_manager.disconnect(session_id, websocket)


# =============================================================================
# Pending Decision Endpoints
# =============================================================================


@router.get(
    "/{session_id}/pending-decisions/{combatant_id}",
    response_model=PendingDecisionsResponse,
)
async def get_pending_decisions(
    session_id: str,
    combatant_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> PendingDecisionsResponse:
    """Get pending decisions for a specific combatant.

    This endpoint is polled by clients to detect when save prompts
    or system trauma selections are needed.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Find the combatant
    combatant = next(
        (c for c in scenario.combatants if c.id == combatant_id),
        None,
    )
    if combatant is None:
        raise NotFoundError("Combatant", combatant_id)

    # Get pending decisions from scenario
    pending = get_pending_decisions_for_combatant(scenario, combatant_id)

    # Convert to response items
    items = [
        PendingDecisionItem(
            decision_id=d.decision_id,
            decision_type=d.decision_type,
            trigger_source=d.trigger_source,
            trigger_round=d.trigger_round,
            save_type=d.save_type,
            save_target=d.save_target,
            save_bonus=d.save_bonus,
            trauma_target=d.trauma_target,
            eligible_mounts=d.eligible_mounts,
            eligible_systems=d.eligible_systems,
            reroll_available=d.reroll_available,
            reroll_source=d.reroll_source,
        )
        for d in pending
    ]

    return PendingDecisionsResponse(
        combatant_id=combatant_id,
        combatant_name=combatant.name,
        pending_decisions=items,
        has_pending=len(items) > 0,
    )


@router.post("/{session_id}/decisions", response_model=DecisionResultResponse)
async def submit_decision(
    session_id: str,
    body: DecisionSubmitRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> DecisionResultResponse:
    """Submit a decision resolution for a save or system trauma.

    Handles hull saves, engineering saves/checks, and system trauma selections.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot submit decision: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Find the decision
    decision = next(
        (d for d in scenario.pending_decisions if d.decision_id == body.decision_id),
        None,
    )
    if decision is None:
        raise NotFoundError("Decision", body.decision_id)

    # Verify combatant matches
    if decision.combatant_id != body.combatant_id:
        raise ValidationError(
            f"Decision belongs to combatant '{decision.combatant_id}', not '{body.combatant_id}'",
            errors=[{"loc": ["combatant_id"], "msg": "Combatant mismatch", "type": "value_error"}],
        )

    # Find the combatant
    combatant = next(
        (c for c in scenario.combatants if c.id == body.combatant_id),
        None,
    )
    if combatant is None:
        raise NotFoundError("Combatant", body.combatant_id)

    # Build resolution
    resolution = DecisionResolution(
        choice=body.choice,
        selected_mount_index=body.selected_mount_index,
        selected_system_id=body.selected_system_id,
    )

    effects_applied: list[dict[str, Any]] = []
    roll_result: int | None = None
    save_succeeded: bool | None = None
    updated_scenario = scenario

    def _replace_combatant(
        scenario_state: MechCombatScenario,
        updated_combatant: CombatantState,
    ) -> MechCombatScenario:
        combatants = list(scenario_state.combatants)
        for idx, existing in enumerate(combatants):
            if existing.id == updated_combatant.id:
                combatants[idx] = updated_combatant
                break
        return scenario_state.model_copy(update={"combatants": combatants})

    # Handle save decisions
    if decision.decision_type in ("hull_save", "engineering_save", "engineering_check"):
        save_result = resolve_save_decision(
            decision,
            resolution,
            target_conditions=list(combatant.conditions) if combatant.conditions else [],
        )
        roll_result = save_result.save_result.roll if save_result.save_result else None
        save_succeeded = save_result.success

        if save_result.voluntarily_failed:
            effects_applied.append({
                "type": "voluntary_fail",
                "decision_type": decision.decision_type,
            })
        elif save_result.save_result:
            effects_applied.append({
                "type": "save_roll",
                "roll": save_result.save_result.roll,
                "total": save_result.save_result.total,
                "target": save_result.save_result.target,
                "success": save_result.success,
                "degree": save_result.save_result.degree,
            })

        # Apply failure effects based on decision type
        if not save_result.success:
            if decision.decision_type == "hull_save":
                # Hull save failure at 2 structure = mech destroyed
                updated_combatant = apply_failed_hull_save(combatant)
                updated_scenario = _replace_combatant(updated_scenario, updated_combatant)
                effects_applied.append({
                    "type": "mech_destroyed",
                    "reason": "hull_save_failed",
                })
            elif decision.decision_type == "engineering_save":
                # Engineering save failure = meltdown countdown starts
                effects_applied.append({
                    "type": "meltdown_countdown",
                    "reason": "engineering_save_failed",
                })
            elif decision.decision_type == "engineering_check":
                # Engineering check failure (dangerous terrain) = take damage
                terrain_rules = DEFAULT_MECH_COMBAT_RULES.terrain
                damage_amount = terrain_rules.dangerous_terrain_damage
                updated_scenario, _change, _structure_result = apply_damage(
                    updated_scenario,
                    combatant.id,
                    damage_amount,
                    armor_piercing=0,
                )
                effects_applied.append({
                    "type": "damage",
                    "target_id": combatant.id,
                    "amount": damage_amount,
                    "source": "dangerous_terrain",
                })
                effects_applied.append({
                    "type": "terrain_damage",
                    "reason": "engineering_check_failed",
                    "trigger_source": decision.trigger_source,
                    "damage": damage_amount,
                    "damage_type": str(terrain_rules.dangerous_terrain_damage_type),
                })

        if decision.decision_type == "engineering_check":
            refreshed = next(
                (c for c in updated_scenario.combatants if c.id == combatant.id),
                combatant,
            )
            refreshed = refreshed.model_copy(
                update={"dangerous_terrain_last_check_round": decision.trigger_round}
            )
            updated_scenario = _replace_combatant(updated_scenario, refreshed)

    # Handle system trauma decisions
    elif decision.decision_type == "system_trauma":
        trauma_result = resolve_trauma_decision(decision, resolution)

        if not trauma_result.valid_selection:
            return DecisionResultResponse(
                success=False,
                error=trauma_result.error_message,
                effects_applied=[],
                scenario=combat_session.scenario,
            )

        if trauma_result.selected_target == "mount":
            updated_combatant = apply_system_trauma_selection(combatant, trauma_result)
            updated_scenario = _replace_combatant(updated_scenario, updated_combatant)
            effects_applied.append({
                "type": "mount_destroyed",
                "mount_index": trauma_result.mount_index,
            })
        else:
            updated_combatant = apply_system_trauma_selection(combatant, trauma_result)
            updated_scenario = _replace_combatant(updated_scenario, updated_combatant)
            effects_applied.append({
                "type": "system_destroyed",
                "system_id": trauma_result.system_id,
            })

    # Remove the resolved decision from scenario
    updated_scenario = remove_decision_from_scenario(updated_scenario, body.decision_id)

    # Persist
    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Broadcast state update to all connected WebSocket clients
    ws_response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": ws_response.model_dump(mode="json")},
    )

    return DecisionResultResponse(
        success=True,
        roll_result=roll_result,
        save_succeeded=save_succeeded,
        effects_applied=effects_applied,
        scenario=combat_session.scenario,
    )


# =============================================================================
# Reserve Spending Endpoint
# =============================================================================


@router.post("/{session_id}/reserves/spend", response_model=CombatSessionResponse)
async def spend_reserve(
    session_id: str,
    body: SpendReserveRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> CombatSessionResponse:
    """Spend a mission reserve during combat.

    Updates the reserve status from 'planned' to 'spent' and broadcasts
    the updated state to all connected clients.
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot spend reserve: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Find the reserve
    reserve_idx = None
    for idx, reserve in enumerate(scenario.mission_reserves):
        if reserve.reserve_id == body.reserve_id:
            reserve_idx = idx
            break

    if reserve_idx is None:
        raise NotFoundError("Reserve", body.reserve_id)

    reserve = scenario.mission_reserves[reserve_idx]

    if reserve.status != "planned":
        raise ValidationError(
            f"Reserve '{body.reserve_id}' has already been spent or earned",
            errors=[{"loc": ["reserve_id"], "msg": f"Reserve status is '{reserve.status}'", "type": "value_error"}],
        )

    # Update reserve status to 'spent'
    updated_reserve = reserve.model_copy(update={"status": "spent"})
    updated_reserves = list(scenario.mission_reserves)
    updated_reserves[reserve_idx] = updated_reserve

    # Create updated scenario with spent reserve
    updated_scenario = scenario.model_copy(update={"mission_reserves": updated_reserves})

    # Persist
    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Broadcast state update to all connected WebSocket clients
    ws_response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": ws_response.model_dump(mode="json")},
    )

    return _session_to_response(combat_session)


# =============================================================================
# Auto NPC Turn Endpoint
# =============================================================================


class AutoNPCTurnResponse(BaseModel):
    """Response model for automated NPC turn execution."""

    success: bool
    actor_id: str
    actor_name: str
    decision_action: str | None = None
    decision_target: str | None = None
    decision_reasoning: str | None = None
    actions_taken: int
    skipped: bool
    skip_reason: str | None = None
    scenario: dict[str, Any]


@router.post("/{session_id}/turns/auto-npc", response_model=AutoNPCTurnResponse)
async def execute_auto_npc_turn(
    session_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(get_current_user),
) -> AutoNPCTurnResponse:
    """Execute an automated NPC turn.

    This endpoint runs a full NPC turn cycle:
    1. Starts the turn
    2. Uses AI to select an action and target
    3. Executes the selected action
    4. Ends the turn

    The current actor must be AI-controlled (ai_controlled=True).
    """
    result = await session.exec(
        select(CombatSessionDB).where(
            CombatSessionDB.id == session_id,
            CombatSessionDB.gm_user_id == user["id"],
        )
    )
    combat_session = result.first()

    if not combat_session:
        raise NotFoundError("Combat session", session_id)

    if combat_session.status != "active":
        raise ValidationError(
            f"Cannot execute NPC turn: session status is '{combat_session.status}'",
            errors=[{"loc": ["status"], "msg": "Session must be active", "type": "value_error"}],
        )

    # Hydrate scenario
    scenario = MechCombatScenario.model_validate(combat_session.scenario)

    # Get current actor from turn order
    current_actor = get_current_actor(
        scenario,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    if current_actor is None:
        raise ValidationError(
            "No current actor found",
            errors=[{"loc": ["turn"], "msg": "Turn order not initialized", "type": "value_error"}],
        )

    # Validate current actor is AI-controlled
    if not current_actor.ai_controlled:
        raise ValidationError(
            f"Current actor '{current_actor.name}' is not AI-controlled",
            errors=[{"loc": ["actor"], "msg": "Actor must be ai_controlled=True", "type": "value_error"}],
        )

    # Execute the NPC turn
    updated_scenario, npc_result = execute_npc_turn(
        scenario,
        current_actor.id,
        combat_session.current_round,
        combat_session.current_turn_index,
    )

    # Update round/turn index based on turn end result
    new_round = combat_session.current_round
    new_turn_index = combat_session.current_turn_index
    if npc_result.turn_end:
        if npc_result.turn_end.round_advanced and npc_result.turn_end.new_round_number:
            new_round = npc_result.turn_end.new_round_number
            new_turn_index = 0
        elif npc_result.turn_end.next_actor_id:
            # Find next turn index
            if new_round - 1 < len(updated_scenario.rounds):
                round_data = updated_scenario.rounds[new_round - 1]
                for idx, turn in enumerate(round_data.turns):
                    if turn.actor_id == npc_result.turn_end.next_actor_id:
                        new_turn_index = idx
                        break
                else:
                    new_turn_index = combat_session.current_turn_index + 1

    # Persist
    combat_session.scenario = updated_scenario.model_dump(mode="json")
    combat_session.current_round = new_round
    combat_session.current_turn_index = new_turn_index
    combat_session.updated_at = utc_now()

    session.add(combat_session)
    await session.commit()
    await session.refresh(combat_session)

    # Clear economy cache (turn ended)
    _clear_session_economy(session_id)

    # Broadcast state update to all connected WebSocket clients
    ws_response = _session_to_response(combat_session)
    await combat_ws_manager.broadcast(
        session_id,
        {"type": "state", "data": ws_response.model_dump(mode="json")},
    )

    return AutoNPCTurnResponse(
        success=not npc_result.skipped,
        actor_id=npc_result.actor_id,
        actor_name=npc_result.actor_name,
        decision_action=npc_result.decision.action if npc_result.decision else None,
        decision_target=npc_result.decision.target_id if npc_result.decision else None,
        decision_reasoning=npc_result.decision.reasoning if npc_result.decision else None,
        actions_taken=npc_result.actions_taken,
        skipped=npc_result.skipped,
        skip_reason=npc_result.skip_reason,
        scenario=combat_session.scenario,
    )
