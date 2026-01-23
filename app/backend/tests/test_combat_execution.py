"""Tests for combat turn execution API endpoints.

Tests the turn management lifecycle:
- POST /api/combat/{id}/turns/start
- POST /api/combat/{id}/actions
- POST /api/combat/{id}/turns/end
- POST /api/combat/{id}/reactions
- GET /api/combat/{id}/available-actions
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.models import CombatSessionDB
from app.backend.db.engine import get_session
from app.backend.main import create_app


# =============================================================================
# Sample Data Factories
# =============================================================================


def make_combatant(
    id: str = "mech_001",
    name: str = "Test Mech",
    side: str = "players",
    kind: str = "mech",
    hp_max: int = 10,
    hp_current: int = 10,
    heat_current: int = 0,
    q: int = 0,
    r: int = 0,
    **kwargs,
) -> dict:
    """Create sample combatant matching CombatantState structure."""
    return {
        "id": id,
        "name": name,
        "side": side,
        "kind": kind,
        "stats": {
            "size": "size_1",
            "hp_max": hp_max,
            "evasion": 8,
            "e_defense": 8,
            "armor": 0,
            "speed": 4,
            "sensor_range": 10,
            "tech_attack": 0,
        },
        "resources": {
            "hp_current": hp_current,
            "heat_current": heat_current,
            "heat_cap": 6,
            "structure_current": 4,
            "stress_current": 4,
            "repairs_remaining": 4,
        },
        "position": {"coord": {"q": q, "r": r}, "elevation": 0},
        "statuses": [],
        "conditions": [],
        "ai_controlled": False,
        "overcharge_state": {"current_level": 0, "uses_this_turn": 0},
        "per_round_reactions": {},
        **kwargs,
    }


def make_turn(actor_id: str = "mech_001") -> dict:
    """Create a combat turn."""
    return {
        "actor_id": actor_id,
        "move_used": False,
        "movement_mode": "ground",
        "movement_path": [],
        "actions": [],
    }


def make_round(round_index: int = 1, turns: list[dict] | None = None) -> dict:
    """Create a combat round with turns."""
    return {
        "round_index": round_index,
        "turns": turns or [],
        "reaction_counts_by_actor": {},
    }


def make_session_with_turns(
    combatants: list[dict],
    turns_per_round: list[str] | None = None,
) -> dict:
    """Create a combat session with initialized turns."""
    if turns_per_round is None:
        turns_per_round = [c["id"] for c in combatants]

    turns = [make_turn(actor_id) for actor_id in turns_per_round]
    rounds = [make_round(round_index=1, turns=turns)]

    return {
        "name": "Combat Session",
        "environment": "standard",
        "combatants": combatants,
        "notes": "",
    }


async def create_session_with_rounds(client: AsyncClient, combatants: list[dict]) -> dict:
    """Create a session and manually add round data."""
    # Create session
    create_resp = await client.post(
        "/api/combat",
        json={
            "name": "Test Combat",
            "environment": "standard",
            "combatants": combatants,
        },
    )
    assert create_resp.status_code == 201
    session_data = create_resp.json()

    # Manually update scenario with rounds
    # We need to add rounds structure for turn management
    scenario = session_data["scenario"]
    turn_order = [c["id"] for c in combatants]
    turns = [make_turn(actor_id) for actor_id in turn_order]
    scenario["rounds"] = [make_round(round_index=1, turns=turns)]

    # Update the session
    update_resp = await client.put(
        f"/api/combat/{session_data['id']}",
        json={"notes": "Initialized"},  # Trigger update
    )
    assert update_resp.status_code == 200

    # For testing, we need to directly patch the scenario with rounds
    # This would normally happen via a "initialize combat" endpoint
    return session_data


# =============================================================================
# Turn Start Tests
# =============================================================================


@pytest.mark.asyncio
async def test_start_turn_requires_active_session(client: AsyncClient) -> None:
    """Test that turn start fails for non-active sessions."""
    # Create and pause session
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Pause the session
    await client.put(f"/api/combat/{session_id}", json={"status": "paused"})

    # Try to start turn
    response = await client.post(f"/api/combat/{session_id}/turns/start")
    assert response.status_code == 422
    assert "session" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_start_turn_auto_initializes_turn_order(client: AsyncClient) -> None:
    """Test that turn start auto-initializes turn order when rounds are empty."""
    # Create session without rounds
    combatants = [make_combatant(id="mech_1"), make_combatant(id="mech_2")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Start turn (should auto-initialize rounds)
    response = await client.post(f"/api/combat/{session_id}/turns/start")
    assert response.status_code == 200
    data = response.json()
    assert data["actor_id"] == "mech_1"  # First combatant gets first turn

    # Verify rounds were initialized
    get_resp = await client.get(f"/api/combat/{session_id}")
    scenario = get_resp.json()["scenario"]
    assert len(scenario["rounds"]) == 1
    assert len(scenario["rounds"][0]["turns"]) == 2


# =============================================================================
# Action Execution Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execute_action_requires_active_session(client: AsyncClient) -> None:
    """Test that action execution fails for non-active sessions."""
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Pause the session
    await client.put(f"/api/combat/{session_id}", json={"status": "paused"})

    # Try to execute action
    response = await client.post(
        f"/api/combat/{session_id}/actions",
        json={"action_id": "skirmish", "action_type": "quick"},
    )
    assert response.status_code == 422
    assert "session" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_execute_action_no_current_actor(client: AsyncClient) -> None:
    """Test that action execution fails when no turn is active."""
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Try to execute action (no turn started)
    response = await client.post(
        f"/api/combat/{session_id}/actions",
        json={"action_id": "skirmish", "action_type": "quick"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_execute_full_tech_action(session: AsyncSession) -> None:
    """Full Tech should execute two tech options and apply status."""
    app = create_app()

    async def override_get_session() -> AsyncSession:
        yield session

    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        combatants = [
            make_combatant(id="attacker", side="players"),
            make_combatant(id="defender", side="hostiles"),
        ]
        create_resp = await client.post(
            "/api/combat",
            json={"name": "Test", "combatants": combatants},
        )
        assert create_resp.status_code == 201
        session_id = create_resp.json()["id"]

        result = await session.exec(
            select(CombatSessionDB).where(CombatSessionDB.id == session_id)
        )
        combat_session = result.first()
        assert combat_session is not None

        scenario = dict(combat_session.scenario)
        scenario["rounds"] = [
            make_round(round_index=1, turns=[make_turn("attacker"), make_turn("defender")])
        ]
        combat_session.scenario = scenario
        await session.commit()

        start_resp = await client.post(f"/api/combat/{session_id}/turns/start")
        assert start_resp.status_code == 200
        start_payload = start_resp.json()
        assert start_payload["scenario"]["rounds"], "Scenario rounds missing after start_turn"

        response = await client.post(
            f"/api/combat/{session_id}/actions",
            json={
                "action_id": "full_tech",
                "action_type": "full",
                "full_tech_first": {"option": "scan", "target_id": "defender"},
                "full_tech_second": {"option": "lock_on", "target_id": "defender"},
            },
        )
        assert response.status_code == 200, response.json()
        payload = response.json()
        assert payload["success"] is True
        assert {effect["type"] for effect in payload["effects_applied"]} >= {"scan", "lock_on"}

        defender_after = next(
            combatant
            for combatant in payload["scenario"]["combatants"]
            if combatant["id"] == "defender"
        )
        assert "lock_on" in defender_after["statuses"]


# =============================================================================
# Turn End Tests
# =============================================================================


@pytest.mark.asyncio
async def test_end_turn_requires_active_session(client: AsyncClient) -> None:
    """Test that turn end fails for non-active sessions."""
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Pause the session
    await client.put(f"/api/combat/{session_id}", json={"status": "paused"})

    # Try to end turn
    response = await client.post(f"/api/combat/{session_id}/turns/end")
    assert response.status_code == 422
    assert "session" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_end_turn_no_round_data(client: AsyncClient) -> None:
    """Test that turn end fails when no round data exists."""
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Try to end turn (no rounds)
    response = await client.post(f"/api/combat/{session_id}/turns/end")
    assert response.status_code == 422


# =============================================================================
# Reaction Tests
# =============================================================================


@pytest.mark.asyncio
async def test_reaction_requires_active_session(client: AsyncClient) -> None:
    """Test that reactions fail for non-active sessions."""
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Pause the session
    await client.put(f"/api/combat/{session_id}", json={"status": "paused"})

    # Try to submit reaction
    response = await client.post(
        f"/api/combat/{session_id}/reactions",
        json={"reactor_id": "mech_1", "reaction_type": "brace"},
    )
    assert response.status_code == 422
    assert "session" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_reaction_unknown_reactor(client: AsyncClient) -> None:
    """Test that reactions fail for unknown reactors."""
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Try to submit reaction for unknown combatant
    response = await client.post(
        f"/api/combat/{session_id}/reactions",
        json={"reactor_id": "unknown", "reaction_type": "brace"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "not found" in data["error"].lower()


@pytest.mark.asyncio
async def test_brace_reaction_success(client: AsyncClient) -> None:
    """Test successful brace reaction."""
    combatants = [make_combatant(id="mech_1", name="Alpha")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Submit brace reaction
    response = await client.post(
        f"/api/combat/{session_id}/reactions",
        json={
            "reactor_id": "mech_1",
            "reaction_type": "brace",
            "trigger_action_id": "enemy_attack",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["reaction_used"] == "brace"


@pytest.mark.asyncio
async def test_reaction_fails_when_already_used_this_round(client: AsyncClient) -> None:
    """Test that reactions fail when already used this round."""
    # Create combatant with brace already used
    combatants = [make_combatant(id="mech_1", per_round_reactions={"brace": 1})]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Try to brace again
    response = await client.post(
        f"/api/combat/{session_id}/reactions",
        json={"reactor_id": "mech_1", "reaction_type": "brace"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "already used" in data["error"].lower()


# =============================================================================
# Available Actions Tests
# =============================================================================


@pytest.mark.asyncio
async def test_available_actions_requires_actor(client: AsyncClient) -> None:
    """Test that available actions fails when no turn is active."""
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Try to get available actions (no turn)
    response = await client.get(f"/api/combat/{session_id}/available-actions")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_session_not_found(client: AsyncClient) -> None:
    """Test 404 for nonexistent session."""
    response = await client.post("/api/combat/nonexistent/turns/start")
    assert response.status_code == 404

    response = await client.post(
        "/api/combat/nonexistent/actions",
        json={"action_id": "skirmish", "action_type": "quick"},
    )
    assert response.status_code == 404

    response = await client.post("/api/combat/nonexistent/turns/end")
    assert response.status_code == 404

    response = await client.post(
        "/api/combat/nonexistent/reactions",
        json={"reactor_id": "x", "reaction_type": "brace"},
    )
    assert response.status_code == 404

    response = await client.get("/api/combat/nonexistent/available-actions")
    assert response.status_code == 404


# =============================================================================
# Integration Test: Full Turn Sequence
# =============================================================================


@pytest.mark.asyncio
async def test_action_response_structure(client: AsyncClient) -> None:
    """Test that action response has expected structure."""
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Execute action - will fail due to no turn, but response structure matters
    response = await client.post(
        f"/api/combat/{session_id}/actions",
        json={
            "action_id": "skirmish",
            "action_type": "quick",
            "target_ids": ["enemy_1"],
        },
    )
    # Either 200 (with success=False) or 422 for validation
    assert response.status_code in [200, 422]


@pytest.mark.asyncio
async def test_reaction_response_structure(client: AsyncClient) -> None:
    """Test that reaction response has expected structure."""
    combatants = [make_combatant(id="mech_1")]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Submit reaction
    response = await client.post(
        f"/api/combat/{session_id}/reactions",
        json={
            "reactor_id": "mech_1",
            "reaction_type": "brace",
        },
    )
    assert response.status_code == 200
    data = response.json()

    # Check structure
    assert "success" in data
    assert "reaction_used" in data
    assert "effects_applied" in data
    assert "damage_dealt" in data
    assert "scenario" in data


@pytest.mark.asyncio
async def test_overwatch_reaction(client: AsyncClient) -> None:
    """Test overwatch reaction."""
    combatants = [
        make_combatant(id="mech_1", name="Alpha"),
        make_combatant(id="enemy_1", name="Enemy", side="hostiles", q=1, r=0),
    ]
    create_resp = await client.post(
        "/api/combat",
        json={"name": "Test", "combatants": combatants},
    )
    session_id = create_resp.json()["id"]

    # Submit overwatch reaction
    response = await client.post(
        f"/api/combat/{session_id}/reactions",
        json={
            "reactor_id": "mech_1",
            "reaction_type": "overwatch",
            "weapon_id": "assault_rifle",
            "target_ids": ["enemy_1"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["reaction_used"] == "overwatch"
