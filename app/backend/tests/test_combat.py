"""Tests for combat session CRUD endpoints.

These tests use the same dict structure as core.mech.combat_state models.
No duplicate schemas - core handles all validation.
"""

import pytest
from httpx import AsyncClient


# =============================================================================
# Sample Data Factories - Match core model structure exactly
# =============================================================================


def make_combatant(
    id: str = "mech_001",
    name: str = "Test Mech",
    side: str = "players",
    kind: str = "mech",
    hp_max: int = 10,
    hp_current: int = 10,
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
            "heat_current": 0,
            "heat_cap": 6,
            "structure_current": 4,
            "stress_current": 4,
            "repairs_remaining": 4,
        },
        "position": {"coord": {"q": 0, "r": 0}, "elevation": 0},
        "statuses": [],
        "conditions": [],
        "ai_controlled": False,
        **kwargs,
    }


def make_session_create(
    name: str = "Test Combat",
    environment: str = "standard",
    combatants: list | None = None,
    **kwargs,
) -> dict:
    """Create sample combat session create request."""
    return {
        "name": name,
        "environment": environment,
        "combatants": combatants or [],
        "notes": "",
        **kwargs,
    }


# =============================================================================
# Basic CRUD Tests
# =============================================================================


@pytest.mark.asyncio
async def test_create_combat_session_minimal(client: AsyncClient) -> None:
    """Test creating a combat session with minimal data."""
    response = await client.post(
        "/api/combat",
        json=make_session_create(name="Empty Battle"),
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Empty Battle"
    assert data["id"].startswith("combat_")
    assert data["status"] == "active"
    assert data["current_round"] == 1
    assert data["scenario"]["environment"] == "standard"
    assert len(data["scenario"]["combatants"]) == 0


@pytest.mark.asyncio
async def test_create_combat_session_with_combatants(client: AsyncClient) -> None:
    """Test creating a combat session with combatants."""
    combatants = [
        make_combatant(id="player_1", name="Alpha", side="players"),
        make_combatant(id="npc_1", name="Enemy Unit", side="hostiles", kind="npc"),
    ]

    response = await client.post(
        "/api/combat",
        json=make_session_create(name="Skirmish", combatants=combatants),
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Skirmish"
    assert len(data["scenario"]["combatants"]) == 2
    assert data["scenario"]["combatants"][0]["name"] == "Alpha"
    assert data["scenario"]["combatants"][1]["side"] == "hostiles"


@pytest.mark.asyncio
async def test_list_combat_sessions(client: AsyncClient) -> None:
    """Test listing combat sessions."""
    await client.post("/api/combat", json=make_session_create(name="Battle 1"))
    await client.post("/api/combat", json=make_session_create(name="Battle 2"))

    response = await client.get("/api/combat")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


@pytest.mark.asyncio
async def test_list_combat_sessions_filter_by_status(client: AsyncClient) -> None:
    """Test filtering sessions by status."""
    create_resp = await client.post(
        "/api/combat", json=make_session_create(name="Completed Battle")
    )
    session_id = create_resp.json()["id"]
    await client.put(f"/api/combat/{session_id}", json={"status": "completed"})

    await client.post("/api/combat", json=make_session_create(name="Active Battle"))

    response = await client.get("/api/combat", params={"status_filter": "completed"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_get_combat_session(client: AsyncClient) -> None:
    """Test getting a combat session by ID."""
    combatants = [make_combatant(id="mech_1", name="Test Mech")]
    create_resp = await client.post(
        "/api/combat",
        json=make_session_create(name="Solo Battle", combatants=combatants),
    )
    session_id = create_resp.json()["id"]

    response = await client.get(f"/api/combat/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session_id
    assert data["name"] == "Solo Battle"
    assert len(data["scenario"]["combatants"]) == 1
    assert data["scenario"]["combatants"][0]["stats"]["hp_max"] == 10


@pytest.mark.asyncio
async def test_get_combat_session_not_found(client: AsyncClient) -> None:
    """Test getting a nonexistent session returns 404."""
    response = await client.get("/api/combat/combat_nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_combat_session(client: AsyncClient) -> None:
    """Test updating a combat session."""
    create_resp = await client.post(
        "/api/combat", json=make_session_create(name="Original Name")
    )
    session_id = create_resp.json()["id"]

    response = await client.put(
        f"/api/combat/{session_id}",
        json={"name": "Updated Name", "status": "paused", "notes": "Taking a break"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Name"
    assert data["status"] == "paused"
    assert data["notes"] == "Taking a break"


@pytest.mark.asyncio
async def test_delete_combat_session(client: AsyncClient) -> None:
    """Test deleting a combat session."""
    create_resp = await client.post(
        "/api/combat", json=make_session_create(name="To Delete")
    )
    session_id = create_resp.json()["id"]

    response = await client.delete(f"/api/combat/{session_id}")
    assert response.status_code == 204

    get_resp = await client.get(f"/api/combat/{session_id}")
    assert get_resp.status_code == 404


# =============================================================================
# Combatant Management Tests
# =============================================================================


@pytest.mark.asyncio
async def test_add_combatant(client: AsyncClient) -> None:
    """Test adding a combatant to a session."""
    create_resp = await client.post(
        "/api/combat", json=make_session_create(name="Growing Battle")
    )
    session_id = create_resp.json()["id"]

    response = await client.post(
        f"/api/combat/{session_id}/combatants",
        json={"combatant": make_combatant(id="new_mech", name="Reinforcement")},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["scenario"]["combatants"]) == 1
    assert data["scenario"]["combatants"][0]["name"] == "Reinforcement"


@pytest.mark.asyncio
async def test_add_duplicate_combatant_fails(client: AsyncClient) -> None:
    """Test adding a combatant with duplicate ID fails."""
    combatants = [make_combatant(id="mech_1", name="Original")]
    create_resp = await client.post(
        "/api/combat", json=make_session_create(combatants=combatants)
    )
    session_id = create_resp.json()["id"]

    response = await client.post(
        f"/api/combat/{session_id}/combatants",
        json={"combatant": make_combatant(id="mech_1", name="Duplicate")},
    )

    assert response.status_code == 422
    data = response.json()
    assert "Duplicate" in data["detail"] or "already exists" in data["detail"]


@pytest.mark.asyncio
async def test_remove_combatant(client: AsyncClient) -> None:
    """Test removing a combatant from a session."""
    combatants = [
        make_combatant(id="mech_1", name="Keep"),
        make_combatant(id="mech_2", name="Remove"),
    ]
    create_resp = await client.post(
        "/api/combat", json=make_session_create(combatants=combatants)
    )
    session_id = create_resp.json()["id"]

    response = await client.delete(f"/api/combat/{session_id}/combatants/mech_2")

    assert response.status_code == 200
    data = response.json()
    assert len(data["scenario"]["combatants"]) == 1
    assert data["scenario"]["combatants"][0]["id"] == "mech_1"


@pytest.mark.asyncio
async def test_remove_nonexistent_combatant(client: AsyncClient) -> None:
    """Test removing a nonexistent combatant returns 404."""
    create_resp = await client.post("/api/combat", json=make_session_create())
    session_id = create_resp.json()["id"]

    response = await client.delete(f"/api/combat/{session_id}/combatants/ghost")
    assert response.status_code == 404


# =============================================================================
# Core Model Validation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_combatant_stats_from_core(client: AsyncClient) -> None:
    """Test that combatant stats match core model structure."""
    combatant = make_combatant(hp_max=50, hp_current=25)
    create_resp = await client.post(
        "/api/combat",
        json=make_session_create(combatants=[combatant]),
    )

    assert create_resp.status_code == 201
    c = create_resp.json()["scenario"]["combatants"][0]
    assert c["stats"]["hp_max"] == 50
    assert c["resources"]["hp_current"] == 25


@pytest.mark.asyncio
async def test_combatant_position_from_core(client: AsyncClient) -> None:
    """Test that position uses core HexPosition structure."""
    combatant = make_combatant(position={"coord": {"q": 3, "r": -2}, "elevation": 2})
    create_resp = await client.post(
        "/api/combat",
        json=make_session_create(combatants=[combatant]),
    )

    assert create_resp.status_code == 201
    pos = create_resp.json()["scenario"]["combatants"][0]["position"]
    assert pos["coord"]["q"] == 3
    assert pos["coord"]["r"] == -2
    assert pos["elevation"] == 2


@pytest.mark.asyncio
async def test_different_environments(client: AsyncClient) -> None:
    """Test creating sessions with different environments."""
    for env in ["standard", "zero_g", "underwater"]:
        response = await client.post(
            "/api/combat",
            json=make_session_create(name=f"Battle in {env}", environment=env),
        )
        assert response.status_code == 201
        assert response.json()["scenario"]["environment"] == env


@pytest.mark.asyncio
async def test_combatant_sides(client: AsyncClient) -> None:
    """Test all combatant sides work correctly."""
    combatants = [
        make_combatant(id="p1", side="players"),
        make_combatant(id="h1", side="hostiles"),
        make_combatant(id="n1", side="neutral"),
    ]

    response = await client.post(
        "/api/combat",
        json=make_session_create(combatants=combatants),
    )

    assert response.status_code == 201
    sides = {c["id"]: c["side"] for c in response.json()["scenario"]["combatants"]}
    assert sides["p1"] == "players"
    assert sides["h1"] == "hostiles"
    assert sides["n1"] == "neutral"


@pytest.mark.asyncio
async def test_invalid_combatant_rejected(client: AsyncClient) -> None:
    """Test that invalid combatant data is rejected by core validation."""
    # Missing required fields
    invalid_combatant = {"id": "bad", "name": "Bad Mech"}  # Missing stats, resources, etc.
    
    response = await client.post(
        "/api/combat",
        json=make_session_create(combatants=[invalid_combatant]),
    )

    assert response.status_code == 422
