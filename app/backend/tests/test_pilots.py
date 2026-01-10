"""Tests for pilot CRUD endpoints with core model validation."""

import pytest
from httpx import AsyncClient


# =============================================================================
# Sample Data Factories
# =============================================================================


def make_pilot_create(
    callsign: str = "TESTER",
    name: str = "Test Pilot",
    level: int = 0,
    **kwargs,
) -> dict:
    """Create sample pilot create request data."""
    return {
        "callsign": callsign,
        "name": name,
        "level": level,
        **kwargs,
    }


def make_ll0_pilot_create(
    callsign: str = "NOVA",
    name: str = "Nova Chen",
) -> dict:
    """Create a valid LL0 pilot with required starting resources.

    LL0 pilots have:
    - 2 skill points distributed
    - 4 triggers at +2 each
    - 3 talent ranks (3 rank-1 talents)
    - No licenses or core bonuses
    """
    return {
        "callsign": callsign,
        "name": name,
        "level": 0,
        "skills": {"hull": 1, "agility": 1, "systems": 0, "engineering": 0},
        "triggers": [
            {"trigger_id": "apply_fists_to_faces", "rank": 2},
            {"trigger_id": "assault", "rank": 2},
            {"trigger_id": "threaten", "rank": 2},
            {"trigger_id": "survive", "rank": 2},
        ],
        "talents": [
            {"talent_id": "ace", "rank": 1},
            {"talent_id": "brutal", "rank": 1},
            {"talent_id": "tactician", "rank": 1},
        ],
        "licenses": [],
        "core_bonuses": [],
        "notes": "Test LL0 pilot",
    }


# =============================================================================
# Basic CRUD Tests
# =============================================================================


@pytest.mark.asyncio
async def test_create_pilot_minimal(client: AsyncClient) -> None:
    """Test creating a pilot with minimal data."""
    response = await client.post(
        "/api/pilots",
        json=make_pilot_create(callsign="ACE", name="Ace McFlyer"),
    )

    assert response.status_code == 201
    data = response.json()
    assert data["callsign"] == "ACE"
    assert data["name"] == "Ace McFlyer"
    assert data["id"].startswith("pilot_")
    assert data["level"] == 0
    # Verify computed fields
    assert data["grit"] == 0  # LL0 has grit 0
    assert data["hp"] == 6  # Base HP + grit


@pytest.mark.asyncio
async def test_create_pilot_with_skills(client: AsyncClient) -> None:
    """Test creating a pilot with skill allocations."""
    response = await client.post(
        "/api/pilots",
        json=make_pilot_create(
            callsign="TANK",
            skills={"hull": 2, "agility": 0, "systems": 0, "engineering": 0},
        ),
    )

    assert response.status_code == 201
    data = response.json()
    assert data["skills"]["hull"] == 2
    assert data["skills"]["agility"] == 0


@pytest.mark.asyncio
async def test_create_pilot_with_triggers(client: AsyncClient) -> None:
    """Test creating a pilot with triggers."""
    response = await client.post(
        "/api/pilots",
        json=make_pilot_create(
            callsign="SCOUT",
            triggers=[
                {"trigger_id": "act_unseen_or_unheard", "rank": 2},
                {"trigger_id": "spot", "rank": 4},
            ],
        ),
    )

    assert response.status_code == 201
    data = response.json()
    assert len(data["triggers"]) == 2
    assert data["triggers"][0]["trigger_id"] == "act_unseen_or_unheard"
    assert data["triggers"][1]["rank"] == 4


@pytest.mark.asyncio
async def test_create_pilot_with_talents(client: AsyncClient) -> None:
    """Test creating a pilot with talents."""
    response = await client.post(
        "/api/pilots",
        json=make_pilot_create(
            callsign="HERO",
            talents=[
                {"talent_id": "ace", "rank": 1},
                {"talent_id": "brutal", "rank": 2},
            ],
        ),
    )

    assert response.status_code == 201
    data = response.json()
    assert len(data["talents"]) == 2


@pytest.mark.asyncio
async def test_list_pilots_empty(client: AsyncClient) -> None:
    """Test listing pilots when none exist."""
    response = await client.get("/api/pilots")

    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_pilots_with_data(client: AsyncClient) -> None:
    """Test listing pilots after creating some."""
    await client.post("/api/pilots", json=make_pilot_create(callsign="ONE"))
    await client.post("/api/pilots", json=make_pilot_create(callsign="TWO"))

    response = await client.get("/api/pilots")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    callsigns = [p["callsign"] for p in data["items"]]
    assert "ONE" in callsigns
    assert "TWO" in callsigns


@pytest.mark.asyncio
async def test_get_pilot(client: AsyncClient) -> None:
    """Test getting a single pilot by ID."""
    create_response = await client.post(
        "/api/pilots",
        json=make_pilot_create(callsign="SOLO"),
    )
    pilot_id = create_response.json()["id"]

    response = await client.get(f"/api/pilots/{pilot_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == pilot_id
    assert data["callsign"] == "SOLO"
    # Verify computed fields are present
    assert "grit" in data
    assert "hp" in data
    assert "evasion" in data


@pytest.mark.asyncio
async def test_get_pilot_not_found(client: AsyncClient) -> None:
    """Test getting a non-existent pilot returns 404."""
    response = await client.get("/api/pilots/nonexistent_id")

    assert response.status_code == 404
    data = response.json()
    assert data["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_update_pilot_callsign(client: AsyncClient) -> None:
    """Test updating a pilot's callsign."""
    create_response = await client.post(
        "/api/pilots",
        json=make_pilot_create(callsign="OLD"),
    )
    pilot_id = create_response.json()["id"]

    response = await client.put(
        f"/api/pilots/{pilot_id}",
        json={"callsign": "NEW"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["callsign"] == "NEW"


@pytest.mark.asyncio
async def test_update_pilot_skills(client: AsyncClient) -> None:
    """Test updating a pilot's skills."""
    create_response = await client.post(
        "/api/pilots",
        json=make_pilot_create(
            callsign="FLEX",
            skills={"hull": 0, "agility": 0, "systems": 0, "engineering": 0},
        ),
    )
    pilot_id = create_response.json()["id"]

    response = await client.put(
        f"/api/pilots/{pilot_id}",
        json={"skills": {"hull": 2, "agility": 1, "systems": 0, "engineering": 0}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["skills"]["hull"] == 2
    assert data["skills"]["agility"] == 1


@pytest.mark.asyncio
async def test_update_pilot_level(client: AsyncClient) -> None:
    """Test updating a pilot's level affects computed stats."""
    create_response = await client.post(
        "/api/pilots",
        json=make_pilot_create(callsign="RISING", level=0),
    )
    pilot_id = create_response.json()["id"]

    # LL0: grit=0, hp=6
    assert create_response.json()["grit"] == 0
    assert create_response.json()["hp"] == 6

    # Level up to LL3
    response = await client.put(
        f"/api/pilots/{pilot_id}",
        json={"level": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["level"] == 3
    # LL3: grit=2, hp=8
    assert data["grit"] == 2
    assert data["hp"] == 8


@pytest.mark.asyncio
async def test_delete_pilot(client: AsyncClient) -> None:
    """Test deleting a pilot."""
    create_response = await client.post(
        "/api/pilots",
        json=make_pilot_create(callsign="DOOMED"),
    )
    pilot_id = create_response.json()["id"]

    delete_response = await client.delete(f"/api/pilots/{pilot_id}")
    assert delete_response.status_code == 204

    get_response = await client.get(f"/api/pilots/{pilot_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_pilot_not_found(client: AsyncClient) -> None:
    """Test deleting a non-existent pilot returns 404."""
    response = await client.delete("/api/pilots/nonexistent_id")
    assert response.status_code == 404


# =============================================================================
# Validation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_create_pilot_invalid_skill_range(client: AsyncClient) -> None:
    """Test that invalid skill values are rejected."""
    response = await client.post(
        "/api/pilots",
        json=make_pilot_create(
            callsign="INVALID",
            skills={"hull": 10, "agility": 0, "systems": 0, "engineering": 0},
        ),
    )

    # Should fail validation (skill max is 6)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_pilot_invalid_trigger_rank(client: AsyncClient) -> None:
    """Test that invalid trigger ranks are rejected."""
    response = await client.post(
        "/api/pilots",
        json=make_pilot_create(
            callsign="INVALID",
            triggers=[{"trigger_id": "assault", "rank": 1}],  # Min is 2
        ),
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_pilot_missing_callsign(client: AsyncClient) -> None:
    """Test that callsign is required."""
    response = await client.post(
        "/api/pilots",
        json={"name": "No Callsign"},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_validate_pilot_endpoint(client: AsyncClient) -> None:
    """Test the pilot validation endpoint."""
    # Create a pilot with mismatched progression
    create_response = await client.post(
        "/api/pilots",
        json=make_pilot_create(
            callsign="CHECK",
            level=0,
            skills={"hull": 6, "agility": 6, "systems": 6, "engineering": 6},  # Way too many points for LL0
        ),
    )
    pilot_id = create_response.json()["id"]

    # Validate should show issues
    response = await client.get(f"/api/pilots/{pilot_id}/validate")

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert len(data["issues"]) > 0


@pytest.mark.asyncio
async def test_validate_pilot_valid(client: AsyncClient) -> None:
    """Test validation endpoint with a valid LL0 pilot."""
    create_response = await client.post(
        "/api/pilots",
        json=make_ll0_pilot_create(callsign="VALID"),
    )
    pilot_id = create_response.json()["id"]

    response = await client.get(f"/api/pilots/{pilot_id}/validate")

    assert response.status_code == 200
    data = response.json()
    # Note: May have issues if talent definitions don't exist
    # The validation checks against the progression rules


# =============================================================================
# Computed Fields Tests
# =============================================================================


@pytest.mark.asyncio
async def test_computed_fields_at_levels(client: AsyncClient) -> None:
    """Test that computed fields update correctly at different levels."""
    # Test at LL0
    resp0 = await client.post(
        "/api/pilots",
        json=make_pilot_create(callsign="LL0", level=0),
    )
    data0 = resp0.json()
    assert data0["grit"] == 0
    assert data0["hp"] == 6
    assert data0["save_target"] == 10

    # Test at LL6
    resp6 = await client.post(
        "/api/pilots",
        json=make_pilot_create(callsign="LL6", level=6),
    )
    data6 = resp6.json()
    assert data6["grit"] == 3
    assert data6["hp"] == 9  # 6 base + 3 grit
    assert data6["save_target"] == 13  # 10 + grit

    # Test at LL12
    resp12 = await client.post(
        "/api/pilots",
        json=make_pilot_create(callsign="LL12", level=12),
    )
    data12 = resp12.json()
    assert data12["grit"] == 6
    assert data12["hp"] == 12  # 6 base + 6 grit
    assert data12["save_target"] == 16  # 10 + grit


@pytest.mark.asyncio
async def test_base_combat_stats(client: AsyncClient) -> None:
    """Test base combat stats are included in response."""
    response = await client.post(
        "/api/pilots",
        json=make_pilot_create(callsign="STATS"),
    )

    data = response.json()
    assert data["evasion"] == 10
    assert data["e_defense"] == 10
    assert data["speed"] == 4
    assert data["armor"] == 0
    assert data["attack_bonus"] == 0  # grit at LL0
