"""Tests for character CRUD endpoints.

Characters are the unified abstraction combining Pilot + Mech(s).
The API uses create_ll0_character factory for sensible defaults.
"""

import pytest
from httpx import AsyncClient


# =============================================================================
# Sample Data Factories
# =============================================================================


def make_character_create(
    callsign: str = "TESTER",
    name: str = "Test Pilot",
    use_ll0_defaults: bool = True,
    **kwargs,
) -> dict:
    """Create sample character create request data."""
    return {
        "callsign": callsign,
        "name": name,
        "use_ll0_defaults": use_ll0_defaults,
        **kwargs,
    }


# =============================================================================
# Basic CRUD Tests
# =============================================================================


@pytest.mark.asyncio
async def test_create_character_minimal(client: AsyncClient) -> None:
    """Test creating a character with just callsign (uses LL0 defaults)."""
    response = await client.post(
        "/api/characters",
        json={"callsign": "ALPHA"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["callsign"] == "ALPHA"
    assert data["id"].startswith("char_")
    assert data["level"] == 0
    # LL0 defaults applied
    assert data["grit"] == 0
    # Should have a mech
    assert len(data["mechs"]) == 1
    assert data["active_mech_id"] is not None
    assert data["active_mech_stats"] is not None


@pytest.mark.asyncio
async def test_create_character_with_name(client: AsyncClient) -> None:
    """Test creating a character with real name."""
    response = await client.post(
        "/api/characters",
        json=make_character_create(callsign="BRAVO", name="Jane Doe"),
    )

    assert response.status_code == 201
    data = response.json()
    assert data["callsign"] == "BRAVO"
    assert data["name"] == "Jane Doe"


@pytest.mark.asyncio
async def test_create_character_with_custom_skills(client: AsyncClient) -> None:
    """Test creating a character with custom skill allocation."""
    response = await client.post(
        "/api/characters",
        json=make_character_create(
            callsign="TANK",
            skills={"hull": 2, "agility": 0, "systems": 0, "engineering": 0},
        ),
    )

    assert response.status_code == 201
    data = response.json()
    assert data["skills"]["hull"] == 2
    # Mech stats should reflect hull bonus (HP = 10 base + 4 from hull)
    assert data["active_mech_stats"]["hp"] == 14


@pytest.mark.asyncio
async def test_create_character_with_mech_name(client: AsyncClient) -> None:
    """Test creating a character with custom mech name."""
    response = await client.post(
        "/api/characters",
        json=make_character_create(
            callsign="NOVA",
            mech_name="RAIJIN",
        ),
    )

    assert response.status_code == 201
    data = response.json()
    assert data["mechs"][0]["name"] == "RAIJIN"


@pytest.mark.asyncio
async def test_list_characters_empty(client: AsyncClient) -> None:
    """Test listing characters when none exist."""
    response = await client.get("/api/characters")

    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_characters_with_data(client: AsyncClient) -> None:
    """Test listing characters after creating some."""
    await client.post("/api/characters", json={"callsign": "ONE"})
    await client.post("/api/characters", json={"callsign": "TWO"})

    response = await client.get("/api/characters")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    callsigns = [c["callsign"] for c in data["items"]]
    assert "ONE" in callsigns
    assert "TWO" in callsigns


@pytest.mark.asyncio
async def test_get_character(client: AsyncClient) -> None:
    """Test getting a single character by ID."""
    create_response = await client.post(
        "/api/characters",
        json={"callsign": "SOLO"},
    )
    char_id = create_response.json()["id"]

    response = await client.get(f"/api/characters/{char_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == char_id
    assert data["callsign"] == "SOLO"
    # Verify computed fields are present
    assert "grit" in data
    assert "pilot_hp" in data
    assert "active_mech_stats" in data


@pytest.mark.asyncio
async def test_get_character_not_found(client: AsyncClient) -> None:
    """Test getting a non-existent character returns 404."""
    response = await client.get("/api/characters/nonexistent_id")

    assert response.status_code == 404
    data = response.json()
    assert data["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_update_character_callsign(client: AsyncClient) -> None:
    """Test updating a character's callsign."""
    create_response = await client.post(
        "/api/characters",
        json={"callsign": "OLD"},
    )
    char_id = create_response.json()["id"]

    response = await client.put(
        f"/api/characters/{char_id}",
        json={"callsign": "NEW"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["callsign"] == "NEW"


@pytest.mark.asyncio
async def test_update_character_skills(client: AsyncClient) -> None:
    """Test updating a character's skills updates mech stats."""
    create_response = await client.post(
        "/api/characters",
        json={"callsign": "FLEX"},
    )
    char_id = create_response.json()["id"]
    initial_hp = create_response.json()["active_mech_stats"]["hp"]

    # Update to agility build
    response = await client.put(
        f"/api/characters/{char_id}",
        json={"skills": {"hull": 0, "agility": 2, "systems": 0, "engineering": 0}},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["skills"]["agility"] == 2
    # HP should be lower (no hull bonus), evasion should be higher
    assert data["active_mech_stats"]["hp"] < initial_hp
    assert data["active_mech_stats"]["evasion"] > 8  # Base evasion


@pytest.mark.asyncio
async def test_delete_character(client: AsyncClient) -> None:
    """Test deleting a character."""
    create_response = await client.post(
        "/api/characters",
        json={"callsign": "DOOMED"},
    )
    char_id = create_response.json()["id"]

    delete_response = await client.delete(f"/api/characters/{char_id}")
    assert delete_response.status_code == 204

    get_response = await client.get(f"/api/characters/{char_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_character_not_found(client: AsyncClient) -> None:
    """Test deleting a non-existent character returns 404."""
    response = await client.delete("/api/characters/nonexistent_id")
    assert response.status_code == 404


# =============================================================================
# Validation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_create_character_missing_callsign(client: AsyncClient) -> None:
    """Test that callsign is required."""
    response = await client.post(
        "/api/characters",
        json={"name": "No Callsign"},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_validate_character_endpoint(client: AsyncClient) -> None:
    """Test the character validation endpoint."""
    create_response = await client.post(
        "/api/characters",
        json={"callsign": "VALID"},
    )
    char_id = create_response.json()["id"]

    response = await client.get(f"/api/characters/{char_id}/validate")

    assert response.status_code == 200
    data = response.json()
    # LL0 character with defaults should be valid
    assert data["valid"] is True


@pytest.mark.asyncio
async def test_validate_character_with_invalid_skills(client: AsyncClient) -> None:
    """Test validation catches invalid skill allocation."""
    # Create character, then update with invalid skills
    create_response = await client.post(
        "/api/characters",
        json={"callsign": "INVALID"},
    )
    char_id = create_response.json()["id"]

    # Update with way too many skill points for LL0
    await client.put(
        f"/api/characters/{char_id}",
        json={"skills": {"hull": 6, "agility": 6, "systems": 0, "engineering": 0}},
    )

    response = await client.get(f"/api/characters/{char_id}/validate")

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert len(data["issues"]) > 0


# =============================================================================
# Mech Management Tests
# =============================================================================


@pytest.mark.asyncio
async def test_add_mech(client: AsyncClient) -> None:
    """Test adding a mech to a character."""
    create_response = await client.post(
        "/api/characters",
        json={"callsign": "MULTI"},
    )
    char_id = create_response.json()["id"]
    assert len(create_response.json()["mechs"]) == 1

    response = await client.post(
        f"/api/characters/{char_id}/mechs",
        json={"name": "BACKUP", "frame_id": "gms_everest"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["mechs"]) == 2
    mech_names = [m["name"] for m in data["mechs"]]
    assert "BACKUP" in mech_names


@pytest.mark.asyncio
async def test_remove_mech(client: AsyncClient) -> None:
    """Test removing a mech from a character."""
    # Create character and add second mech
    create_response = await client.post(
        "/api/characters",
        json={"callsign": "MULTI"},
    )
    char_id = create_response.json()["id"]

    add_response = await client.post(
        f"/api/characters/{char_id}/mechs",
        json={"name": "BACKUP", "frame_id": "gms_everest"},
    )
    backup_mech_id = [m for m in add_response.json()["mechs"] if m["name"] == "BACKUP"][0]["id"]

    # Remove the backup mech
    response = await client.delete(f"/api/characters/{char_id}/mechs/{backup_mech_id}")

    assert response.status_code == 200
    data = response.json()
    assert len(data["mechs"]) == 1
    assert data["mechs"][0]["name"] != "BACKUP"


@pytest.mark.asyncio
async def test_set_active_mech(client: AsyncClient) -> None:
    """Test setting the active mech."""
    # Create character and add second mech
    create_response = await client.post(
        "/api/characters",
        json={"callsign": "SWITCHER"},
    )
    char_id = create_response.json()["id"]
    first_mech_id = create_response.json()["active_mech_id"]

    add_response = await client.post(
        f"/api/characters/{char_id}/mechs",
        json={"name": "BACKUP", "frame_id": "gms_everest"},
    )
    backup_mech_id = [m for m in add_response.json()["mechs"] if m["name"] == "BACKUP"][0]["id"]

    # Switch to backup mech
    response = await client.put(f"/api/characters/{char_id}/mechs/{backup_mech_id}/activate")

    assert response.status_code == 200
    data = response.json()
    assert data["active_mech_id"] == backup_mech_id
    assert data["active_mech_id"] != first_mech_id


# =============================================================================
# Computed Fields Tests
# =============================================================================


@pytest.mark.asyncio
async def test_mech_stats_computed_from_pilot(client: AsyncClient) -> None:
    """Test that mech stats are computed from pilot skills and grit."""
    # Create LL0 character with Hull +2
    response = await client.post(
        "/api/characters",
        json=make_character_create(
            callsign="STATS",
            skills={"hull": 2, "agility": 0, "systems": 0, "engineering": 0},
        ),
    )

    assert response.status_code == 201
    data = response.json()

    # GMS Everest base: HP 10, evasion 8, speed 4, heat_cap 6
    # With Hull +2: HP += 4 (2 * 2)
    # With Grit 0 (LL0): HP += 0
    stats = data["active_mech_stats"]
    assert stats["hp"] == 14  # 10 + 4
    assert stats["evasion"] == 8  # Base, no agility
    assert stats["speed"] == 4  # Base
    assert stats["heat_cap"] == 6  # Base, no engineering


@pytest.mark.asyncio
async def test_ll0_character_has_default_triggers_talents(client: AsyncClient) -> None:
    """Test that LL0 character gets default triggers and talents."""
    response = await client.post(
        "/api/characters",
        json={"callsign": "DEFAULT"},
    )

    assert response.status_code == 201
    data = response.json()

    # LL0 should have 4 triggers at +2 each
    assert len(data["triggers"]) == 4
    assert all(t["rank"] == 2 for t in data["triggers"])

    # LL0 should have 3 rank-1 talents
    assert len(data["talents"]) == 3
    assert all(t["rank"] == 1 for t in data["talents"])
