"""Tests for compendium API endpoints."""

import pytest
from httpx import AsyncClient

from core.mech.compendium import ALL_FRAMES, ALL_WEAPONS, ALL_SYSTEMS
from core.pilot.gear import PILOT_GEAR_DEFINITIONS

@pytest.mark.asyncio
async def test_list_backgrounds(client: AsyncClient):
    """Test listing all backgrounds."""
    response = await client.get("/api/compendium/backgrounds")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] == 20  # 20 backgrounds

    # Check structure of first item
    bg = data["items"][0]
    assert "id" in bg
    assert "name" in bg
    assert "triggers" in bg
    assert len(bg["triggers"]) == 4  # Each background has 4 triggers


@pytest.mark.asyncio
async def test_list_triggers(client: AsyncClient):
    """Test listing all triggers."""
    response = await client.get("/api/compendium/triggers")
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == 22  # 22 triggers

    # Check structure
    trigger = data["items"][0]
    assert "id" in trigger
    assert "name" in trigger


@pytest.mark.asyncio
async def test_list_talents(client: AsyncClient):
    """Test listing all talents."""
    response = await client.get("/api/compendium/talents")
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == 34  # 34 talents

    # Check structure
    talent = data["items"][0]
    assert "id" in talent
    assert "name" in talent
    assert "ranks" in talent
    assert talent["ranks"] == 3


@pytest.mark.asyncio
async def test_background_triggers_are_valid(client: AsyncClient):
    """Verify all background trigger references are valid trigger IDs."""
    bgs_response = await client.get("/api/compendium/backgrounds")
    triggers_response = await client.get("/api/compendium/triggers")

    backgrounds = bgs_response.json()["items"]
    valid_trigger_ids = {t["id"] for t in triggers_response.json()["items"]}

    for bg in backgrounds:
        for trigger_id in bg["triggers"]:
            assert trigger_id in valid_trigger_ids, (
                f"Background '{bg['name']}' references invalid trigger: {trigger_id}"
            )


@pytest.mark.asyncio
async def test_list_frames(client: AsyncClient):
    """Test listing all mech frames."""
    response = await client.get("/api/compendium/frames")
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == len(ALL_FRAMES)

    frame = data["items"][0]
    assert "id" in frame
    assert "name" in frame
    assert "manufacturer" in frame
    assert "license_id" in frame
    assert "license_rank" in frame


@pytest.mark.asyncio
async def test_list_weapons(client: AsyncClient):
    """Test listing all mech weapons."""
    response = await client.get("/api/compendium/weapons")
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == len(ALL_WEAPONS)

    weapon = data["items"][0]
    assert "id" in weapon
    assert "name" in weapon
    assert "license_id" in weapon
    assert "license_rank" in weapon


@pytest.mark.asyncio
async def test_list_systems(client: AsyncClient):
    """Test listing all mech systems."""
    response = await client.get("/api/compendium/systems")
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == len(ALL_SYSTEMS)

    system = data["items"][0]
    assert "id" in system
    assert "name" in system
    assert "license_id" in system
    assert "license_rank" in system


@pytest.mark.asyncio
async def test_list_pilot_gear(client: AsyncClient):
    """Test listing all pilot gear."""
    response = await client.get("/api/compendium/pilot-gear")
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == len(PILOT_GEAR_DEFINITIONS)

    gear = data["items"][0]
    assert "id" in gear
    assert "name" in gear
    assert "category" in gear
