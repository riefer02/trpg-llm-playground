"""Tests for compendium API endpoints."""

import pytest
from httpx import AsyncClient


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
