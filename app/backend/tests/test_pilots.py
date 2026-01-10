"""Tests for pilot CRUD endpoints."""

import pytest
from httpx import AsyncClient

from app.backend.tests.conftest import make_pilot_data


@pytest.mark.asyncio
async def test_create_pilot(client: AsyncClient) -> None:
    """Test creating a new pilot."""
    pilot_data = make_pilot_data(name="Ace McFlyer", callsign="ACE")
    
    response = await client.post("/api/pilots", json=pilot_data)
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Ace McFlyer"
    assert data["data"]["callsign"] == "ACE"
    assert data["id"].startswith("pilot_")


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
    # Create two pilots
    await client.post("/api/pilots", json=make_pilot_data(name="Pilot One"))
    await client.post("/api/pilots", json=make_pilot_data(name="Pilot Two"))
    
    response = await client.get("/api/pilots")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    names = [p["name"] for p in data["items"]]
    assert "Pilot One" in names
    assert "Pilot Two" in names


@pytest.mark.asyncio
async def test_get_pilot(client: AsyncClient) -> None:
    """Test getting a single pilot by ID."""
    # Create a pilot
    create_response = await client.post(
        "/api/pilots",
        json=make_pilot_data(name="Solo Pilot"),
    )
    pilot_id = create_response.json()["id"]
    
    # Get the pilot
    response = await client.get(f"/api/pilots/{pilot_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == pilot_id
    assert data["name"] == "Solo Pilot"


@pytest.mark.asyncio
async def test_get_pilot_not_found(client: AsyncClient) -> None:
    """Test getting a non-existent pilot returns 404."""
    response = await client.get("/api/pilots/nonexistent_id")
    
    assert response.status_code == 404
    data = response.json()
    assert data["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_update_pilot(client: AsyncClient) -> None:
    """Test updating a pilot."""
    # Create a pilot
    create_response = await client.post(
        "/api/pilots",
        json=make_pilot_data(name="Original Name"),
    )
    pilot_id = create_response.json()["id"]
    
    # Update the pilot
    response = await client.put(
        f"/api/pilots/{pilot_id}",
        json={"name": "Updated Name", "callsign": "NEW"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Name"
    assert data["data"]["callsign"] == "NEW"


@pytest.mark.asyncio
async def test_delete_pilot(client: AsyncClient) -> None:
    """Test deleting a pilot."""
    # Create a pilot
    create_response = await client.post(
        "/api/pilots",
        json=make_pilot_data(name="Doomed Pilot"),
    )
    pilot_id = create_response.json()["id"]
    
    # Delete the pilot
    delete_response = await client.delete(f"/api/pilots/{pilot_id}")
    assert delete_response.status_code == 204
    
    # Verify it's gone
    get_response = await client.get(f"/api/pilots/{pilot_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_pilot_not_found(client: AsyncClient) -> None:
    """Test deleting a non-existent pilot returns 404."""
    response = await client.delete("/api/pilots/nonexistent_id")
    
    assert response.status_code == 404
