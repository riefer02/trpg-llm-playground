"""Tests for health check endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """Test basic health check returns 200."""
    response = await client.get("/api/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_database_health_check(client: AsyncClient) -> None:
    """Test database health check with in-memory SQLite."""
    response = await client.get("/api/health/db")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
