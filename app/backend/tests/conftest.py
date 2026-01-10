"""Pytest configuration and fixtures for backend tests.

Uses in-memory SQLite for fast tests without Docker dependency.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.main import create_app
from app.backend.db.engine import get_session
from app.backend.db.models import PilotDB, MechDB, CampaignDB, CombatSessionDB, CharacterDB  # noqa: F401


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create in-memory SQLite engine for testing.
    
    SQLite is used for tests because:
    1. No Docker/PostgreSQL setup needed
    2. Fast (in-memory)
    3. Good enough for testing ORM code
    
    Note: Some PostgreSQL-specific features won't work in tests.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture
async def session(async_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Get test database session."""
    async_session_factory = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(async_engine: AsyncEngine) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client with database override.
    
    Usage:
        async def test_health(client: AsyncClient):
            response = await client.get("/api/health")
            assert response.status_code == 200
    """
    app = create_app()
    
    # Override database session dependency
    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async_session_factory = sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session_factory() as session:
            yield session
    
    app.dependency_overrides[get_session] = override_get_session
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


# ============================================================================
# Sample Data Factories
# ============================================================================


def make_mech_data(
    name: str = "Test Mech",
    frame_id: str = "mf_standard_pattern_i_everest",
    **kwargs,
) -> dict:
    """Create sample mech data for tests."""
    return {
        "name": name,
        "frame_id": frame_id,
        "data": {
            "name": name,
            "frame_id": frame_id,
            **kwargs,
        },
    }
