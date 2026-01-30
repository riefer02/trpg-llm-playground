"""Database engine and session management.

Provides async database connections using SQLModel with asyncpg.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.config import settings

# Global engine reference
_engine: AsyncEngine | None = None
_async_session_factory: sessionmaker | None = None


async def init_db() -> None:
    """Initialize database connection.
    
    Called during application startup via lifespan context manager.
    Tables are created via Alembic migrations (run `make db-migrate`).
    """
    global _engine, _async_session_factory

    _engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        future=True,
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
    )

    _async_session_factory = sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def close_db() -> None:
    """Close database connections.
    
    Called during application shutdown via lifespan context manager.
    """
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions.
    
    Usage:
        @router.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    if not _async_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
