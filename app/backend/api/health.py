"""Health check endpoints.

Provides endpoints for monitoring application and database health.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from app.backend.db.engine import get_session

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"


class DatabaseHealthResponse(BaseModel):
    """Database health check response."""

    status: str
    database: str


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check.
    
    Returns 200 if the application is running.
    Used by load balancers and container orchestration.
    """
    return HealthResponse(status="healthy")


@router.get("/db", response_model=DatabaseHealthResponse)
async def database_health_check(
    session: AsyncSession = Depends(get_session),
) -> DatabaseHealthResponse:
    """Database connectivity check.
    
    Executes a simple query to verify database is accessible.
    Returns 200 if database connection succeeds, 500 otherwise.
    """
    try:
        await session.exec(text("SELECT 1"))
        return DatabaseHealthResponse(status="healthy", database="connected")
    except Exception as e:
        return DatabaseHealthResponse(status="unhealthy", database=str(e))
