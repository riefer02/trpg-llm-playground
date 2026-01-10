"""Main API router that aggregates all route modules."""

from fastapi import APIRouter

from app.backend.api.health import router as health_router
from app.backend.api.pilots import router as pilots_router
from app.backend.api.characters import router as characters_router
from app.backend.api.combat import router as combat_router

api_router = APIRouter()

# Include sub-routers
api_router.include_router(health_router)
api_router.include_router(pilots_router)
api_router.include_router(characters_router)
api_router.include_router(combat_router)
