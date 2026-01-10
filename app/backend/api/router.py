"""Main API router that aggregates all route modules."""

from fastapi import APIRouter

from app.backend.api.health import router as health_router
from app.backend.api.pilots import router as pilots_router
from app.backend.api.characters import router as characters_router
from app.backend.api.combat import router as combat_router
from app.backend.api.compendium import router as compendium_router

api_router = APIRouter()

# Include sub-routers
api_router.include_router(health_router)
api_router.include_router(compendium_router)  # Reference data
api_router.include_router(characters_router)  # Primary user-facing
api_router.include_router(pilots_router)  # Internal primitive
api_router.include_router(combat_router)
