"""FastAPI application factory.

This module provides the main FastAPI application with proper
lifecycle management, middleware configuration, and error handling.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.backend.config import settings
from app.backend.db.engine import init_db, close_db
from app.backend.api.router import api_router
from app.backend.exceptions import AppError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown."""
    await init_db()
    yield
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Lancer Combat API",
        description="Backend API for Lancer TTRPG web application",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    app.add_exception_handler(AppError, app_error_handler)

    # Include API routes
    app.include_router(api_router, prefix="/api")

    return app


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle custom application errors with consistent JSON response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "code": exc.code,
            **({"errors": exc.errors} if exc.errors else {}),
        },
    )


# Application instance for uvicorn
app = create_app()
