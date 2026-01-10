"""FastAPI dependency injection providers.

This module contains all shared dependencies used across API endpoints.
Using dependency injection makes it easy to swap implementations
(e.g., stub auth -> JWT auth) without changing endpoint code.
"""

from typing import Any


async def get_current_user() -> dict[str, Any]:
    """Get the current authenticated user.
    
    STUB IMPLEMENTATION: Returns a mock user for development.
    Replace with JWT validation for production.
    
    Usage:
        @router.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user_id": user["id"]}
    
    Future implementation will:
    1. Extract JWT from Authorization header
    2. Validate token signature and expiration
    3. Return user claims from token payload
    """
    return {
        "id": "user_dev_001",
        "email": "dev@lancer.local",
        "name": "Development User",
        "roles": ["user", "gm"],
    }


async def require_gm(user: dict[str, Any] = None) -> dict[str, Any]:
    """Require GM role for access.
    
    STUB IMPLEMENTATION: Always passes for development.
    
    Usage:
        @router.post("/campaigns")
        async def create_campaign(user: dict = Depends(require_gm)):
            ...
    """
    if user is None:
        user = await get_current_user()
    
    # In production, check: if "gm" not in user.get("roles", []):
    #     raise ForbiddenError("GM role required")
    
    return user
