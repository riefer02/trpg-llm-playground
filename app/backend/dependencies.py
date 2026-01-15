"""FastAPI dependency injection providers.

This module contains all shared dependencies used across API endpoints.
Using dependency injection makes it easy to swap implementations
(e.g., stub auth -> JWT auth) without changing endpoint code.
"""

from typing import Any

from fastapi import Request


def _stub_user() -> dict[str, Any]:
    return {
        "id": "user_dev_001",
        "email": "dev@lancer.local",
        "name": "Development User",
        "roles": ["user", "gm"],
    }


async def get_current_user(request: Request) -> dict[str, Any]:
    """Get the current authenticated user.

    STUB IMPLEMENTATION: Returns a mock user for development.
    Request headers can override the default user to facilitate testing:
    - `X-User-Id`: overrides the user id
    - `X-User-Name`: overrides the display name
    - `X-User-Email`: overrides the email
    - `X-User-Roles`: comma-separated roles

    Replace with JWT validation for production.
    """
    user = _stub_user()

    headers = request.headers

    if user_id := headers.get("x-user-id"):
        user["id"] = user_id
    if user_name := headers.get("x-user-name"):
        user["name"] = user_name
    if user_email := headers.get("x-user-email"):
        user["email"] = user_email
    if roles_header := headers.get("x-user-roles"):
        user["roles"] = [
            role.strip() for role in roles_header.split(",") if role.strip()
        ]

    return user


async def require_gm(user: dict[str, Any] | None = None) -> dict[str, Any]:
    """Require GM role for access.

    STUB IMPLEMENTATION: Always passes for development.

    Usage:
        @router.post("/campaigns")
        async def create_campaign(user: dict = Depends(require_gm)):
            ...
    """
    if user is None:
        user = _stub_user()

    # In production, check: if "gm" not in user.get("roles", []):
    #     raise ForbiddenError("GM role required")

    return user
