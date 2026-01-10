"""Custom exception classes for consistent error handling.

All API errors inherit from AppError to ensure consistent JSON responses.
"""

from typing import Any


class AppError(Exception):
    """Base exception for application errors.
    
    All custom exceptions should inherit from this class to ensure
    consistent error response format across the API.
    """

    def __init__(
        self,
        detail: str,
        code: str = "APP_ERROR",
        status_code: int = 500,
        errors: list[dict[str, Any]] | None = None,
    ):
        self.detail = detail
        self.code = code
        self.status_code = status_code
        self.errors = errors
        super().__init__(detail)


class NotFoundError(AppError):
    """Resource not found."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            detail=f"{resource} with id '{identifier}' not found",
            code="NOT_FOUND",
            status_code=404,
        )


class ValidationError(AppError):
    """Validation failed."""

    def __init__(self, detail: str, errors: list[dict[str, Any]] | None = None):
        super().__init__(
            detail=detail,
            code="VALIDATION_ERROR",
            status_code=422,
            errors=errors,
        )


class ConflictError(AppError):
    """Resource conflict (e.g., duplicate, concurrent modification)."""

    def __init__(self, detail: str):
        super().__init__(
            detail=detail,
            code="CONFLICT",
            status_code=409,
        )


class UnauthorizedError(AppError):
    """Authentication required."""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            detail=detail,
            code="UNAUTHORIZED",
            status_code=401,
        )


class ForbiddenError(AppError):
    """Access denied."""

    def __init__(self, detail: str = "Access denied"):
        super().__init__(
            detail=detail,
            code="FORBIDDEN",
            status_code=403,
        )
