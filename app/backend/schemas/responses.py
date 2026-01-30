"""Shared response schema primitives.

These are the building blocks for consistent API responses across all endpoints.
"""

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# =============================================================================
# Database Metadata
# =============================================================================


class DatabaseMetadata(BaseModel):
    """Common database record metadata.

    Use this as a mixin or base for response models that include DB records.

    Example:
        class CharacterResponse(DatabaseMetadata):
            callsign: str
            ...
    """

    id: str = Field(..., description="Unique identifier")
    user_id: str = Field(..., description="Owner user ID")
    campaign_id: str | None = Field(default=None, description="Associated campaign")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# =============================================================================
# List Responses
# =============================================================================


class ListResponse(BaseModel, Generic[T]):
    """Generic list response with metadata.

    Provides consistent structure for all list endpoints.

    Example:
        @router.get("", response_model=ListResponse[CharacterSummary])
        async def list_characters(...):
            return ListResponse(
                items=characters,
                total=len(characters),
            )
    """

    items: list[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total count of items")
    limit: int | None = Field(default=None, description="Max items returned")
    offset: int | None = Field(default=None, description="Items skipped")

    @property
    def has_more(self) -> bool:
        """Check if there are more items beyond this page."""
        if self.limit is None or self.offset is None:
            return False
        return self.offset + len(self.items) < self.total


# =============================================================================
# Validation Responses
# =============================================================================


class ValidationIssue(BaseModel):
    """A single validation issue with severity.

    Used for game rule validation (not HTTP validation errors).
    HTTP validation errors use the exception system.

    Example:
        ValidationIssue(
            severity="error",
            field="skills",
            message="Too many skill points at LL0",
            code="INVALID_SKILL_POINTS",
        )
    """

    severity: str = Field(
        ...,
        description="Issue severity: 'error', 'warning', or 'info'",
    )
    field: str | None = Field(
        default=None,
        description="Field path where issue occurred",
    )
    message: str = Field(..., description="Human-readable description")
    code: str | None = Field(
        default=None,
        description="Machine-readable error code",
    )


class ValidationResponse(BaseModel):
    """Validation result for game rule checking.

    Returned by validation endpoints to help users fix issues.

    Example:
        @router.get("/{id}/validate", response_model=ValidationResponse)
        async def validate_character(...):
            result = validate_character(character)
            return ValidationResponse(
                valid=result.is_valid,
                issues=[...],
            )
    """

    valid: bool = Field(..., description="Whether all checks passed")
    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="List of validation issues",
    )

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == "warning"]


# =============================================================================
# Error Response (for OpenAPI docs)
# =============================================================================


class ErrorDetail(BaseModel):
    """Structured error detail for validation errors."""

    loc: list[str | int] = Field(..., description="Location of the error")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ErrorResponse(BaseModel):
    """Standard error response format.

    Used for OpenAPI documentation of error responses.
    Actual errors are raised via exceptions.py.
    """

    detail: str = Field(..., description="Error description")
    code: str = Field(..., description="Error code (e.g., 'NOT_FOUND', 'VALIDATION_ERROR')")
    errors: list[ErrorDetail] | None = Field(
        default=None,
        description="Detailed validation errors (for 422 responses)",
    )
