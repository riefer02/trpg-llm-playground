"""Shared API schemas for consistent response formats.

This module provides reusable primitives for API responses:
- ListResponse: Paginated list with metadata
- ValidationResponse: Structured validation results
- DatabaseMetadata: Common DB fields (id, timestamps, user_id)

Design Principle:
    Use these primitives for consistent, well-documented API responses.
    Don't duplicate field definitions across endpoint-specific schemas.
"""

from app.backend.schemas.responses import (
    ListResponse,
    ValidationIssue,
    ValidationResponse,
    DatabaseMetadata,
)

__all__ = [
    "ListResponse",
    "ValidationIssue",
    "ValidationResponse",
    "DatabaseMetadata",
]
