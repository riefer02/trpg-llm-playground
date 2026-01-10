"""Shared utilities for the backend API layer.

These helpers bridge the gap between HTTP requests and the core type system.
The core handles all validation - these just convert between formats.

Design Principle:
    Core is the source of truth. API layer is a thin wrapper.
    Never duplicate validation logic that core already provides.
"""

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from app.backend.exceptions import ValidationError

T = TypeVar("T", bound=BaseModel)


def validate_core_model(model_class: type[T], data: dict[str, Any], context: str) -> T:
    """Validate data against a core Pydantic model.

    This is the standard pattern for API endpoints that accept JSON data
    and need to validate it against core models. Core handles all the
    game-rule validation; we just convert the error format for HTTP.

    Args:
        model_class: The core Pydantic model class (e.g., SkillSet, Pilot)
        data: Raw dict from request JSON
        context: Human-readable context for error messages (e.g., "skills", "pilot")

    Returns:
        Validated instance of model_class

    Raises:
        ValidationError: If validation fails, with structured error details

    Example:
        ```python
        from app.backend.utils import validate_core_model
        from core.pilot import SkillSet

        # In an endpoint:
        skills = validate_core_model(SkillSet, request.skills, "skills")
        ```
    """
    try:
        return model_class.model_validate(data)
    except PydanticValidationError as e:
        raise ValidationError(
            f"Invalid {context}",
            errors=_format_validation_errors(e),
        )


def validate_core_model_list(
    model_class: type[T],
    data_list: list[dict[str, Any]],
    context: str,
) -> list[T]:
    """Validate a list of items against a core Pydantic model.

    Args:
        model_class: The core Pydantic model class
        data_list: List of raw dicts from request JSON
        context: Singular context name (e.g., "trigger", "talent")

    Returns:
        List of validated model instances

    Raises:
        ValidationError: If any item fails validation
    """
    return [validate_core_model(model_class, item, context) for item in data_list]


def _format_validation_errors(e: PydanticValidationError) -> list[dict[str, Any]]:
    """Convert Pydantic validation errors to API error format.

    This maintains the structured error information that Pydantic provides
    while formatting it for consistent HTTP responses.
    """
    return [
        {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
        for err in e.errors()
    ]


def core_validation_error_to_api(e: PydanticValidationError, context: str) -> ValidationError:
    """Convert a Pydantic ValidationError to an API ValidationError.

    Use this when you catch a validation error from core and need to
    re-raise it as an HTTP-friendly error.

    Args:
        e: The Pydantic ValidationError from core
        context: Human-readable context for the error message

    Returns:
        ValidationError suitable for HTTP response
    """
    return ValidationError(f"Invalid {context}", errors=_format_validation_errors(e))
