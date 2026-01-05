"""Shared validation primitives for typed Lancer mechanics."""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from pydantic import Field
from core.shared.models import FrozenModel

if TYPE_CHECKING:
    from typing import TypeVar

    V = TypeVar("V", bound="ValidationIssue")


class ValidationIssue(FrozenModel):
    """A validation issue with a code, message, and severity level."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"

    def with_severity(self: V, severity: Literal["error", "warning"]) -> V:
        """Return a copy with a different severity."""
        return self.model_copy(update={"severity": severity})


class ValidationResult(FrozenModel):
    """A validation result containing issues and an overall validity flag."""

    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    @classmethod
    def from_issues(cls, issues: list[ValidationIssue]) -> ValidationResult:
        """Create a ValidationResult from a list of issues."""
        return cls(valid=not any(i.severity == "error" for i in issues), issues=issues)

    def filter_errors(self) -> list[ValidationIssue]:
        """Return only error-level issues."""
        return [i for i in self.issues if i.severity == "error"]

    def filter_warnings(self) -> list[ValidationIssue]:
        """Return only warning-level issues."""
        return [i for i in self.issues if i.severity == "warning"]

    def append(self, issue: ValidationIssue) -> ValidationResult:
        """Return a new ValidationResult with an additional issue."""
        return self.model_copy(update={"issues": self.issues + [issue]})

    def extend(self, issues: list[ValidationIssue]) -> ValidationResult:
        """Return a new ValidationResult with additional issues."""
        return self.model_copy(update={"issues": self.issues + issues})
