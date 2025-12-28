"""Validation helpers for pilot progression rules."""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from pydantic import BaseModel, Field

from core.pilot.progression import get_level_progression

if TYPE_CHECKING:
    from core.pilot.pilot import Pilot


class ProgressionIssue(BaseModel):
    """A progression validation issue."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"

    model_config = {"frozen": True}


class ProgressionValidation(BaseModel):
    """Validation result for pilot progression."""

    valid: bool
    issues: list[ProgressionIssue] = Field(default_factory=list)

    model_config = {"frozen": True}


def validate_pilot_progression(pilot: Pilot) -> ProgressionValidation:
    """Validate a pilot against the license level progression table."""
    progression = get_level_progression(pilot.level)
    issues: list[ProgressionIssue] = []

    trigger_ids = [trigger.trigger_id for trigger in pilot.triggers]
    if len(set(trigger_ids)) != len(trigger_ids):
        issues.append(
            ProgressionIssue(
                code="duplicate_triggers",
                message="Trigger list contains duplicates; increase ranks instead.",
            )
        )

    for trigger in pilot.triggers:
        if trigger.rank % 2 != 0:
            issues.append(
                ProgressionIssue(
                    code="invalid_trigger_rank",
                    message=(
                        f"Trigger '{trigger.trigger_id}' rank {trigger.rank} is invalid; "
                        "trigger ranks must be in +2 increments."
                    ),
                )
            )

    if pilot.total_license_levels() > progression.license_points:
        issues.append(
            ProgressionIssue(
                code="license_points_exceeded",
                message=(
                    f"License points {pilot.total_license_levels()} exceed "
                    f"allowed {progression.license_points} for level {pilot.level}."
                ),
            )
        )

    if pilot.total_talent_ranks() > progression.total_talent_points:
        issues.append(
            ProgressionIssue(
                code="talent_points_exceeded",
                message=(
                    f"Talent points {pilot.total_talent_ranks()} exceed "
                    f"allowed {progression.total_talent_points} for level {pilot.level}."
                ),
            )
        )

    if pilot.skills.total_points() > progression.total_mech_skill_points:
        issues.append(
            ProgressionIssue(
                code="mech_skill_points_exceeded",
                message=(
                    f"Mech skill points {pilot.skills.total_points()} exceed "
                    f"allowed {progression.total_mech_skill_points} for level {pilot.level}."
                ),
            )
        )

    if pilot.total_trigger_points() > progression.pilot_trigger_points:
        issues.append(
            ProgressionIssue(
                code="trigger_points_exceeded",
                message=(
                    f"Trigger points {pilot.total_trigger_points()} exceed "
                    f"allowed {progression.pilot_trigger_points} for level {pilot.level}."
                ),
            )
        )

    if len(pilot.triggers) < 4:
        issues.append(
            ProgressionIssue(
                code="insufficient_triggers",
                message="Pilots should have at least 4 triggers.",
            )
        )

    if pilot.level == 0:
        if pilot.skills.total_points() != progression.total_mech_skill_points:
            issues.append(
                ProgressionIssue(
                    code="ll0_skill_points_invalid",
                    message="LL0 pilots must allocate exactly 2 mech skill points.",
                )
            )
        if pilot.total_talent_ranks() != progression.total_talent_points:
            issues.append(
                ProgressionIssue(
                    code="ll0_talent_points_invalid",
                    message="LL0 pilots must have exactly three rank I talents.",
                )
            )
        if pilot.total_trigger_points() != progression.pilot_trigger_points or len(pilot.triggers) != 4:
            issues.append(
                ProgressionIssue(
                    code="ll0_trigger_points_invalid",
                    message="LL0 pilots must have exactly four triggers at +2 each.",
                )
            )

    if pilot.total_license_levels() < progression.license_points:
        issues.append(
            ProgressionIssue(
                code="license_points_unspent",
                message=(
                    f"License points {pilot.total_license_levels()} are below "
                    f"expected {progression.license_points} for level {pilot.level}."
                ),
                severity="warning",
            )
        )

    if pilot.total_talent_ranks() < progression.total_talent_points:
        issues.append(
            ProgressionIssue(
                code="talent_points_unspent",
                message=(
                    f"Talent points {pilot.total_talent_ranks()} are below "
                    f"expected {progression.total_talent_points} for level {pilot.level}."
                ),
                severity="warning",
            )
        )

    if pilot.skills.total_points() < progression.total_mech_skill_points:
        issues.append(
            ProgressionIssue(
                code="mech_skill_points_unspent",
                message=(
                    f"Mech skill points {pilot.skills.total_points()} are below "
                    f"expected {progression.total_mech_skill_points} for level {pilot.level}."
                ),
                severity="warning",
            )
        )

    if pilot.total_trigger_points() < progression.pilot_trigger_points:
        issues.append(
            ProgressionIssue(
                code="trigger_points_unspent",
                message=(
                    f"Trigger points {pilot.total_trigger_points()} are below "
                    f"expected {progression.pilot_trigger_points} for level {pilot.level}."
                ),
                severity="warning",
            )
        )

    return ProgressionValidation(valid=not any(i.severity == "error" for i in issues), issues=issues)
