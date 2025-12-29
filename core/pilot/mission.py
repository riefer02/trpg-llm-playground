"""Mission cadence, downtime actions, and reserve handling models for Lancer TTRPG."""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel


MissionPhaseType = Literal["uptime", "downtime"]
DowntimeActionCategory = Literal[
    "recovery",
    "intel",
    "resource",
    "contact",
    "project",
    "tradeoff",
    "other",
]
DowntimeOutcomeType = Literal[
    "reserve",
    "info",
    "contact",
    "project",
    "recovery",
    "custom",
]
ReserveSource = Literal["downtime_action", "mission_reward", "other"]


class MissionCadenceRules(FrozenModel):
    """Cadence and refresh rules for missions."""

    min_uptime_scenes: int = Field(default=1, ge=0)
    max_uptime_scenes: int = Field(default=4, ge=0)
    downtime_actions_per_pilot: int = Field(default=2, ge=0)
    downtime_actions_repeatable: bool = True
    full_repair_on_mission_start: bool = True
    core_power_refresh_on_mission_start: bool = True
    reserves_expire_on_mission_end: bool = True
    reserve_pool_scope: Literal["shared", "per_pilot"] = "shared"



DEFAULT_MISSION_CADENCE_RULES = MissionCadenceRules()


class ReserveEntry(FrozenModel):
    """A single reserve available during a mission."""

    id: str = Field(..., description="Unique reserve identifier")
    name: str = Field(..., description="Display name")
    source: ReserveSource = "downtime_action"
    uses_remaining: int = Field(default=1, ge=0)
    shared: bool = True
    expires_on_mission_end: bool = True



class DowntimeActionDefinition(FrozenModel):
    """Definition for a downtime action."""

    id: str = Field(..., description="Unique downtime action identifier")
    name: str = Field(..., description="Display name")
    category: DowntimeActionCategory
    outcomes: list[DowntimeOutcomeType] = Field(default_factory=list)
    requires_skill_check: bool | None = Field(
        default=None,
        description="None means the GM determines whether a check is needed",
    )
    grants_reserve: bool = False



class DowntimeActionUse(FrozenModel):
    """A single downtime action taken by a pilot."""

    action_id: str
    outcome: DowntimeOutcomeType
    reserve: ReserveEntry | None = None



class DowntimePlan(FrozenModel):
    """Recorded downtime actions for a pilot."""

    pilot_id: str
    actions: list[DowntimeActionUse] = Field(default_factory=list)



DOWNTIME_ACTION_DEFINITIONS: list[DowntimeActionDefinition] = [
    DowntimeActionDefinition(
        id="get_a_damn_drink",
        name="Get a Damn Drink",
        category="recovery",
        outcomes=["recovery"],
    ),
    DowntimeActionDefinition(
        id="get_a_clue",
        name="Get a Clue",
        category="intel",
        outcomes=["info"],
    ),
    DowntimeActionDefinition(
        id="get_a_hold_of_something",
        name="Get a Hold of Something",
        category="resource",
        outcomes=["reserve"],
        grants_reserve=True,
    ),
    DowntimeActionDefinition(
        id="get_connected",
        name="Get Connected",
        category="contact",
        outcomes=["contact", "reserve"],
        grants_reserve=True,
    ),
    DowntimeActionDefinition(
        id="get_organized",
        name="Get Organized",
        category="project",
        outcomes=["project", "reserve"],
        grants_reserve=True,
    ),
    DowntimeActionDefinition(
        id="power_at_a_cost",
        name="Power at a Cost",
        category="tradeoff",
        outcomes=["reserve", "custom"],
        grants_reserve=True,
    ),
]


DOWNTIME_ACTIONS_BY_ID = {action.id: action for action in DOWNTIME_ACTION_DEFINITIONS}


def get_downtime_action_definition(action_id: str) -> DowntimeActionDefinition | None:
    """Look up a downtime action definition by ID."""
    return DOWNTIME_ACTIONS_BY_ID.get(action_id)


class DowntimeIssue(FrozenModel):
    """A downtime validation issue."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"



class DowntimeValidation(FrozenModel):
    """Validation result for downtime planning."""

    valid: bool
    issues: list[DowntimeIssue] = Field(default_factory=list)



def validate_downtime_plan(
    plan: DowntimePlan,
    rules: MissionCadenceRules = DEFAULT_MISSION_CADENCE_RULES,
    action_definitions: dict[str, DowntimeActionDefinition] | None = None,
) -> DowntimeValidation:
    """Validate a downtime plan against cadence rules and action definitions."""
    issues: list[DowntimeIssue] = []
    definitions = action_definitions or DOWNTIME_ACTIONS_BY_ID

    if len(plan.actions) > rules.downtime_actions_per_pilot:
        issues.append(
            DowntimeIssue(
                code="too_many_downtime_actions",
                message=(
                    f"Downtime actions {len(plan.actions)} exceed "
                    f"allowed {rules.downtime_actions_per_pilot}."
                ),
            )
        )

    seen_reserves: set[str] = set()
    for action_use in plan.actions:
        definition = definitions.get(action_use.action_id)
        if not definition:
            issues.append(
                DowntimeIssue(
                    code="unknown_downtime_action",
                    message=f"Unknown downtime action: {action_use.action_id}.",
                )
            )
            continue

        if definition.outcomes and action_use.outcome not in definition.outcomes:
            issues.append(
                DowntimeIssue(
                    code="invalid_downtime_outcome",
                    message=(
                        f"Outcome {action_use.outcome} is not valid for "
                        f"{definition.name}."
                    ),
                )
            )

        if definition.grants_reserve and action_use.reserve is None:
            issues.append(
                DowntimeIssue(
                    code="missing_reserve",
                    message=f"{definition.name} should grant a reserve.",
                )
            )

        if action_use.reserve:
            if action_use.reserve.id in seen_reserves:
                issues.append(
                    DowntimeIssue(
                        code="duplicate_reserve_id",
                        message=f"Reserve ID {action_use.reserve.id} is duplicated.",
                    )
                )
            seen_reserves.add(action_use.reserve.id)
            if not definition.grants_reserve:
                issues.append(
                    DowntimeIssue(
                        code="unexpected_reserve",
                        message=(
                            f"Reserve provided for {definition.name}, "
                            "but this action does not grant reserves."
                        ),
                        severity="warning",
                    )
                )
            if action_use.reserve.source != "downtime_action":
                issues.append(
                    DowntimeIssue(
                        code="reserve_source_mismatch",
                        message=(
                            f"Reserve {action_use.reserve.id} source should be "
                            "downtime_action for this action."
                        ),
                        severity="warning",
                    )
                )

    return DowntimeValidation(valid=not any(i.severity == "error" for i in issues), issues=issues)
