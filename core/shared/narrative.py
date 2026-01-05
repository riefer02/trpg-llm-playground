"""Narrative check tier rules and helpers for Lancer TTRPG."""

from typing import Literal
from pydantic import Field, model_validator
from core.shared.models import FrozenModel

from core.shared.rolls import RollType


NarrativeCheckTier = Literal["standard", "risky", "heroic"]


class NarrativeCheckTierRule(FrozenModel):
    """Tier-specific narrative check behavior."""

    tier: NarrativeCheckTier
    success_threshold: int = Field(..., ge=0)
    consequence_threshold: int | None = Field(
        default=None,
        ge=0,
        description="Roll needed to avoid consequences on success",
    )
    allows_push: bool = True


class NarrativeHelpRule(FrozenModel):
    """Rules for helping on a narrative check."""

    accuracy_bonus: int = Field(default=1, ge=0)
    helpers_share_consequences: bool = True


class NarrativePushRule(FrozenModel):
    """Rules for pushing a narrative check to a higher tier."""

    from_tier: NarrativeCheckTier
    to_tier: NarrativeCheckTier
    requires_gm_approval: bool = False


class NarrativeCheckRules(FrozenModel):
    """Combined narrative check rules."""

    default_target: int = Field(default=10, ge=0)
    tiers: list[NarrativeCheckTierRule] = Field(default_factory=list)
    help_rule: NarrativeHelpRule = Field(default_factory=NarrativeHelpRule)
    push_rules: list[NarrativePushRule] = Field(default_factory=list)
    voluntary_failure_allowed: bool = True
    voluntary_failure_applies_to: list[RollType] = Field(
        default_factory=lambda: ["skill_check", "save"],
    )


NARRATIVE_TIER_RULES: list[NarrativeCheckTierRule] = [
    NarrativeCheckTierRule(
        tier="standard",
        success_threshold=10,
        consequence_threshold=None,
        allows_push=True,
    ),
    NarrativeCheckTierRule(
        tier="risky",
        success_threshold=10,
        consequence_threshold=20,
        allows_push=True,
    ),
    NarrativeCheckTierRule(
        tier="heroic",
        success_threshold=20,
        consequence_threshold=20,
        allows_push=False,
    ),
]


NARRATIVE_TIER_RULES_BY_TIER = {rule.tier: rule for rule in NARRATIVE_TIER_RULES}


NARRATIVE_PUSH_RULES: list[NarrativePushRule] = [
    NarrativePushRule(
        from_tier="standard", to_tier="risky", requires_gm_approval=False
    ),
    NarrativePushRule(from_tier="risky", to_tier="heroic", requires_gm_approval=True),
]


DEFAULT_NARRATIVE_CHECK_RULES = NarrativeCheckRules(
    tiers=NARRATIVE_TIER_RULES,
    push_rules=NARRATIVE_PUSH_RULES,
)


def get_narrative_tier_rule(tier: NarrativeCheckTier) -> NarrativeCheckTierRule | None:
    """Look up a tier rule by name."""
    return NARRATIVE_TIER_RULES_BY_TIER.get(tier)


NarrativeEnvironmentType = Literal[
    "urban",
    "wilderness",
    "space_station",
    "space_open",
    "underwater",
    "underground",
    "industrial",
    "residential",
    "natural_ruins",
    "artificial_ruins",
    "ship_interior",
    "vehicle_interior",
    "mixed",
]


NarrativeTimePressure = Literal["none", "urgent", "critical"]


NarrativeNPCDisposition = Literal[
    "hostile", "unfriendly", "neutral", "friendly", "allied"
]


class NarrativeCombatConstraints(FrozenModel):
    """Constraints for narrative combat mode per PR2 rules.

    PR2 states: "Direct bodily harm to pilots in narrative play can only occur
    on risky rolls as a complication - even on rolls of 10-19, in which a pilot
    accomplishes their goal."
    """

    harm_only_on_risky_complication: bool = Field(
        default=True,
        description="Harm only occurs on risky rolls as complication per PR2",
    )
    no_attack_rolls: bool = Field(
        default=True,
        description="No attack rolls in narrative combat - goals resolve directly",
    )
    no_npc_turns: bool = Field(
        default=True,
        description="NPCs don't take turns in narrative combat",
    )
    no_npc_hp_tracking: bool = Field(
        default=True,
        description="Don't track NPC HP - success means goal accomplished",
    )
    npc_goals_require_skill_checks: bool = Field(
        default=False,
        description="NPC actions require skill checks from pilots, not NPC rolls",
    )
    allow_granted_harm_on_standard: bool = Field(
        default=False,
        description="Allow harm to be granted by abilities even on standard rolls",
    )


class NarrativeGoalOutcome(FrozenModel):
    """Outcome resolution for a pilot's goal in narrative combat."""

    goal_id: str | None = Field(
        default=None,
        description="Goal identifier if tracked",
    )
    goal_description: str = Field(
        ..., description="What the pilot was trying to accomplish"
    )
    success: bool = Field(..., description="Whether the goal was accomplished")
    tier_attained: NarrativeCheckTier = Field(
        default="standard",
        description="The tier at which success was achieved",
    )
    harm_involved: bool = Field(
        default=False,
        description="Whether physical harm was part of the goal",
    )
    harm_suffered: bool = Field(
        default=False,
        description="Whether the pilot suffered harm as complication",
    )
    complication_description: str | None = Field(
        default=None,
        description="If harm or other complication occurred",
    )


PilotSkillType = Literal["hull", "agility", "systems", "engineering"]


NarrativeComplicationType = Literal[
    "harm",
    "time",
    "resources",
    "collateral",
    "position",
    "effect",
]


NarrativeComplicationSeverity = Literal["minor", "major", "lethal"]


NarrativeComplicationTrigger = Literal[
    "failure",
    "risky_success",
    "heroic_success",
    "other",
]


NarrativeResolutionRequirementType = Literal[
    "skill_check",
    "time_passed",
    "resource_spend",
    "position_change",
    "goal_completion",
    "effect_reversal",
    "other",
]


class NarrativeResolutionRequirement(FrozenModel):
    """Requirements for resolving a narrative complication."""

    requirement_type: NarrativeResolutionRequirementType
    description: str = Field(..., description="What clears this complication")
    required_tier: NarrativeCheckTier | None = Field(
        default=None,
        description="Narrative tier required (if resolution needs a check)",
    )
    required_skill: PilotSkillType | None = Field(
        default=None,
        description="Pilot skill used to resolve (if check-based)",
    )
    required_amount: int | None = Field(
        default=None,
        ge=0,
        description="Quantity of time/resources/etc required",
    )


class NarrativeComplication(FrozenModel):
    """Complication that persists in narrative combat until resolved."""

    id: str = Field(..., description="Unique complication identifier")
    complication_type: NarrativeComplicationType
    description: str = Field(..., description="What is going wrong")
    severity: NarrativeComplicationSeverity = Field(
        default="minor",
        description="Severity of the complication",
    )
    established_before_roll: bool = Field(
        default=True,
        description="Whether the complication was established before the roll",
    )
    trigger: NarrativeComplicationTrigger | None = Field(
        default=None,
        description="What triggered the complication",
    )
    target_ids: list[str] = Field(
        default_factory=list,
        description="IDs of pilots or targets affected",
    )
    resolution_requirements: list[NarrativeResolutionRequirement] = Field(
        default_factory=list,
        description="Requirements to resolve the complication",
    )
    harm_damage: int | None = Field(
        default=None,
        ge=0,
        description="Damage amount if this is a harm complication",
    )

    @model_validator(mode="after")
    def validate_harm_details(self) -> "NarrativeComplication":
        if self.harm_damage is not None and self.complication_type != "harm":
            raise ValueError(
                "harm_damage is only valid for complications of type 'harm'."
            )
        return self


NarrativeComplicationStatus = Literal["active", "resolved", "escalated"]


class NarrativeComplicationState(FrozenModel):
    """State wrapper for a complication in narrative combat."""

    complication: NarrativeComplication
    status: NarrativeComplicationStatus = "active"
    resolved_by: str | None = Field(
        default=None,
        description="Who resolved the complication",
    )
    resolution_notes: str | None = Field(
        default=None,
        description="How the complication was resolved",
    )


NarrativeGoalConditionType = Literal[
    "skill_check",
    "position_reached",
    "target_removed",
    "resource_spend",
    "time_elapsed",
    "complication_resolved",
    "other",
]


class NarrativeGoalCondition(FrozenModel):
    """Condition that must be met to complete a narrative goal."""

    id: str = Field(..., description="Unique condition identifier")
    condition_type: NarrativeGoalConditionType
    description: str = Field(..., description="What must be true for this condition")
    required_tier: NarrativeCheckTier | None = Field(
        default=None,
        description="Tier needed if condition is check-based",
    )
    required_skill: PilotSkillType | None = Field(
        default=None,
        description="Skill used if condition is check-based",
    )
    required_amount: int | None = Field(
        default=None,
        ge=0,
        description="Quantity needed (time/resources/etc)",
    )
    target_id: str | None = Field(
        default=None,
        description="Target or object tied to this condition",
    )


NarrativeGoalStatus = Literal["active", "completed", "failed", "blocked"]


class NarrativeGoal(FrozenModel):
    """Goal definition for narrative combat."""

    id: str = Field(..., description="Unique goal identifier")
    description: str = Field(..., description="Pilot's stated goal")
    success_conditions: list[NarrativeGoalCondition] = Field(
        default_factory=list,
        description="Conditions that represent success",
    )
    successes_required: int | None = Field(
        default=None,
        ge=1,
        description="Successful checks required to complete the goal",
    )
    failure_limit: int | None = Field(
        default=None,
        ge=0,
        description="Failures allowed before the goal fails",
    )
    harm_involved: bool = Field(
        default=False,
        description="Whether the goal involves physical harm",
    )
    repeat_requires_change: bool = Field(
        default=True,
        description="Checks cannot repeat until circumstances change",
    )

    @model_validator(mode="before")
    @classmethod
    def default_successes_required(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        success_conditions = data.get("success_conditions") or []
        successes_required = data.get("successes_required")

        if successes_required is None:
            data["successes_required"] = (
                len(success_conditions) if success_conditions else 1
            )
        elif success_conditions and successes_required > len(success_conditions):
            raise ValueError(
                "successes_required cannot exceed number of success_conditions."
            )

        return data


class NarrativeGoalAttempt(FrozenModel):
    """Resolution details for a single narrative goal attempt."""

    roll_result: int = Field(..., ge=1, le=20)
    target: int = Field(default=10, ge=0)
    modifiers: int = Field(default=0)
    difficulty_modifier: int = Field(default=0, ge=0)
    tier: NarrativeCheckTier = "standard"
    total_result: int = Field(..., description="Final total after modifiers")
    success: bool = Field(..., description="Whether the goal check succeeded")
    consequence_suffered: bool = Field(
        default=False,
        description="Whether consequences apply (risky/heroic)",
    )
    complication_type: NarrativeComplicationType | None = Field(
        default=None,
        description="Type of complication suffered, if any",
    )
    complication_description: str | None = Field(
        default=None,
        description="Complication details if any",
    )
    pilot_id: str | None = Field(
        default=None,
        description="Pilot taking the action",
    )
    skill_used: PilotSkillType | None = Field(
        default=None,
        description="Pilot skill used for the check",
    )
    action_id: str | None = Field(
        default=None,
        description="Action identifier tied to the attempt",
    )
    action_description: str | None = Field(
        default=None,
        description="Narrative description of the action",
    )


class NarrativeGoalState(FrozenModel):
    """State wrapper for a narrative goal."""

    goal: NarrativeGoal
    status: NarrativeGoalStatus = "active"
    successes: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)
    last_outcome: NarrativeGoalOutcome | None = Field(
        default=None,
        description="Most recent goal outcome",
    )
    last_attempt: NarrativeGoalAttempt | None = Field(
        default=None,
        description="Most recent goal attempt details",
    )
    attempt_history: list[NarrativeGoalAttempt] = Field(default_factory=list)


class NarrativeGoalTracker(FrozenModel):
    """Tracker for active narrative goals."""

    goals: list[NarrativeGoalState] = Field(default_factory=list)


class NarrativeCombatState(FrozenModel):
    """Narrative combat state tracking complications and goals."""

    scene_id: str | None = Field(
        default=None,
        description="Optional scene identifier",
    )
    complications: list[NarrativeComplicationState] = Field(default_factory=list)
    goal_tracker: NarrativeGoalTracker = Field(
        default_factory=NarrativeGoalTracker
    )


def add_narrative_complication(
    state: NarrativeCombatState,
    complication: NarrativeComplication,
    status: NarrativeComplicationStatus = "active",
) -> NarrativeCombatState:
    """Append a complication to narrative combat state."""
    entry = NarrativeComplicationState(
        complication=complication,
        status=status,
    )
    return state.model_copy(
        update={"complications": [*state.complications, entry]}
    )


def resolve_narrative_complication(
    state: NarrativeCombatState,
    complication_id: str,
    resolution_notes: str | None = None,
    resolved_by: str | None = None,
) -> NarrativeCombatState:
    """Resolve a narrative complication by ID."""
    updated: list[NarrativeComplicationState] = []
    found = False
    for entry in state.complications:
        if entry.complication.id == complication_id:
            found = True
            updated.append(
                entry.model_copy(
                    update={
                        "status": "resolved",
                        "resolution_notes": resolution_notes,
                        "resolved_by": resolved_by,
                    }
                )
            )
        else:
            updated.append(entry)

    if not found:
        raise ValueError(f"Unknown complication '{complication_id}'.")

    return state.model_copy(update={"complications": updated})


def add_narrative_goal(
    tracker: NarrativeGoalTracker,
    goal: NarrativeGoal,
    status: NarrativeGoalStatus = "active",
) -> NarrativeGoalTracker:
    """Append a narrative goal to the tracker."""
    entry = NarrativeGoalState(goal=goal, status=status)
    return tracker.model_copy(update={"goals": [*tracker.goals, entry]})


def resolve_narrative_goal_check(
    tracker: NarrativeGoalTracker,
    goal_id: str,
    roll_result: int,
    target: int = 10,
    modifiers: int = 0,
    difficulty_modifier: int = 0,
    tier: NarrativeCheckTier = "standard",
    skill_used: PilotSkillType | None = None,
    action_id: str | None = None,
    action_description: str | None = None,
    pilot_id: str | None = None,
    complication_type: NarrativeComplicationType | None = None,
    complication_description: str | None = None,
    constraints: NarrativeCombatConstraints | None = None,
    circumstances_changed: bool = True,
    tier_rules: list[NarrativeCheckTierRule] | None = None,
) -> tuple[NarrativeGoalTracker, NarrativeGoalOutcome]:
    """Resolve a narrative goal check and update the tracker."""
    updated: list[NarrativeGoalState] = []
    found = False
    outcome: NarrativeGoalOutcome | None = None

    for entry in tracker.goals:
        if entry.goal.id != goal_id:
            updated.append(entry)
            continue

        found = True
        goal = entry.goal
        if entry.status in {"completed", "failed"}:
            raise ValueError(f"Goal '{goal_id}' is already {entry.status}.")

        if (
            goal.repeat_requires_change
            and not circumstances_changed
            and entry.last_outcome
            and not entry.last_outcome.success
        ):
            raise ValueError(
                "Goal check cannot be repeated until circumstances change."
            )

        success, consequence_suffered, total_result = compute_check_success(
            roll_result=roll_result,
            target=target,
            modifiers=modifiers,
            difficulty_modifier=difficulty_modifier,
            tier=tier,
            tier_rules=tier_rules,
        )

        effective_complication_type = (
            complication_type if consequence_suffered else None
        )
        effective_complication_description = (
            complication_description if consequence_suffered else None
        )

        harm_suffered = False
        if effective_complication_type == "harm":
            allow_harm = True
            if constraints and constraints.harm_only_on_risky_complication:
                allow_harm = tier != "standard" or (
                    constraints.allow_granted_harm_on_standard
                )
            harm_suffered = allow_harm

        outcome = NarrativeGoalOutcome(
            goal_id=goal.id,
            goal_description=goal.description,
            success=success,
            tier_attained=tier,
            harm_involved=goal.harm_involved,
            harm_suffered=harm_suffered,
            complication_description=effective_complication_description,
        )

        attempt = NarrativeGoalAttempt(
            roll_result=roll_result,
            target=target,
            modifiers=modifiers,
            difficulty_modifier=difficulty_modifier,
            tier=tier,
            total_result=total_result,
            success=success,
            consequence_suffered=consequence_suffered,
            complication_type=effective_complication_type,
            complication_description=effective_complication_description,
            pilot_id=pilot_id,
            skill_used=skill_used,
            action_id=action_id,
            action_description=action_description,
        )

        successes = entry.successes + (1 if success else 0)
        failures = entry.failures + (0 if success else 1)
        attempts = entry.attempts + 1

        status = entry.status
        required_successes = goal.successes_required or 1
        if success and successes >= required_successes:
            status = "completed"
        elif goal.failure_limit is not None and failures >= goal.failure_limit:
            status = "failed"

        updated.append(
            entry.model_copy(
                update={
                    "status": status,
                    "successes": successes,
                    "failures": failures,
                    "attempts": attempts,
                    "last_outcome": outcome,
                    "last_attempt": attempt,
                    "attempt_history": [*entry.attempt_history, attempt],
                }
            )
        )

    if not found:
        raise ValueError(f"Unknown goal '{goal_id}'.")

    if outcome is None:
        raise ValueError(f"Unable to resolve goal '{goal_id}'.")

    return tracker.model_copy(update={"goals": updated}), outcome


class NarrativeScenarioSettings(FrozenModel):
    """Global settings for a narrative scenario/scene.

    These settings apply to the entire scene and inform how mechanics resolve.
    """

    environment: NarrativeEnvironmentType = Field(
        default="urban",
        description="Physical environment of the scene",
    )
    is_combat: bool = Field(
        default=False, description="Whether this is narrative combat"
    )
    combat_constraints: NarrativeCombatConstraints | None = Field(
        default=None,
        description="Constraints that apply if is_combat is True",
    )
    time_pressure: NarrativeTimePressure = Field(
        default="none",
        description="Time pressure level for the scene",
    )
    npc_disposition: NarrativeNPCDisposition = Field(
        default="neutral",
        description="General disposition of NPCs in the scene",
    )
    has_allies_present: bool = Field(
        default=False,
        description="Whether allied NPCs are present",
    )
    has_enemies_present: bool = Field(
        default=True,
        description="Whether enemy NPCs are present",
    )
    cover_available: bool = Field(
        default=True,
        description="Whether cover is available in the environment",
    )


DEFAULT_NARRATIVE_SCENARIO_SETTINGS = NarrativeScenarioSettings()


class PrecedenceRule(FrozenModel):
    """Rule that overrides a more general rule (specific overshadows general)."""

    rule_id: str = Field(..., description="ID of the rule being overridden")
    overrides_rule_id: str = Field(
        ...,
        description="ID of the more general rule this overrides",
    )
    precedence_level: int = Field(
        default=1,
        ge=1,
        description="Higher values take precedence when multiple overrides apply",
    )
    context_description: str = Field(
        ...,
        description="When this precedence rule applies",
    )

class SkillChallengeType(FrozenModel):
    """Type/purpose of a skill challenge."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="What kind of challenge this represents")


SKILL_CHALLENGE_TYPES: list[SkillChallengeType] = [
    SkillChallengeType(
        id="combat",
        name="Combat",
        description="Direct confrontation, infiltration, or combat-related challenge",
    ),
    SkillChallengeType(
        id="social",
        name="Social",
        description="Negotiation, persuasion, deception, or social interaction",
    ),
    SkillChallengeType(
        id="infiltration",
        name="Infiltration",
        description="Sneaking, bypassing security, or covert operations",
    ),
    SkillChallengeType(
        id="investigation",
        name="Investigation",
        description="Gathering information, searching, or solving mysteries",
    ),
    SkillChallengeType(
        id="chase",
        name="Chase",
        description="Pursuit, escape, or vehicle-based challenges",
    ),
    SkillChallengeType(
        id="environmental",
        name="Environmental",
        description="Surviving hazards, navigating terrain, or overcoming obstacles",
    ),
    SkillChallengeType(
        id="technical",
        name="Technical",
        description="Hacking, engineering, or technical skill challenges",
    ),
    SkillChallengeType(
        id="combined",
        name="Combined",
        description="Mixed challenge requiring multiple different approaches",
    ),
]


SKILL_CHALLENGE_TYPE_BY_ID = {t.id: t for t in SKILL_CHALLENGE_TYPES}


class IndividualCheckResult(FrozenModel):
    """Result of a single participant's check in a skill challenge."""

    participant_id: str = Field(..., description="ID of the pilot making the check")
    trigger_used: str = Field(..., description="Trigger name/ID used for this check")
    skill_context: PilotSkillType = Field(
        ...,
        description="Which pilot skill was used",
    )
    roll_result: int = Field(..., ge=1, le=20, description="d20 roll result")
    modifiers_applied: str | None = Field(
        default=None,
        description="Description of modifiers applied",
    )
    difficulty_modifier: int = Field(
        default=0,
        ge=0,
        description="Additional difficulty applied to this check",
    )
    total_result: int = Field(
        ...,
        description="Final total after modifiers and difficulty",
    )
    is_success: bool = Field(..., description="Whether this individual check succeeded")
    consequence_suffered: bool = Field(
        default=False,
        description="Whether a consequence was suffered (for risky/heroic)",
    )
    consequence_description: str | None = Field(
        default=None,
        description="If a consequence occurred",
    )


class SkillChallengeResult(FrozenModel):
    """Result of a skill challenge resolution."""

    total_participants: int = Field(..., description="Number of participants")
    success_count: int = Field(
        ..., description="Number of successful individual checks"
    )
    failure_count: int = Field(..., description="Number of failed individual checks")
    is_success: bool = Field(..., description="Whether the challenge succeeded")
    required_for_success: int = Field(
        ...,
        description="Participants needed for success",
    )
    was_tie: bool = Field(
        default=False,
        description="Whether this was a tie (50% success chance)",
    )
    tie_roll_result: int | None = Field(
        default=None,
        ge=0,
        le=1,
        description="50% roll result if tie (0=fail, 1=success)",
    )
    overall_consequences: list[str] = Field(
        default_factory=list,
        description="Consequences that apply to the group",
    )
    individual_results: list[IndividualCheckResult] = Field(
        default_factory=list,
        description="Individual check results for record-keeping",
    )


class SkillChallengeDefinition(FrozenModel):
    """Definition of a skill challenge scenario."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    challenge_type: str = Field(
        ...,
        description="Type ID from SKILL_CHALLENGE_TYPES",
    )
    description: str = Field(
        ...,
        description="Narrative description of the challenge",
    )
    target_difficulty: int = Field(
        default=10,
        ge=0,
        description="Base difficulty target for checks",
    )
    default_tier: NarrativeCheckTier = Field(
        default="standard",
        description="Default tier for challenge checks",
    )
    participant_count_min: int = Field(
        default=1,
        ge=1,
        description="Minimum participants for the challenge",
    )
    participant_count_max: int = Field(
        default=6,
        ge=1,
        description="Maximum participants for the challenge",
    )
    allows_help: bool = Field(
        default=True,
        description="Whether participants can help each other",
    )
    allows_push: bool = Field(
        default=True,
        description="Whether participants can push their checks",
    )
    time_constraint_turns: int | None = Field(
        default=None,
        description="If set, challenge must be resolved in N turns",
    )


class SkillChallengeUse(FrozenModel):
    """An instance of a skill challenge being resolved in play."""

    definition: SkillChallengeDefinition = Field(
        ...,
        description="The challenge definition being used",
    )
    scenario_settings: NarrativeScenarioSettings = Field(
        default_factory=NarrativeScenarioSettings,
        description="Scenario settings that apply to this challenge",
    )
    participant_ids: list[str] = Field(
        default_factory=list,
        description="IDs of participating pilots",
    )
    individual_checks: list[IndividualCheckResult] = Field(
        default_factory=list,
        description="Results of individual participant checks",
    )
    resolution: SkillChallengeResult | None = Field(
        default=None,
        description="Resolved outcome once challenge is complete",
    )


def resolve_skill_challenge(
    challenge: SkillChallengeUse,
    tier_rules: list[NarrativeCheckTierRule] | None = None,
) -> SkillChallengeResult:
    """Resolve a skill challenge and return the result.

    Args:
        challenge: The skill challenge to resolve
        tier_rules: Optional custom tier rules (uses defaults if None)

    Returns:
        The resolved challenge result
    """
    if tier_rules is None:
        tier_rules = NARRATIVE_TIER_RULES

    if not challenge.individual_checks:
        return SkillChallengeResult(
            total_participants=len(challenge.participant_ids),
            success_count=0,
            failure_count=0,
            is_success=False,
            required_for_success=0,
            overall_consequences=["No checks were made"],
        )

    results = challenge.individual_checks
    success_count = sum(1 for r in results if r.is_success)
    failure_count = len(results) - success_count
    total_participants = len(results)

    required_for_success = (total_participants // 2) + 1

    if success_count > failure_count:
        is_success = True
        was_tie = False
        tie_roll_result = None
    elif failure_count > success_count:
        is_success = False
        was_tie = False
        tie_roll_result = None
    else:
        was_tie = True
        import random

        tie_roll_result = random.randint(0, 1)
        is_success = tie_roll_result == 1

    overall_consequences = []
    for result in results:
        if result.consequence_suffered and result.consequence_description:
            overall_consequences.append(
                f"{result.participant_id}: {result.consequence_description}"
            )

    return SkillChallengeResult(
        total_participants=total_participants,
        success_count=success_count,
        failure_count=failure_count,
        is_success=is_success,
        required_for_success=required_for_success,
        was_tie=was_tie,
        tie_roll_result=tie_roll_result,
        overall_consequences=overall_consequences,
        individual_results=results,
    )


def compute_check_success(
    roll_result: int,
    target: int,
    modifiers: int = 0,
    difficulty_modifier: int = 0,
    tier: NarrativeCheckTier = "standard",
    tier_rules: list[NarrativeCheckTierRule] | None = None,
) -> tuple[bool, bool, int]:
    """Compute if a narrative check succeeds and if consequences apply.

    Args:
        roll_result: The d20 roll
        target: Base target number (usually 10)
        modifiers: Accuracy/difficulty modifiers to apply
        difficulty_modifier: Additional difficulty (e.g., +1 for difficult checks)
        tier: The check tier (standard/risky/heroic)
        tier_rules: Optional custom tier rules

    Returns:
        Tuple of (is_success, consequence_suffered, total_result)
    """
    if tier_rules is None:
        tier_rules = NARRATIVE_TIER_RULES

    tier_rule_map = {r.tier: r for r in tier_rules}
    rule = tier_rule_map.get(tier, tier_rule_map["standard"])

    total_result = roll_result + modifiers - difficulty_modifier
    is_success = total_result >= rule.success_threshold

    consequence_suffered = False
    if rule.consequence_threshold is not None:
        if total_result < rule.consequence_threshold:
            consequence_suffered = True

    return is_success, consequence_suffered, total_result
