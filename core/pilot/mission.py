"""Mission cadence, downtime actions, and reserve handling models for Lancer TTRPG."""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.effects import MechanicalEffect


RollTier = Literal["failure", "partial", "success"]


class RollTierOutcome(FrozenModel):
    """Outcome for a specific roll tier (9-, 10-19, 20+)."""

    tier: RollTier
    description: str = Field(..., description="Narrative description of outcome")
    grants_reserve: bool = Field(
        default=False,
        description="Whether this outcome grants a reserve",
    )
    reserve_id: str | None = Field(
        default=None,
        description="Reserve ID to grant if applicable",
    )
    reserve_uses: int | None = Field(
        default=None,
        ge=0,
        description="Uses for the reserve if applicable",
    )
    custom_effect: MechanicalEffect | None = Field(
        default=None,
        description="Mechanical effect to apply (e.g., organization stat change)",
    )
    triggers_gm_choice: bool = Field(
        default=False,
        description="Requires GM to choose between options",
    )
    choice_options: list[str] = Field(
        default_factory=list,
        description="Options for GM choice if applicable",
    )


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
ReserveType = Literal["narrative", "mech", "tactical"]
ReserveSource = Literal["downtime_action", "mission_reward", "other"]


class OrganizationState(FrozenModel):
    """State tracking for Get Organized downtime action."""

    efficiency: int = Field(default=0, ge=0, le=6)
    influence: int = Field(default=0, ge=0, le=6)

    @classmethod
    def initial_state(cls) -> "OrganizationState":
        """Create initial state: +2 to one track, +0 to other."""
        return cls(efficiency=2, influence=0)

    def gain(self, is_success: bool) -> "OrganizationState":
        """Gain stats based on roll outcome."""
        if is_success:
            new_eff = min(6, self.efficiency + 2)
            new_inf = min(6, self.influence + 2)
        else:
            new_eff = min(6, self.efficiency + 1)
            new_inf = min(6, self.influence + 1)
        return self.model_copy(update={"efficiency": new_eff, "influence": new_inf})

    def degrade(self) -> "OrganizationState":
        """Degrade by 2 on failure (minimum 0)."""
        new_eff = max(0, self.efficiency - 2)
        new_inf = max(0, self.influence - 2)
        return self.model_copy(update={"efficiency": new_eff, "influence": new_inf})


class OrganizationDefinition(FrozenModel):
    """Definition for an organization created via Get Organized."""

    id: str = Field(..., description="Unique organization identifier")
    name: str = Field(..., description="Organization name")
    purpose: Literal[
        "military",
        "scientific",
        "academic",
        "criminal",
        "humanitarian",
        "industrial",
        "entertainment",
        "political",
    ] = Field(..., description="Organization purpose/goal")
    state: OrganizationState = Field(
        default_factory=OrganizationState.initial_state,
        description="Current efficiency/influence state",
    )


class TriggerGrant(FrozenModel):
    """Trigger granted via Get Focused downtime action."""

    trigger_name: str = Field(..., description="Name/description of trigger")
    trigger_bonus: int = Field(default=2, ge=0, le=6)
    skill_context: str = Field(..., description="Skill or context for trigger")


class MissionCadenceRules(FrozenModel):
    """Cadence and refresh rules for missions."""

    min_uptime_scenes: int = Field(default=1, ge=0)
    max_uptime_scenes: int = Field(default=4, ge=0)
    downtime_actions_per_pilot: int = Field(default=1, ge=0)
    downtime_actions_long_session: int = Field(default=2, ge=0)
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
    reserve_type: ReserveType = Field(
        default="narrative", description="Type of reserve"
    )
    source: ReserveSource = "downtime_action"
    uses_remaining: int = Field(default=1, ge=0)
    shared: bool = True
    expires_on_mission_end: bool = True
    mechanical_effect: MechanicalEffect | None = Field(
        default=None,
        description="Mechanical effect when reserve is used",
    )


class ReserveDefinition(FrozenModel):
    """Definition for a reserve type from the PR2 tables."""

    id: str = Field(..., description="Reserve identifier (used in reserves table)")
    name: str = Field(..., description="Display name")
    reserve_type: ReserveType
    d20_min: int = Field(..., ge=1, le=20, description="Minimum d20 roll")
    d20_max: int = Field(..., ge=1, le=20, description="Maximum d20 roll")
    description: str = Field(..., description="Narrative description")
    mechanical_effect: MechanicalEffect | None = Field(
        default=None,
        description="Mechanical effect when granted/used",
    )
    uses_modifier: int | None = Field(
        default=None,
        description="Modifier to uses (e.g., +1/+2 for ammo)",
    )


NARRATIVE_RESERVES: list[ReserveDefinition] = [
    ReserveDefinition(
        id="narrative_access",
        name="Access",
        reserve_type="narrative",
        d20_min=1,
        d20_max=2,
        description="Gain a keycard, invite, bribes, or insider access to a location",
    ),
    ReserveDefinition(
        id="narrative_backing",
        name="Backing",
        reserve_type="narrative",
        d20_min=3,
        d20_max=4,
        description="Political support from someone powerful - invoke as leverage",
    ),
    ReserveDefinition(
        id="narrative_supplies",
        name="Supplies",
        reserve_type="narrative",
        d20_min=5,
        d20_max=6,
        description="Cross hazardous/hostile area without skill check",
    ),
    ReserveDefinition(
        id="narrative_disguise",
        name="Disguise",
        reserve_type="narrative",
        d20_min=7,
        d20_max=8,
        description="Prepare disguise/cover identity to sneak in uncontested",
    ),
    ReserveDefinition(
        id="narrative_diversion",
        name="Diversion",
        reserve_type="narrative",
        d20_min=9,
        d20_max=10,
        description="Prepare diversion giving time to act without consequence",
    ),
    ReserveDefinition(
        id="narrative_blackmail",
        name="Blackmail",
        reserve_type="narrative",
        d20_min=11,
        d20_max=12,
        description="Gain blackmail or sensitive information on a person",
    ),
    ReserveDefinition(
        id="narrative_reputation",
        name="Reputation",
        reserve_type="narrative",
        d20_min=13,
        d20_max=14,
        description="Good reputation - start mission on good footing with everyone",
    ),
    ReserveDefinition(
        id="narrative_safe_harbor",
        name="Safe Harbor",
        reserve_type="narrative",
        d20_min=15,
        d20_max=16,
        description="Guaranteed safe location to convene, plan, or recuperate",
    ),
    ReserveDefinition(
        id="narrative_tracking",
        name="Tracking",
        reserve_type="narrative",
        d20_min=17,
        d20_max=18,
        description="Know location of important objects or people for mission",
    ),
    ReserveDefinition(
        id="narrative_knowledge",
        name="Knowledge",
        reserve_type="narrative",
        d20_min=19,
        d20_max=20,
        description="Gain knowledge of local history, customs, or culture etiquette",
    ),
]


MECH_RESERVES: list[ReserveDefinition] = [
    ReserveDefinition(
        id="mech_ammo_1",
        name="Ammo (+1 use)",
        reserve_type="mech",
        d20_min=1,
        d20_max=2,
        description="Get extra uses (+1) to a limited weapon or system",
        uses_modifier=1,
    ),
    ReserveDefinition(
        id="mech_ammo_2",
        name="Ammo (+2 uses)",
        reserve_type="mech",
        d20_min=3,
        d20_max=4,
        description="Get extra uses (+2) to a limited weapon or system",
        uses_modifier=2,
    ),
    ReserveDefinition(
        id="mech_rented_gear",
        name="Rented Gear",
        reserve_type="mech",
        d20_min=5,
        d20_max=6,
        description="Access to weapon or mech gear for mission only",
    ),
    ReserveDefinition(
        id="mech_extra_repairs",
        name="Extra Repairs (+2)",
        reserve_type="mech",
        d20_min=7,
        d20_max=8,
        description="Start mission with +2 repairs on your mech",
    ),
    ReserveDefinition(
        id="mech_core_battery",
        name="Core Battery",
        reserve_type="mech",
        d20_min=9,
        d20_max=10,
        description="Consume to gain core power (can't have more than 1 at a time)",
    ),
    ReserveDefinition(
        id="mech_deployable_shield",
        name="Deployable Shield",
        reserve_type="mech",
        d20_min=11,
        d20_max=12,
        description="Size 1 deployable, grants soft cover to allies in burst 2",
    ),
    ReserveDefinition(
        id="mech_redundant_repair",
        name="Redundant Repair",
        reserve_type="mech",
        d20_min=13,
        d20_max=14,
        description="1/mission: make stabilize action as free",
    ),
    ReserveDefinition(
        id="mech_systems_reinforcement",
        name="Systems Reinforcement",
        reserve_type="mech",
        d20_min=15,
        d20_max=16,
        description="+1 accuracy to HASE checks (choose one) this mission",
    ),
    ReserveDefinition(
        id="mech_smart_ammo",
        name="Smart Ammo",
        reserve_type="mech",
        d20_min=17,
        d20_max=18,
        description="Weapons of choice gain smart tag this mission",
    ),
    ReserveDefinition(
        id="mech_boosted_servos",
        name="Boosted Servos",
        reserve_type="mech",
        d20_min=19,
        d20_max=20,
        description="Immune to slowed condition this mission",
    ),
]


TACTICAL_RESERVES: list[ReserveDefinition] = [
    ReserveDefinition(
        id="tactical_scouting",
        name="Scouting",
        reserve_type="tactical",
        d20_min=1,
        d20_max=2,
        description="Detailed info on mechs/threats: number, type, statistics",
    ),
    ReserveDefinition(
        id="tactical_vehicle",
        name="Vehicle",
        reserve_type="tactical",
        d20_min=3,
        d20_max=4,
        description="Transport vehicle or starship for mission (tier 1 NPC)",
    ),
    ReserveDefinition(
        id="tactical_reinforcements",
        name="Reinforcements",
        reserve_type="tactical",
        d20_min=5,
        d20_max=6,
        description="Call NPC mech ally once per mission (tier 1-3 NPC)",
    ),
    ReserveDefinition(
        id="tactical_env_shielding",
        name="Environmental Shielding",
        reserve_type="tactical",
        d20_min=7,
        d20_max=8,
        description="Ignore particular battlefield hazard or dangerous terrain",
    ),
    ReserveDefinition(
        id="tactical_accuracy",
        name="Accuracy (+1)",
        reserve_type="tactical",
        d20_min=9,
        d20_max=10,
        description="+1 accuracy on particular skill or action this mission",
    ),
    ReserveDefinition(
        id="tactical_bombardment",
        name="Bombardment",
        reserve_type="tactical",
        d20_min=11,
        d20_max=12,
        description="Call artillery/orbital bombardment (Full Action, range 30, blast 2, 3d6 explosive)",
    ),
    ReserveDefinition(
        id="tactical_extended_harness",
        name="Extended Harness",
        reserve_type="tactical",
        d20_min=13,
        d20_max=14,
        description="Carry extra pilot weapon + two pilot gear items",
    ),
    ReserveDefinition(
        id="tactical_ambush",
        name="Ambush",
        reserve_type="tactical",
        d20_min=15,
        d20_max=16,
        description="Choose where next battle takes place (terrain, cover setup)",
    ),
    ReserveDefinition(
        id="tactical_orbital_drop",
        name="Orbital Drop",
        reserve_type="tactical",
        d20_min=17,
        d20_max=18,
        description="Start mission dropping from orbit into fortified location",
    ),
    ReserveDefinition(
        id="tactical_nhp_assistant",
        name="NHP Assistant",
        reserve_type="tactical",
        d20_min=19,
        d20_max=20,
        description="Gain NHP (GM-controlled) that advises on situation",
    ),
]


ALL_RESERVES: list[ReserveDefinition] = (
    NARRATIVE_RESERVES + MECH_RESERVES + TACTICAL_RESERVES
)


RESERVES_BY_ID = {r.id: r for r in ALL_RESERVES}


def roll_for_reserve(
    d20_roll: int,
    reserve_type: ReserveType,
) -> ReserveDefinition | None:
    """Look up reserve by d20 roll and type."""
    for r in ALL_RESERVES:
        if r.reserve_type == reserve_type and r.d20_min <= d20_roll <= r.d20_max:
            return r
    return None


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
    roll_tier_outcomes: list[RollTierOutcome] = Field(
        default_factory=list,
        description="Roll-tier specific outcomes (9-, 10-19, 20+)",
    )
    requires_location: str | None = Field(
        default=None,
        description="Required location type (e.g., 'populated_area')",
    )


DOWNTIME_ACTION_DEFINITIONS: list[DowntimeActionDefinition] = [
    DowntimeActionDefinition(
        id="power_at_a_cost",
        name="Power at a Cost",
        category="tradeoff",
        outcomes=["reserve", "custom"],
        requires_skill_check=None,
        grants_reserve=True,
        roll_tier_outcomes=[
            RollTierOutcome(
                tier="success",
                description="You get what you want, but the GM chooses 1-2 tradeoffs",
                triggers_gm_choice=True,
                choice_options=[
                    "Takes more time than anticipated",
                    "Really damn risky",
                    "Must give up or leave something behind",
                    "Piss off someone powerful",
                    "Go wildly off the plan",
                    "Need more information to proceed safely",
                    "Result falls apart soon",
                    "Need to gather more resources first",
                    "Get lesser version of what you want",
                ],
            ),
        ],
    ),
    DowntimeActionDefinition(
        id="buy_some_time",
        name="Buy Some Time",
        category="recovery",
        outcomes=["reserve", "custom"],
        requires_skill_check=True,
        grants_reserve=True,
        roll_tier_outcomes=[
            RollTierOutcome(
                tier="failure",
                description="Buy little time, only if drastic measures taken now",
            ),
            RollTierOutcome(
                tier="partial",
                description="Buy enough time, but situation becomes precarious. Next time treat as failure.",
            ),
            RollTierOutcome(
                tier="success",
                description="Buy enough time until next mission. If repeated, becomes partial next time.",
            ),
        ],
    ),
    DowntimeActionDefinition(
        id="get_a_damn_drink",
        name="Get a Damn Drink",
        category="recovery",
        outcomes=["recovery"],
        requires_skill_check=True,
        requires_location="populated_area",
        roll_tier_outcomes=[
            RollTierOutcome(
                tier="failure",
                description="Wake in gutter with only ONE of: dignity, possessions, or memory",
                choice_options=["dignity", "possessions", "memory"],
            ),
            RollTierOutcome(
                tier="partial",
                description="Gain ONE of: reputation, friend/connection, useful item/info, opportunity. Lose ONE of the same.",
                grants_reserve=True,
            ),
            RollTierOutcome(
                tier="success",
                description="Gain TWO of: reputation, friend/connection, useful item/info, opportunity. Lose nothing.",
                grants_reserve=True,
            ),
        ],
    ),
    DowntimeActionDefinition(
        id="get_creative",
        name="Get Creative",
        category="project",
        outcomes=["project", "reserve"],
        requires_skill_check=True,
        grants_reserve=True,
        roll_tier_outcomes=[
            RollTierOutcome(
                tier="failure",
                description="No progress. If previous failure on same project, next is partial.",
            ),
            RollTierOutcome(
                tier="partial",
                description="Make progress but can't finish. Need 2 of: materials, knowledge, tools, workspace.",
                choice_options=[
                    "quality_materials",
                    "specific_knowledge",
                    "specialized_tools",
                    "good_workspace",
                ],
            ),
            RollTierOutcome(
                tier="success",
                description="Finish project. If complex, treat as partial with 1 requirement.",
            ),
        ],
    ),
    DowntimeActionDefinition(
        id="get_focused",
        name="Get Focused",
        category="recovery",
        outcomes=["custom"],
        requires_skill_check=False,
        roll_tier_outcomes=[
            RollTierOutcome(
                tier="success",
                description="Gain new trigger at +2. Can repeat up to +6.",
            ),
        ],
    ),
    DowntimeActionDefinition(
        id="get_organized",
        name="Get Organized",
        category="project",
        outcomes=["project", "reserve"],
        requires_skill_check=True,
        grants_reserve=True,
        roll_tier_outcomes=[
            RollTierOutcome(
                tier="failure",
                description="Organization folds unless: reduce by 2 OR take action (pay debts, prove worthiness, get bailed out, make aggressive move)",
                triggers_gm_choice=True,
                choice_options=[
                    "reduce_by_2",
                    "pay_debts",
                    "prove_worthiness",
                    "get_bailed_out",
                    "aggressive_move",
                ],
            ),
            RollTierOutcome(
                tier="partial",
                description="Organization stable. Gain +2 to efficiency OR influence (max 6).",
                grants_reserve=True,
            ),
            RollTierOutcome(
                tier="success",
                description="Organization gains +2 to BOTH efficiency and influence (max 6).",
                grants_reserve=True,
            ),
        ],
    ),
    DowntimeActionDefinition(
        id="gather_information",
        name="Gather Information",
        category="intel",
        outcomes=["info", "reserve"],
        requires_skill_check=True,
        grants_reserve=True,
        roll_tier_outcomes=[
            RollTierOutcome(
                tier="failure",
                description="Choose: leave now OR get info but immediately get into trouble.",
                triggers_gm_choice=True,
                choice_options=["leave_now", "get_info_with_trouble"],
            ),
            RollTierOutcome(
                tier="partial",
                description="Find info with complication: leave evidence OR dispatch someone/implicate innocent.",
                triggers_gm_choice=True,
                choice_options=["leave_evidence", "dispatch_someone"],
            ),
            RollTierOutcome(
                tier="success",
                description="Get info cleanly, no complications.",
            ),
        ],
    ),
    DowntimeActionDefinition(
        id="get_connected",
        name="Get Connected",
        category="contact",
        outcomes=["contact", "reserve"],
        requires_skill_check=True,
        grants_reserve=True,
        roll_tier_outcomes=[
            RollTierOutcome(
                tier="failure",
                description="Do favor immediately or connection won't help. If do favor now, they'll go along.",
            ),
            RollTierOutcome(
                tier="partial",
                description="Connection helps but favor owed later. If don't repay, next time treated as failure.",
            ),
            RollTierOutcome(
                tier="success",
                description="Connection helps with no strings attached. If repeated, becomes partial.",
            ),
        ],
    ),
    DowntimeActionDefinition(
        id="scrounge_and_barter",
        name="Scrounge and Barter",
        category="resource",
        outcomes=["reserve"],
        requires_skill_check=True,
        grants_reserve=True,
        roll_tier_outcomes=[
            RollTierOutcome(
                tier="failure",
                description="Get item with drawback: stolen, degraded, or held by someone.",
                triggers_gm_choice=True,
                choice_options=["stolen", "degraded", "held_by_someone"],
            ),
            RollTierOutcome(
                tier="partial",
                description="Get item by trading one cost: time, dignity, or reputation.",
                grants_reserve=True,
            ),
            RollTierOutcome(
                tier="success",
                description="Clean acquisition.",
                grants_reserve=True,
            ),
        ],
    ),
]


DOWNTIME_ACTIONS_BY_ID = {action.id: action for action in DOWNTIME_ACTION_DEFINITIONS}


def get_downtime_action_definition(action_id: str) -> DowntimeActionDefinition | None:
    """Look up a downtime action definition by ID."""
    return DOWNTIME_ACTIONS_BY_ID.get(action_id)


class DowntimeActionUse(FrozenModel):
    """A single downtime action taken by a pilot."""

    action_id: str
    outcome: DowntimeOutcomeType
    roll_result: int | None = Field(
        default=None, ge=1, le=20, description="d20 roll result"
    )
    roll_tier: RollTier | None = Field(
        default=None, description="Computed from roll_result"
    )
    reserve: ReserveEntry | None = None
    organization_state: OrganizationState | None = None
    trigger_grant: TriggerGrant | None = None

    def with_roll_tier(self) -> "DowntimeActionUse":
        """Return a copy with roll_tier computed from roll_result."""
        if self.roll_result is not None and self.roll_tier is None:
            tier = _get_roll_tier(self.roll_result)
            return DowntimeActionUse(
                action_id=self.action_id,
                outcome=self.outcome,
                roll_result=self.roll_result,
                roll_tier=tier,
                reserve=self.reserve,
                organization_state=self.organization_state,
                trigger_grant=self.trigger_grant,
            )
        return self


class DowntimePlan(FrozenModel):
    """Recorded downtime actions for a pilot."""

    pilot_id: str
    actions: list[DowntimeActionUse] = Field(default_factory=list)


class DowntimeIssue(FrozenModel):
    """A downtime validation issue."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


class DowntimeValidation(FrozenModel):
    """Validation result for downtime planning."""

    valid: bool
    issues: list[DowntimeIssue] = Field(default_factory=list)


def _get_roll_tier(roll_result: int) -> RollTier:
    """Determine roll tier from d20 result."""
    if roll_result <= 9:
        return "failure"
    elif roll_result <= 19:
        return "partial"
    else:
        return "success"


def validate_downtime_plan(
    plan: DowntimePlan,
    rules: MissionCadenceRules = DEFAULT_MISSION_CADENCE_RULES,
    action_definitions: dict[str, DowntimeActionDefinition] | None = None,
    is_long_session: bool = False,
) -> DowntimeValidation:
    """Validate a downtime plan against cadence rules and action definitions."""
    issues: list[DowntimeIssue] = []
    definitions = action_definitions or DOWNTIME_ACTIONS_BY_ID
    max_actions = (
        rules.downtime_actions_long_session
        if is_long_session
        else rules.downtime_actions_per_pilot
    )

    if len(plan.actions) > max_actions:
        issues.append(
            DowntimeIssue(
                code="too_many_downtime_actions",
                message=(
                    f"Downtime actions {len(plan.actions)} exceed "
                    f"allowed {max_actions}."
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

        if definition.roll_tier_outcomes:
            if action_use.roll_result is None:
                issues.append(
                    DowntimeIssue(
                        code="missing_roll_result",
                        message=f"{definition.name} requires a roll result.",
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

    return DowntimeValidation(
        valid=not any(i.severity == "error" for i in issues), issues=issues
    )
