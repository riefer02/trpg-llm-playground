"""Scenario/Objective system for Lancer TTRPG mission tracking.

Provides type-safe primitives for:
- Mission objectives with dependencies and status tracking
- Mission state management (briefing, active, debrief)
- Simple progress scoring (completed/total percentage)
- Integration with combat loop for victory conditions

Per PR2 mission structure (1429-3203):
- Missions are discrete goals completed in finite time
- Brief establishes goals and stakes before mission
- Objectives can be multiple with dependencies
- Success = completing mission goal (partial success possible)
- Level up on any mission completion (success or failure)
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING, cast, Any
from pydantic import Field, model_validator
from core.shared.models import FrozenModel
from core.shared.id_helpers import CombatantIdField

if TYPE_CHECKING:
    pass


MissionObjectiveType = Literal[
    "escort",
    "defend",
    "extract",
    "destroy",
    "infiltrate",
    "investigate",
    "control",
    "custom",
]


ObjectiveStatus = Literal["pending", "in_progress", "blocked", "completed", "failed"]


ObjectiveCriterionType = Literal[
    "target_destroyed",
    "position_reached",
    "target_escorted",
    "area_secured",
    "intel_gathered",
    "time_elapsed",
    "custom",
]


MissionOutcomeType = Literal["success", "partial", "failure", "catastrophic"]


class ObjectiveCriterion(FrozenModel):
    """Criterion that must be met for objective completion."""

    criterion_type: ObjectiveCriterionType
    description: str = Field(..., description="What must be true for this criterion")
    target_id: CombatantIdField | None = Field(
        default=None,
        description="Target ID for target-based criteria",
    )
    required_amount: int | None = Field(
        default=None,
        ge=1,
        description="Quantity needed for amount-based criteria",
    )


class MissionObjective(FrozenModel):
    """Single mission objective with dependencies and completion criteria.

    Attributes:
        id: Unique identifier for this objective
        description: What this objective requires
        objective_type: Type of mission objective
        status: Current status of this objective
        priority: Higher = more important (used for scoring)
        depends_on: List of objective IDs that must complete first
        completion_criteria: List of criteria that define completion
        is_optional: If True, can fail without mission failure
    """

    id: str = Field(..., description="Unique identifier")
    description: str = Field(..., description="What this objective requires")
    objective_type: MissionObjectiveType
    status: ObjectiveStatus = Field(default="pending")
    priority: int = Field(default=1, ge=1, description="Higher = more important")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Objective IDs that must complete first",
    )
    completion_criteria: list[ObjectiveCriterion] = Field(
        default_factory=list,
        description="Criteria that define when objective is complete",
    )
    is_optional: bool = Field(
        default=False,
        description="If True, can fail without mission failure",
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "MissionObjective":
        if self.id in self.depends_on:
            raise ValueError(f"Objective cannot depend on itself: {self.id}")
        return self


class MissionObjectiveState(FrozenModel):
    """Tracks the current state of a mission objective.

    Attributes:
        objective: The objective definition
        criteria_met: IDs of completion criteria that have been met
        attempts: Number of attempts made toward this objective
    """

    objective: MissionObjective
    criteria_met: list[str] = Field(default_factory=list)
    attempts: int = Field(default=0, ge=0)

    @property
    def is_complete(self) -> bool:
        """Check if all completion criteria are met."""
        if not self.objective.completion_criteria:
            return self.objective.status == "completed"
        return all(
            c.criterion_type == "custom" or c.description in self.criteria_met
            for c in self.objective.completion_criteria
        )

    @property
    def completion_percentage(self) -> float:
        """Calculate percentage of criteria met."""
        criteria = self.objective.completion_criteria
        if not criteria:
            return 1.0 if self.is_complete else 0.0
        return len(self.criteria_met) / len(criteria) if criteria else 0.0


class MissionStakes(FrozenModel):
    """Mission stakes per PR2 rules (2835-2841).

    Attributes:
        stakes_type: Type of stakes (personal, faction, immediate, gradual)
        description: Human-readable description of stakes
        consequences_success: What happens on success
        consequences_failure: What happens on failure
        consequences_partial: What happens on partial success
    """

    stakes_type: Literal["personal", "faction", "immediate", "gradual"]
    description: str = Field(..., description="Description of stakes")
    consequences_success: str | None = Field(default=None)
    consequences_failure: str | None = Field(default=None)
    consequences_partial: str | None = Field(default=None)


class Mission(FrozenModel):
    """Complete mission definition.

    Attributes:
        id: Unique mission identifier
        name: Display name
        description: Mission summary
        objectives: List of mission objectives
        stakes: Mission stakes (can be None for low-stakes missions)
        scenario_type: Optional combat scenario type for combat integration
        time_limit: Optional turn limit (None = no limit)
        is_critical: If True, mission cannot be failed/abandoned
    """

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Mission summary")
    objectives: list[MissionObjective] = Field(
        default_factory=list,
        description="Mission objectives",
    )
    stakes: MissionStakes | None = Field(default=None)
    scenario_type: (
        Literal["escort", "control", "extract", "hold_out", "gauntlet", "recon"] | None
    ) = Field(
        default=None,
    )
    time_limit: int | None = Field(default=None, ge=1)
    is_critical: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_objective_ids(self) -> "Mission":
        objective_ids = {obj.id for obj in self.objectives}
        for obj in self.objectives:
            for dep in obj.depends_on:
                if dep not in objective_ids:
                    raise ValueError(
                        f"Objective {obj.id} depends on non-existent objective: {dep}"
                    )
        return self


MissionPhase = Literal["briefing", "active", "debrief"]


class MissionState(FrozenModel):
    """Active mission state tracking.

    Attributes:
        mission: The mission definition
        objective_states: Map of objective ID to current state
        current_phase: Current mission phase
        current_turn: Current turn number
        completion_score: 0.0-1.0 progress score
        is_victory: None = ongoing, True = success, False = failure
    """

    mission: Mission
    objective_states: dict[str, MissionObjectiveState] = Field(
        default_factory=dict,
    )
    current_phase: MissionPhase = Field(default="briefing")
    current_turn: int = Field(default=1, ge=1)
    completion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_victory: bool | None = Field(default=None)

    def __init__(self, **data: object) -> None:
        if "objective_states" not in data or not data["objective_states"]:
            mission = data.get("mission")
            if mission:
                data["objective_states"] = {
                    obj.id: MissionObjectiveState(objective=obj)
                    for obj in mission.objectives
                }
        super().__init__(**data)

    def get_objective_state(self, objective_id: str) -> MissionObjectiveState | None:
        """Get the state for a specific objective."""
        return self.objective_states.get(objective_id)

    def get_required_objectives(self) -> list[MissionObjective]:
        """Get objectives that have unmet dependencies."""
        completed_ids = {
            oid
            for oid, state in self.objective_states.items()
            if state.objective.status == "completed"
        }
        return [
            obj
            for obj in self.mission.objectives
            if obj.status != "completed"
            and all(dep in completed_ids for dep in obj.depends_on)
        ]

    def get_blocked_objectives(self) -> list[MissionObjective]:
        """Get objectives with unmet dependencies."""
        completed_ids = {
            oid
            for oid, state in self.objective_states.items()
            if state.objective.status == "completed"
        }
        return [
            obj
            for obj in self.mission.objectives
            if obj.status != "completed"
            and not all(dep in completed_ids for dep in obj.depends_on)
        ]


class MissionBriefing(FrozenModel):
    """Pre-mission briefing content.

    Attributes:
        mission_id: Reference to mission
        objectives_revealed: Objectives shown to players
        stakes_revealed: Stakes info shared with players
        reserves_available: Reserve IDs available for this mission
        preparation_turns: Turns of prep time before mission
    """

    mission_id: str
    objectives_revealed: list[str] = Field(
        default_factory=list,
        description="Objective IDs revealed to players",
    )
    stakes_revealed: bool = Field(default=True)
    reserves_available: list[str] = Field(
        default_factory=list,
        description="Reserve IDs available",
    )
    preparation_turns: int = Field(default=1, ge=1)


class MissionDebrief(FrozenModel):
    """Post-mission debrief summary.

    Attributes:
        mission_id: Reference to mission
        outcome: Mission outcome type
        objectives_completed: IDs of completed objectives
        objectives_failed: IDs of failed objectives
        objectives_blocked: IDs of objectives never attempted
        completion_score: 0.0-1.0 progress score
        experience_gained: License levels earned (1 per PR2)
        reserves_granted: Reserve IDs granted as reward
        reserves_spent: Reserve IDs consumed during mission
        rewards_narrative: Narrative description of rewards
    """

    mission_id: str
    outcome: MissionOutcomeType
    objectives_completed: list[str] = Field(default_factory=list)
    objectives_failed: list[str] = Field(default_factory=list)
    objectives_blocked: list[str] = Field(default_factory=list)
    completion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    experience_gained: int = Field(default=1, ge=0, description="License levels gained")
    reserves_granted: list[str] = Field(default_factory=list)
    reserves_spent: list[str] = Field(default_factory=list)
    rewards_narrative: str | None = Field(default=None)


def calculate_mission_progress(mission_state: MissionState) -> float:
    """Calculate mission progress as percentage of objectives completed.

    Simple scoring: (completed_count / total_count) for non-optional objectives.
    Optional objectives don't affect the score.

    Args:
        mission_state: Current mission state

    Returns:
        Progress percentage between 0.0 and 1.0
    """
    objectives = mission_state.mission.objectives
    non_optional = [obj for obj in objectives if not obj.is_optional]

    if not non_optional:
        return 1.0

    completed = sum(
        1
        for obj in non_optional
        if mission_state.objective_states.get(
            obj.id, MissionObjectiveState(objective=obj)
        ).is_complete
    )

    return completed / len(non_optional)


def check_objective_prerequisites(
    mission_state: MissionState,
    objective_id: str,
) -> tuple[bool, list[str]]:
    """Check if an objective's dependencies are met.

    Args:
        mission_state: Current mission state
        objective_id: ID of objective to check

    Returns:
        Tuple of (all_met: bool, unmet_dependencies: list[str])
    """
    objective = None
    for obj in mission_state.mission.objectives:
        if obj.id == objective_id:
            objective = obj
            break

    if objective is None:
        return False, [f"Unknown objective: {objective_id}"]

    if not objective.depends_on:
        return True, []

    completed_ids = {
        oid
        for oid, state in mission_state.objective_states.items()
        if state.objective.status == "completed"
    }

    unmet = [dep for dep in objective.depends_on if dep not in completed_ids]
    return len(unmet) == 0, unmet


def update_objective_status(
    mission_state: MissionState,
    objective_id: str,
    new_status: ObjectiveStatus,
) -> MissionState:
    """Update the status of an objective.

    Args:
        mission_state: Current mission state
        objective_id: ID of objective to update
        new_status: New status to set

    Returns:
        Updated mission state
    """
    if objective_id not in mission_state.objective_states:
        raise ValueError(f"Unknown objective: {objective_id}")

    objective_state = mission_state.objective_states[objective_id]
    updated_objective = objective_state.objective.model_copy(
        update={"status": new_status}
    )
    updated_state = objective_state.model_copy(update={"objective": updated_objective})

    updated_objective_states = {
        **mission_state.objective_states,
        objective_id: updated_state,
    }
    new_completion_score = calculate_mission_progress(
        mission_state.model_copy(update={"objective_states": updated_objective_states})
    )

    return mission_state.model_copy(
        update={
            "objective_states": updated_objective_states,
            "completion_score": new_completion_score,
        }
    )


def mark_objective_criterion_met(
    mission_state: MissionState,
    objective_id: str,
    criterion_description: str,
) -> MissionState:
    """Mark a completion criterion as met for an objective.

    Args:
        mission_state: Current mission state
        objective_id: ID of objective
        criterion_description: Description of criterion to mark complete

    Returns:
        Updated mission state
    """
    if objective_id not in mission_state.objective_states:
        raise ValueError(f"Unknown objective: {objective_id}")

    objective_state = mission_state.objective_states[objective_id]
    criteria_met = list(objective_state.criteria_met)

    if criterion_description not in criteria_met:
        criteria_met.append(criterion_description)

    updated_state = objective_state.model_copy(update={"criteria_met": criteria_met})

    updated_objective_states = {
        **mission_state.objective_states,
        objective_id: updated_state,
    }
    new_completion_score = calculate_mission_progress(
        mission_state.model_copy(update={"objective_states": updated_objective_states})
    )

    return mission_state.model_copy(
        update={
            "objective_states": updated_objective_states,
            "completion_score": new_completion_score,
        }
    )


def check_mission_completion(mission_state: MissionState) -> bool:
    """Check if the mission should end.

    Mission ends when:
    - All non-optional objectives are complete (victory)
    - Time limit is reached and objectives incomplete (partial/failure)
    - All objectives failed (catastrophic)

    Args:
        mission_state: Current mission state

    Returns:
        True if mission should end
    """
    objectives = mission_state.mission.objectives
    non_optional = [obj for obj in objectives if not obj.is_optional]

    if not non_optional:
        return True

    all_complete = all(
        mission_state.objective_states.get(
            obj.id, MissionObjectiveState(objective=obj)
        ).is_complete
        for obj in non_optional
    )

    if all_complete:
        return True

    if (
        mission_state.mission.time_limit
        and mission_state.current_turn >= mission_state.mission.time_limit
    ):
        return True

    return False


def resolve_mission_outcome(mission_state: MissionState) -> MissionDebrief:
    """Generate the final mission debrief.

    Args:
        mission_state: Final mission state

    Returns:
        Complete mission debrief
    """
    objectives = mission_state.mission.objectives
    non_optional = [obj for obj in objectives if not obj.is_optional]

    completed = []
    failed = []
    blocked = []

    for obj in objectives:
        state = mission_state.objective_states.get(obj.id)
        if state and state.is_complete:
            completed.append(obj.id)
        elif state and state.objective.status == "failed":
            failed.append(obj.id)
        else:
            blocked.append(obj.id)

    if not non_optional:
        score = 1.0
    else:
        score = mission_state.completion_score

    if not non_optional:
        outcome: MissionOutcomeType = "success"
    elif score >= 1.0:
        outcome = "success"
    elif score >= 0.5:
        outcome = "partial"
    elif score > 0.0 or len(failed) > 0:
        outcome = "failure"
    else:
        outcome = "catastrophic"

    return MissionDebrief(
        mission_id=mission_state.mission.id,
        outcome=outcome,
        objectives_completed=completed,
        objectives_failed=failed,
        objectives_blocked=blocked,
        completion_score=score,
    )


def start_mission(mission: Mission) -> MissionState:
    """Initialize a mission state from a mission definition.

    Args:
        mission: Mission definition

    Returns:
        Initialized mission state ready for briefing phase
    """
    return MissionState(
        mission=mission,
        current_phase="briefing",
        current_turn=1,
    )


def advance_mission_phase(
    mission_state: MissionState,
    new_phase: MissionPhase,
) -> MissionState:
    """Advance the mission to a new phase.

    Args:
        mission_state: Current mission state
        new_phase: Phase to advance to

    Returns:
        Updated mission state
    """
    phase_order = ["briefing", "active", "debrief"]
    current_idx = phase_order.index(mission_state.current_phase)
    new_idx = phase_order.index(new_phase)

    if new_idx < current_idx:
        raise ValueError(
            f"Cannot regress from {mission_state.current_phase} to {new_phase}"
        )

    return mission_state.model_copy(update={"current_phase": new_phase})


def advance_mission_turn(mission_state: MissionState) -> MissionState:
    """Advance the mission by one turn.

    Args:
        mission_state: Current mission state

    Returns:
        Updated mission state with incremented turn
    """
    if mission_state.current_phase != "active":
        raise ValueError("Can only advance turns during active phase")

    return mission_state.model_copy(
        update={"current_turn": mission_state.current_turn + 1}
    )


# ============================================================================
# SITREP Mission Type Templates (Priority 48)
# ============================================================================

SitrepZoneType = Literal["deployment", "extraction", "objective", "ingress"]

SitrepDeploymentType = Literal["players_first", "enemies_first", "roll_off"]

SitrepReservePattern = Literal["none", "half", "normal", "double", "increasing"]

SitrepType = Literal["escort", "control", "extract", "hold_out", "gauntlet", "recon"]

SitrepVictoryOutcome = Literal["players_win", "enemies_win", "draw", "ongoing"]

ZoneControlState = Literal[
    "player_controlled", "enemy_controlled", "contested", "neutral"
]

ReserveSpawnCause = Literal["round_start", "turn_end", "event_triggered", "manual"]

VictoryConditionType = Literal[
    "extract_objective",
    "control_zones",
    "score_above_threshold",
    "outnumber_enemies",
    "control_real_objective",
    "survive_rounds",
]


class SitrepZone(FrozenModel):
    """Generic zone for SITREP configurations per PR2 12550-12565.

    Attributes:
        zone_type: Type of zone (deployment, extraction, objective, ingress)
        width: Width for rectangular zones (e.g., 4 for 4x4 control zones)
        height: Height for rectangular zones
        location: Descriptive location (e.g., "north_edge", "quadrant_nw")
        terrain_notes: Notes about terrain or special properties
    """

    zone_type: SitrepZoneType
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    location: str | None = None
    terrain_notes: str | None = None


class VictoryCondition(FrozenModel):
    """Structured victory condition for validation per PR2 rules.

    Attributes:
        condition_type: Type of victory condition
        threshold: Optional numeric threshold (e.g., zones to control)
        description: Human-readable description of the condition
    """

    condition_type: VictoryConditionType
    threshold: int | None = Field(default=None, ge=0)
    description: str = Field(..., description="Human-readable description")


class ZoneControlStateTracker(FrozenModel):
    zone_id: str
    state: ZoneControlState = Field(default="neutral")
    controlling_side: str | None = Field(default=None)
    last_checked_turn: int = Field(default=1, ge=1)


class SitrepVictoryCondition(FrozenModel):
    condition_type: VictoryConditionType
    target_value: int | None = Field(default=None, ge=1)
    current_value: int = Field(default=0, ge=0)
    is_met: bool = Field(default=False)
    description: str = Field(..., description="Human-readable description")


class SitrepDeployment(FrozenModel):
    player_zones: list[SitrepZone] = Field(default_factory=list)
    enemy_zones: list[SitrepZone] = Field(default_factory=list)
    ingress_zones: list[SitrepZone] = Field(default_factory=list)
    reserve_pool: list[str] = Field(default_factory=list)
    reserves_spawned_per_round: int | None = Field(default=None, ge=1)
    last_ingress_zone: str | None = Field(default=None)


class SitrepResolution(FrozenModel):
    template_type: SitrepType
    current_round: int = Field(default=1, ge=1)
    max_rounds: int = Field(default=6, ge=1)
    player_score: int = Field(default=0, ge=0)
    enemy_score: int = Field(default=0, ge=0)
    zone_states: dict[str, ZoneControlStateTracker] = Field(default_factory=dict)
    victory_conditions: list[SitrepVictoryCondition] = Field(default_factory=list)
    extraction_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    surviving_players: int = Field(default=0, ge=0)
    surviving_enemies: int = Field(default=0, ge=0)
    outcome: SitrepVictoryOutcome | None = Field(default=None)
    turn_limit_reached: bool = Field(default=False)
    reserves_remaining: int = Field(default=0, ge=0)
    deployment: SitrepDeployment | None = Field(default=None)


class SitrepTemplate(FrozenModel):
    """Base template for SITREP scenario types per PR2 12537-12762.

    Templates define mechanical parameters for different mission types.
    They are reusable starting points for creating missions.

    Attributes:
        sitrep_type: SITREP type identifier (escort, control, extract, etc.)
        name: Display name (ESCORT, CONTROL, etc.)
        description: Brief description of the mission type
        duration_rounds: Default mission duration in rounds (usually 6)
        deployment_type: Who deploys first or how deployment is determined
        deployment_zones: Where player characters deploy
        extraction_zone: Where objectives are extracted (escort/extract)
        objective_zones: Zones that are objectives (control, holdout, etc.)
        ingress_zones: Where enemy reinforcements can enter
        reserve_pattern: How many reserves the enemy has
        reserve_per_round: How many reserves enter per round
        victory_conditions: List of conditions for mission victory
        special_rules: Additional rules specific to this mission type
    """

    sitrep_type: SitrepType = Field(..., description="SITREP type identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Mission type description")
    duration_rounds: int = Field(default=6, ge=1)
    deployment_type: SitrepDeploymentType = Field(default="players_first")
    deployment_zones: list[SitrepZone] = Field(default_factory=list)
    extraction_zone: SitrepZone | None = None
    objective_zones: list[SitrepZone] = Field(default_factory=list)
    ingress_zones: list[SitrepZone] = Field(default_factory=list)
    reserve_pattern: SitrepReservePattern = Field(default="normal")
    reserve_per_round: int | None = Field(default=None, ge=1)
    victory_conditions: list[VictoryCondition] = Field(default_factory=list)
    special_rules: list[str] = Field(default_factory=list)


ESCORT_TEMPLATE = SitrepTemplate(
    sitrep_type="escort",
    name="ESCORT",
    description="Bring an objective safely to extraction zone. "
    "Per PR2 12602-12696: Players must extract the objective and themselves.",
    duration_rounds=6,
    deployment_type="players_first",
    deployment_zones=[SitrepZone(zone_type="deployment", location="map_edge")],
    extraction_zone=SitrepZone(zone_type="extraction", location="opposite_edge"),
    objective_zones=[SitrepZone(zone_type="objective", location="center")],
    ingress_zones=[
        SitrepZone(zone_type="ingress", location="left_flank"),
        SitrepZone(zone_type="ingress", location="right_flank"),
    ],
    reserve_pattern="increasing",
    reserve_per_round=1,
    victory_conditions=[
        VictoryCondition(
            condition_type="extract_objective",
            description="Extract the objective to the extraction zone",
        )
    ],
    special_rules=[
        "Objective has 10 HP/size, evasion 10, e-defense 10, no armor",
        "Enemy forces do not willingly damage the objective",
        "Players deploy first, then enemies in deployment zone",
        "Enemy can deploy 1 NPC or 4 grunts per round in ingress zones",
        "Cannot use same ingress zone twice in a row",
        "Free action to extract when in extraction zone",
        "At end of round 6, un-extracted objective means enemy victory",
    ],
)


CONTROL_TEMPLATE = SitrepTemplate(
    sitrep_type="control",
    name="CONTROL",
    description="Control 4 zones for 6 rounds to score more points than enemies. "
    "Per PR2 12697-12717: Maintain control of transmission towers, terminals, or hangars.",
    duration_rounds=6,
    deployment_type="roll_off",
    deployment_zones=[SitrepZone(zone_type="deployment", location="quadrant")],
    objective_zones=[
        SitrepZone(zone_type="objective", width=4, height=4, location="quadrant_nw"),
        SitrepZone(zone_type="objective", width=4, height=4, location="quadrant_ne"),
        SitrepZone(zone_type="objective", width=4, height=4, location="quadrant_sw"),
        SitrepZone(zone_type="objective", width=4, height=4, location="quadrant_se"),
    ],
    ingress_zones=[SitrepZone(zone_type="ingress", location="map_perimeter")],
    reserve_pattern="normal",
    victory_conditions=[
        VictoryCondition(
            condition_type="control_zones",
            threshold=3,
            description="Control 3 or more zones at end of round 6",
        )
    ],
    special_rules=[
        "4 objective zones placed in map quadrants",
        "Zone controlled if only one side's actors are inside",
        "Zone contested if actors from multiple sides are inside",
        "Score 1 point per controlled zone at end of each round",
        "Bonus +1 point if all 4 zones controlled at end of round",
        "Side with highest score at end of round 6 wins",
    ],
)


EXTRACT_TEMPLATE = SitrepTemplate(
    sitrep_type="extract",
    name="EXTRACT",
    description="Extract an objective from hostile territory. "
    "Per PR2 12717-12755: Similar to escort but enemies held in reserve initially.",
    duration_rounds=8,
    deployment_type="players_first",
    deployment_zones=[SitrepZone(zone_type="deployment", location="map_edge")],
    extraction_zone=SitrepZone(zone_type="extraction", location="same_as_deployment"),
    objective_zones=[SitrepZone(zone_type="objective", location="opposite_edge")],
    ingress_zones=[
        SitrepZone(zone_type="ingress", location="left_flank"),
        SitrepZone(zone_type="ingress", location="right_flank"),
    ],
    reserve_pattern="double",
    reserve_per_round=2,
    victory_conditions=[
        VictoryCondition(
            condition_type="extract_objective",
            description="Extract the objective to the extraction zone",
        )
    ],
    special_rules=[
        "Objective has 10 HP/size, evasion 10, e-defense 10, no armor",
        "Round 1: no enemy forces (all in reserve)",
        "Enemy can deploy 2 NPCs or 4 grunts per round in ingress zones",
        "Extraction zone is the same as deployment zone",
        "Free action to extract when in extraction zone",
        "At end of round 8, un-extracted objective means enemy victory",
    ],
)


HOLDOUT_TEMPLATE = SitrepTemplate(
    sitrep_type="hold_out",
    name="HOLDOUT",
    description="Defend an area against overwhelming odds. "
    "Per PR2 12755-12781: Buy time for allies or fight to the death.",
    duration_rounds=6,
    deployment_type="players_first",
    deployment_zones=[SitrepZone(zone_type="deployment", location="center")],
    objective_zones=[
        SitrepZone(zone_type="objective", width=10, height=5, location="center")
    ],
    ingress_zones=[SitrepZone(zone_type="ingress", location="map_perimeter")],
    reserve_pattern="half",
    victory_conditions=[
        VictoryCondition(
            condition_type="score_above_threshold",
            threshold=1,
            description="Maintain score of 1 or greater at end of round 6",
        )
    ],
    special_rules=[
        "Objective zone is approximately 10 spaces by 5 spaces",
        "Area around objective should have size 1-2 hard cover",
        "Players start with 4 points",
        "-1 point for each enemy inside the objective zone",
        "Score can go negative",
        "Enemy holds half forces in reserve",
        "Players deploy first, then GM deploys half their forces",
    ],
)


GAUNTLET_TEMPLATE = SitrepTemplate(
    sitrep_type="gauntlet",
    name="GAUNTLET",
    description="Push through hostile territory to secure an enemy position. "
    "Per PR2 12781-12800: Mission done under duress through unfriendly terrain.",
    duration_rounds=6,
    deployment_type="enemies_first",
    deployment_zones=[SitrepZone(zone_type="deployment", location="map_edge")],
    objective_zones=[
        SitrepZone(
            zone_type="objective",
            location="opposite_edge",
            terrain_notes="fortified",
        )
    ],
    ingress_zones=[SitrepZone(zone_type="ingress", location="map_perimeter")],
    reserve_pattern="half",
    reserve_per_round=None,
    victory_conditions=[
        VictoryCondition(
            condition_type="outnumber_enemies",
            description="More player characters than enemy characters in objective zone at end of round 6",
        )
    ],
    special_rules=[
        "Enemy deploys half their forces first",
        "Players deploy second",
        "Area around enemy deployment zone has size 1 and 2 hard cover",
        "Enemy holds half forces in reserve until end of round 1",
        "Count ultras as 4 players, elites as 2, grunts as 1/4",
        "More PCs than enemies in objective zone = player victory",
    ],
)


RECON_TEMPLATE = SitrepTemplate(
    sitrep_type="recon",
    name="RECON",
    description="Identify and control the real objective zone. "
    "Per PR2 12800-12860: Dangerous reconnaissance mission to identify targets.",
    duration_rounds=6,
    deployment_type="players_first",
    deployment_zones=[SitrepZone(zone_type="deployment", location="map_edge")],
    objective_zones=[
        SitrepZone(
            zone_type="objective",
            width=4,
            height=4,
            location="quadrant_nw",
            terrain_notes="may_be_real",
        ),
        SitrepZone(
            zone_type="objective",
            width=4,
            height=4,
            location="quadrant_ne",
            terrain_notes="may_be_real",
        ),
        SitrepZone(
            zone_type="objective",
            width=4,
            height=4,
            location="quadrant_sw",
            terrain_notes="may_be_real",
        ),
        SitrepZone(
            zone_type="objective",
            width=4,
            height=4,
            location="quadrant_se",
            terrain_notes="may_be_real",
        ),
    ],
    ingress_zones=[SitrepZone(zone_type="ingress", location="map_perimeter")],
    reserve_pattern="normal",
    reserve_per_round=1,
    victory_conditions=[
        VictoryCondition(
            condition_type="control_real_objective",
            description="Control the real objective zone at end of round 6",
        )
    ],
    special_rules=[
        "GM secretly chooses one objective as the real objective",
        "Full action to investigate if an objective is real (information shareable)",
        "Do not need to control zone to investigate it",
        "Control zone by being the only side inside at end of round",
        "Enemy has normal forces, can hold any number in reserve",
        "Enemy can deploy 1 NPC or 4 grunts per round in ingress zone",
    ],
)


SITREP_TEMPLATES: dict[str, SitrepTemplate] = {
    "escort": ESCORT_TEMPLATE,
    "control": CONTROL_TEMPLATE,
    "extract": EXTRACT_TEMPLATE,
    "hold_out": HOLDOUT_TEMPLATE,
    "gauntlet": GAUNTLET_TEMPLATE,
    "recon": RECON_TEMPLATE,
}


def get_sitrep_template(sitrep_type: str) -> SitrepTemplate | None:
    """Lookup a SITREP template by type.

    Args:
        sitrep_type: The SITREP type identifier (e.g., "escort", "control")

    Returns:
        The matching SitrepTemplate, or None if not found
    """
    return SITREP_TEMPLATES.get(sitrep_type)


def create_mission_from_template(
    template: SitrepTemplate,
    mission_id: str,
    name: str,
    description: str,
    **template_overrides: Any,
) -> Mission:
    """Create a Mission from a SITREP template with customization.

    Args:
        template: The SITREP template to base the mission on
        mission_id: Unique identifier for the new mission
        name: Display name for the mission
        description: Description of the mission
        **template_overrides: Any fields to override in the template

    Returns:
        A new Mission configured according to the template
    """
    merged_template = template.model_copy(update=template_overrides)

    objectives: list[MissionObjective] = []
    for i, zone in enumerate(merged_template.objective_zones):
        objectives.append(
            MissionObjective(
                id=f"obj_{i + 1}",
                description=f"Control or interact with {zone.location or f'objective zone {i + 1}'}",
                objective_type="control"
                if len(merged_template.objective_zones) > 1
                else "custom",
            )
        )

    mission = Mission(
        id=mission_id,
        name=name,
        description=description,
        objectives=objectives,
        scenario_type=cast(
            Literal["escort", "control", "extract", "hold_out", "gauntlet", "recon"]
            | None,
            merged_template.sitrep_type,
        ),
        time_limit=merged_template.duration_rounds,
    )

    return mission
