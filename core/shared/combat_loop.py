"""Narrative combat loop primitives for Lancer TTRPG.

This module provides type-safe helpers for narrative combat flow including:
- Scene phase management (opening/action/complication/resolution)
- Initiative tracking (players always have initiative unless they stall)
- Scene timer and time pressure
- Victory condition checking

Per PR2 rules (3064-3071):
- Players always have initiative in narrative play
- NPCs don't take independent actions; respond to player rolls
- Scenes end when focus cuts away or activity naturally ends
"""

from __future__ import annotations

from typing import Literal, Any, TYPE_CHECKING
from pydantic import Field, model_validator, field_validator
from core.shared.models import FrozenModel

if TYPE_CHECKING:
    from core.shared.narrative import NarrativeCombatState, NarrativeGoalTracker


ScenePhase = Literal["opening", "action", "complication", "resolution"]

SceneScenarioType = Literal[
    "escort",
    "control",
    "extract",
    "hold_out",
    "gauntlet",
    "recon",
    "custom",
]

TimePressureLevel = Literal["none", "urgent", "critical"]


class CombatLoopState(FrozenModel):
    """State machine for narrative combat flow.

    Manages initiative, scene phases, and pacing per PR2 3064-3071.

    Attributes:
        scene_id: Optional scene identifier
        phase: Current scene phase
        round_number: Current round within the scene
        initiative_holder: Who currently has narrative initiative
        has_gm_turn: Whether GM has taken over (players stalled)
        scene_timer: Turns remaining if time-limited
        active_goals: Goal IDs currently being pursued
        active_complications: Complication IDs currently active
        turn_history: Ordered list of actors who have taken turns
    """

    scene_id: str | None = None
    phase: ScenePhase = "opening"
    round_number: int = Field(default=1, ge=1)
    initiative_holder: str | None = None
    has_gm_turn: bool = False
    scene_timer: int | None = Field(default=None, ge=1)
    active_goals: list[str] = Field(default_factory=list)
    active_complications: list[str] = Field(default_factory=list)
    turn_history: list[str] = Field(default_factory=list)
    scenario_type: SceneScenarioType | None = None
    scenario_settings: dict[str, str] | None = None

    @model_validator(mode="before")
    @classmethod
    def init_turn_history(cls, data: Any) -> Any:
        """Initialize turn_history from initiative_holder if needed."""
        if isinstance(data, dict):
            if data.get("initiative_holder") and not data.get("turn_history"):
                data["turn_history"] = [data["initiative_holder"]]
        return data


class InitiativeTracker(FrozenModel):
    """Tracks who has initiative in narrative play per PR2.

    Players always have initiative in narrative play unless they stall
    or pass responsibility to the GM.

    Attributes:
        current_initiative: Who currently has initiative
        initiative_queue: Ordered list for contested initiative
        last_player_roll: Most recent player roll for initiative
        gm_took_initiative: Whether GM has taken over this scene
        player_stall_count: How many times players have stalled
    """

    current_initiative: str | None = None
    initiative_queue: list[str] = Field(default_factory=list)
    last_player_roll: int | None = Field(default=None, ge=1, le=20)
    gm_took_initiative: bool = False
    player_stall_count: int = Field(default=0, ge=0)

    def transfer_to_gm(self) -> "InitiativeTracker":
        """GM takes initiative when players stall (PR2 3071).

        Players always have initiative unless they don't take action,
        stall, or pass off responsibility.
        """
        return self.model_copy(
            update={
                "gm_took_initiative": True,
                "current_initiative": None,
                "player_stall_count": self.player_stall_count + 1,
            }
        )

    def transfer_to_player(self, player_id: str, roll: int) -> "InitiativeTracker":
        """Player regains initiative after GM turn or new action."""
        return self.model_copy(
            update={
                "current_initiative": player_id,
                "gm_took_initiative": False,
                "last_player_roll": roll,
                "player_stall_count": 0,
            }
        )

    def player_stalls(self) -> "InitiativeTracker":
        """Record that players have stalled, potentially transferring initiative."""
        return self.transfer_to_gm()


class SceneTimer(FrozenModel):
    """Time tracking for scenes with optional pressure.

    Supports escalating pressure levels as time runs out.

    Attributes:
        total_turns: Total turns allocated for the scene
        remaining_turns: Turns remaining
        pressure_level: Current time pressure
        escalation_threshold: Turn number for pressure escalation
    """

    total_turns: int | None = Field(default=None, ge=1)
    remaining_turns: int | None = Field(default=None, ge=1)
    pressure_level: TimePressureLevel = "none"
    escalation_threshold: int | None = Field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def init_remaining_from_total(cls, data: Any) -> Any:
        """Initialize remaining_turns from total if not set."""
        if isinstance(data, dict):
            if (
                data.get("total_turns") is not None
                and data.get("remaining_turns") is None
            ):
                data["remaining_turns"] = data["total_turns"]
        return data

    def tick(self) -> "SceneTimer":
        """Advance scene timer by one turn.

        Escalates pressure as remaining turns decrease.
        """
        if self.remaining_turns is None or self.remaining_turns <= 0:
            return self

        remaining = self.remaining_turns - 1
        pressure = self.pressure_level

        if self.total_turns is not None:
            if remaining < self.total_turns // 3:
                pressure = "critical"
            elif remaining <= self.total_turns // 2:
                pressure = "urgent"

        return self.model_copy(
            update={
                "remaining_turns": remaining,
                "pressure_level": pressure,
            }
        )


class SceneMetrics(FrozenModel):
    """Tracks scene progress and pacing statistics.

    Attributes:
        turns_taken: Total turns in the scene
        goals_completed: Goals successfully achieved
        complications_survived: Complications resolved without failure
        complications_escalated: Complications that worsened
        time_elapsed: Abstract time units passed
    """

    turns_taken: int = Field(default=0, ge=0)
    goals_completed: int = Field(default=0, ge=0)
    complications_survived: int = Field(default=0, ge=0)
    complications_escalated: int = Field(default=0, ge=0)
    time_elapsed: int | None = Field(default=None, ge=0)


def advance_phase(
    state: CombatLoopState,
    trigger: Literal[
        "goal_announced", "action_taken", "complication_occurred", "goal_completed"
    ],
) -> CombatLoopState:
    """Advance scene phase based on narrative trigger.

    Phase progression follows the narrative flow:
    - opening -> action (goal announced or action taken)
    - action -> complication (complication occurs)
    - action -> resolution (goal completed)
    - complication -> action (action taken to resolve)
    - complication -> resolution (goal completed despite complication)

    Args:
        state: Current combat loop state
        trigger: What narrative event triggered the transition

    Returns:
        Updated state with new phase
    """
    phase_transitions: dict[str, dict[str, str]] = {
        "opening": {
            "goal_announced": "action",
            "action_taken": "action",
        },
        "action": {
            "complication_occurred": "complication",
            "goal_completed": "resolution",
        },
        "complication": {
            "action_taken": "action",
            "goal_completed": "resolution",
        },
        "resolution": {},
    }

    next_phase = phase_transitions.get(state.phase, {}).get(trigger, state.phase)
    return state.model_copy(update={"phase": next_phase})


def transfer_initiative(
    tracker: InitiativeTracker,
    to_actor: str,
    roll: int | None = None,
    is_gm: bool = False,
) -> InitiativeTracker:
    """Transfer narrative initiative to an actor.

    Args:
        tracker: Current initiative tracker
        to_actor: Actor receiving initiative
        roll: Optional d20 roll result (required for players)
        is_gm: Whether the recipient is the GM

    Returns:
        Updated initiative tracker
    """
    if is_gm:
        return tracker.transfer_to_gm()
    return tracker.transfer_to_player(to_actor, roll or 10)


def start_scene(
    scene_id: str,
    first_actor: str,
    scenario_type: SceneScenarioType | None = None,
    scene_timer: int | None = None,
) -> tuple[CombatLoopState, InitiativeTracker, SceneTimer | None]:
    """Initialize a new narrative scene.

    Args:
        scene_id: Unique scene identifier
        first_actor: First actor to have initiative
        scenario_type: Optional scenario type (escort, control, etc.)
        scene_timer: Optional turn limit for the scene

    Returns:
        Tuple of (combat_loop_state, initiative_tracker, scene_timer)
    """
    loop_state = CombatLoopState(
        scene_id=scene_id,
        phase="opening",
        initiative_holder=first_actor,
        scenario_type=scenario_type,
    )

    init_tracker = InitiativeTracker(current_initiative=first_actor)

    timer: SceneTimer | None = None
    if scene_timer:
        timer = SceneTimer(total_turns=scene_timer, remaining_turns=scene_timer)

    return loop_state, init_tracker, timer


def end_scene(
    state: CombatLoopState,
    completion_type: Literal[
        "goals_achieved", "time_expired", "mutual_withdrawal", "gm_cut"
    ],
) -> CombatLoopState:
    """End the current scene and transition to resolution.

    Args:
        state: Current combat loop state
        completion_type: How the scene ended

    Returns:
        Updated state in resolution phase
    """
    return state.model_copy(
        update={
            "phase": "resolution",
        }
    )


VictoryConditionType = Literal[
    "all_goals_complete",
    "time_elapsed",
    "all_enemies_defeated",
    "survival_complete",
    "custom",
]


class VictoryCondition(FrozenModel):
    """Condition that ends the scene/mission.

    Victory conditions define when a scene or mission is complete.
    Multiple conditions can be combined for complex objectives.

    Attributes:
        condition_type: Type of victory condition
        description: Human-readable description
        target_value: Numeric target (e.g., 6 rounds for time limit)
        completion_message: Message shown when condition is met
        is_optional: Whether this condition can be failed without mission failure
    """

    condition_type: VictoryConditionType
    description: str
    target_value: int | None = Field(default=None, ge=1)
    completion_message: str | None = None
    is_optional: bool = False


class MissionCompletionResult(FrozenModel):
    """Result when a mission or scene ends.

    Captures the outcome and relevant metrics for record-keeping.

    Attributes:
        victory: Whether the mission was successful
        completion_type: How the mission ended
        goals_achieved: Number of goals completed
        goals_total: Total number of goals
        complications_survived: Complications resolved
        complications_escalated: Complications that worsened
        partial_success: Whether criteria were partially met
        sequel_flag: Hint for continuation ("mission continues in...")
        completion_turn: Turn number when mission ended
    """

    victory: bool
    completion_type: str
    goals_achieved: int = Field(default=0, ge=0)
    goals_total: int = Field(default=0, ge=0)
    complications_survived: int = Field(default=0, ge=0)
    complications_escalated: int = Field(default=0, ge=0)
    partial_success: bool = False
    sequel_flag: str | None = None
    completion_turn: int = Field(default=1, ge=1)


class VictoryCheckResult(FrozenModel):
    """Intermediate result of checking victory conditions.

    Attributes:
        conditions_met: List of condition IDs that are satisfied
        conditions_pending: List of condition IDs not yet satisfied
        all_met: Whether all non-optional conditions are met
    """

    conditions_met: list[str] = Field(default_factory=list)
    conditions_pending: list[str] = Field(default_factory=list)
    all_met: bool = False


def check_victory_conditions(
    state: CombatLoopState,
    conditions: list[VictoryCondition],
    completed_goal_ids: list[str],
    active_hostile_ids: list[str],
) -> VictoryCheckResult:
    """Check if victory conditions are satisfied.

    Args:
        state: Current combat loop state
        conditions: List of victory conditions to check
        completed_goal_ids: IDs of goals that have been completed
        active_hostile_ids: IDs of still-active hostile combatants

    Returns:
        VictoryCheckResult with status of each condition
    """
    met: list[str] = []
    pending: list[str] = []

    for condition in conditions:
        is_met = False

        match condition.condition_type:
            case "all_goals_complete":
                is_met = all(g in completed_goal_ids for g in state.active_goals)

            case "time_elapsed":
                if state.round_number >= (condition.target_value or 6):
                    is_met = True

            case "all_enemies_defeated":
                is_met = len(active_hostile_ids) == 0

            case "survival_complete":
                is_met = state.round_number >= (condition.target_value or 1)

            case "custom":
                is_met = False

        if is_met:
            met.append(condition.condition_type)
        else:
            pending.append(condition.condition_type)

    optional_conditions = [c for c in conditions if c.is_optional]
    non_optional = [c for c in conditions if not c.is_optional]
    all_non_optional_met = all(c.condition_type in met for c in non_optional)

    return VictoryCheckResult(
        conditions_met=met,
        conditions_pending=pending,
        all_met=len(non_optional) == 0 or all_non_optional_met,
    )


def resolve_mission_completion(
    state: CombatLoopState,
    conditions: list[VictoryCondition],
    check_result: VictoryCheckResult,
    completed_goal_ids: list[str],
    complications_survived: int = 0,
    complications_escalated: int = 0,
) -> MissionCompletionResult:
    """Generate completion result when victory conditions are met.

    Args:
        state: Final combat loop state
        conditions: All victory conditions
        check_result: Result of victory condition check
        completed_goal_ids: Goals actually completed
        complications_survived: Complications resolved
        complications_escalated: Complications that worsened

    Returns:
        MissionCompletionResult with full outcome details
    """
    victory = check_result.all_met

    if victory:
        primary_condition = next(
            (c for c in conditions if c.condition_type in check_result.conditions_met),
            conditions[0] if conditions else None,
        )
        completion_message = (
            primary_condition.completion_message
            if primary_condition
            else "Mission accomplished"
        )
    else:
        completion_message = None

    goals_achieved_count = len(
        [g for g in completed_goal_ids if g in state.active_goals]
    )

    return MissionCompletionResult(
        victory=victory,
        completion_type=check_result.conditions_met[0]
        if check_result.conditions_met
        else "incomplete",
        goals_achieved=goals_achieved_count,
        goals_total=len(state.active_goals),
        complications_survived=complications_survived,
        complications_escalated=complications_escalated,
        partial_success=(
            len(check_result.conditions_met) > 0 or goals_achieved_count > 0
        )
        and not victory,
        sequel_flag=None,
        completion_turn=state.round_number,
    )


class NarrativeScenario(FrozenModel):
    """Combined narrative scenario state.

    Integrates combat loop flow with goal tracking for complete
    narrative play state management.

    Attributes:
        loop_state: Scene phase, initiative, and timing
        combat_state: Goals and complications tracking
    """

    loop_state: CombatLoopState
    combat_state: "NarrativeCombatState"


def start_narrative_scenario(
    scene_id: str,
    first_actor: str,
    scenario_type: SceneScenarioType | None = None,
    scene_timer: int | None = None,
) -> tuple[
    CombatLoopState, "NarrativeCombatState", InitiativeTracker, SceneTimer | None
]:
    """Initialize a complete narrative scenario.

    Creates all state needed for narrative play:
    - CombatLoopState for flow management
    - NarrativeCombatState for goals/complications
    - InitiativeTracker for narrative initiative
    - Optional SceneTimer for time pressure

    Args:
        scene_id: Unique scene identifier
        first_actor: First actor to have initiative
        scenario_type: Optional scenario type (escort, control, etc.)
        scene_timer: Optional turn limit for the scene

    Returns:
        Tuple of (loop_state, combat_state, initiative_tracker, scene_timer)
    """
    from core.shared.narrative import NarrativeCombatState, NarrativeGoalTracker

    loop_state = CombatLoopState(
        scene_id=scene_id,
        phase="opening",
        initiative_holder=first_actor,
        scenario_type=scenario_type,
    )

    combat_state = NarrativeCombatState(
        scene_id=scene_id,
        goal_tracker=NarrativeGoalTracker(),
    )

    init_tracker = InitiativeTracker(current_initiative=first_actor)

    timer: SceneTimer | None = None
    if scene_timer:
        timer = SceneTimer(total_turns=scene_timer, remaining_turns=scene_timer)

    return loop_state, combat_state, init_tracker, timer


def advance_narrative_turn(
    loop_state: CombatLoopState,
    actor_id: str,
    trigger: Literal[
        "goal_announced", "action_taken", "complication_occurred", "goal_completed"
    ],
) -> CombatLoopState:
    """Advance narrative scene after an actor takes action.

    Handles phase transitions and turn history tracking.

    Args:
        loop_state: Current combat loop state
        actor_id: Actor who just acted
        trigger: What narrative event occurred

    Returns:
        Updated combat loop state with new phase and history
    """
    updated = advance_phase(loop_state, trigger)

    if actor_id not in updated.turn_history:
        updated = updated.model_copy(
            update={"turn_history": [*updated.turn_history, actor_id]}
        )

    return updated


def check_narrative_victory(
    loop_state: CombatLoopState,
    combat_state: "NarrativeCombatState",
    active_hostiles: list[str],
    victory_conditions: list[VictoryCondition],
) -> tuple[bool, MissionCompletionResult | None]:
    """Check if narrative victory conditions are met.

    Args:
        loop_state: Current combat loop state
        combat_state: Goals and complications state
        active_hostiles: IDs of still-active hostile combatants
        victory_conditions: List of conditions to check

    Returns:
        Tuple of (is_victory, completion_result)
    """
    completed_goal_ids = [
        g.goal.id for g in combat_state.goal_tracker.goals if g.status == "completed"
    ]

    check_result = check_victory_conditions(
        loop_state,
        victory_conditions,
        completed_goal_ids,
        active_hostiles,
    )

    if check_result.all_met:
        result = resolve_mission_completion(
            loop_state,
            victory_conditions,
            check_result,
            completed_goal_ids,
            complications_survived=len(
                [c for c in combat_state.complications if c.status == "resolved"]
            ),
        )
        return True, result

    return False, None
