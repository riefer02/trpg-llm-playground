"""Phased tactical combat tracker for Lancer mech combat.

Unified tracker combining turn order management with phase-level action tracking:
- Nomination-based turn order from TacticalInitiativeTracker (PR2 3703-3725)
- Phase tracking per actor (start/normal/end) per PR2 3726-4406
- Protocol timing enforcement (only at start of turn)
- Prepared action lockout (expires at start of next turn)
- Per-round reaction tracking (resets at round boundary)

This module bridges the tactical initiative system with the existing timing module's
validation functions (validate_protocol_timing, validate_action_while_prepared, etc.).
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field

from core.shared.models import FrozenModel

from core.shared.combat.tactical_initiative import (
    TacticalInitiativeTracker,
    CombatSide,
    start_tactical_combat,
)
from core.mech.combat_rules import TurnOrderRules
from core.mech.timing import (
    TurnPhase,
    PreparedActionState,
    TurnPhaseState,
    ActionTimingValidationSettings,
    TimingValidationResult,
    validate_protocol_timing,
    validate_action_while_prepared,
    validate_per_round_reaction,
    DEFAULT_TIMING_VALIDATION,
)
from core.mech.action_economy import (
    ActionEconomyState,
    ActionEconomyResult,
    use_full_action as use_full_action_economy,
    use_quick_action as use_quick_action_economy,
    use_overcharge as use_overcharge_economy,
    use_reaction as use_reaction_economy,
    validate_action_economy,
    reset_economy_for_new_turn,
    get_action_economy_summary,
)
from core.shared.enums import ActionType


class PhasedTacticalTracker(FrozenModel):
    """Unified tracker for tactical combat with phase-level action tracking.

    Combines TacticalInitiativeTracker (turn order, nomination, alternation)
    with per-actor phase management (start/normal/end phases).

    Per PR2 3726-4406:
    - On a turn: move + (2 quick OR 1 full) + any free actions + any reactions
    - Protocols: only at start of turn (line 4406)
    - Prepared actions: block all actions until start of next turn (line 4347)
    - Reactions: 1/turn, any number per round, reset at round boundary (lines 4379-4384)

    Attributes:
        turn_order_rules: Configuration for turn order behavior
        current_actor_id: Actor currently taking their turn
        current_side: Side of current actor ("players" or "hostiles")
        round_index: Current round number (starts at 1)
        actors_who_acted_this_round: Track who has acted this round {actor_id: True}
        all_combatants: All combatants in combat {actor_id: side}
        last_actor_id: Previous actor (for nomination validation)
        actor_priorities: Priority overrides {actor_id: priority_value}
        actor_phases: Current phase per actor {actor_id: TurnPhase}
        actor_prepared_actions: Prepared action state per actor
        actor_protocols: Active protocol per actor {actor_id: protocol_id or None}
        per_round_reactions: Reaction usage tracking {actor_id: {reaction_id: count}}
        actor_action_economy: Action economy state per actor {actor_id: ActionEconomyState}
    """

    turn_order_rules: TurnOrderRules = Field(default_factory=TurnOrderRules)
    current_actor_id: str | None = Field(default=None)
    current_side: CombatSide | None = Field(default=None)
    round_index: int = Field(default=1, ge=1)
    actors_who_acted_this_round: dict[str, bool] = Field(default_factory=dict)
    all_combatants: dict[str, CombatSide] = Field(default_factory=dict)
    last_actor_id: str | None = Field(default=None)
    actor_priorities: dict[str, int] = Field(default_factory=dict)

    actor_phases: dict[str, TurnPhase] = Field(default_factory=dict)
    actor_prepared_actions: dict[str, PreparedActionState | None] = Field(
        default_factory=dict
    )
    actor_protocols: dict[str, str | None] = Field(default_factory=dict)
    per_round_reactions: dict[str, dict[str, int]] = Field(default_factory=dict)
    actor_action_economy: dict[str, ActionEconomyState] = Field(default_factory=dict)

    def get_side(self, actor_id: str) -> CombatSide | None:
        """Get the side for an actor."""
        return self.all_combatants.get(actor_id)

    def has_acted_this_round(self, actor_id: str) -> bool:
        """Check if an actor has already acted this round."""
        return self.actors_who_acted_this_round.get(actor_id, False)

    def get_current_phase(self, actor_id: str) -> TurnPhase | None:
        """Get the current phase for an actor."""
        return self.actor_phases.get(actor_id)

    def get_prepared_action(self, actor_id: str) -> PreparedActionState | None:
        """Get the prepared action for an actor."""
        return self.actor_prepared_actions.get(actor_id)

    def get_active_protocol(self, actor_id: str) -> str | None:
        """Get the active protocol for an actor."""
        return self.actor_protocols.get(actor_id)

    def get_reaction_count(self, actor_id: str, reaction_id: str) -> int:
        """Get the usage count for a reaction."""
        return self.per_round_reactions.get(actor_id, {}).get(reaction_id, 0)

    def get_action_economy(self, actor_id: str) -> ActionEconomyState | None:
        """Get the action economy state for an actor."""
        return self.actor_action_economy.get(actor_id)

    def can_take_full_action(self, actor_id: str) -> bool:
        """Check if an actor can take a full action."""
        economy = self.get_action_economy(actor_id)
        return economy is not None and economy.full_actions_remaining > 0

    def can_take_quick_action(self, actor_id: str) -> bool:
        """Check if an actor can take a quick action."""
        economy = self.get_action_economy(actor_id)
        return economy is not None and economy.quick_actions_remaining > 0

    def can_overcharge(self, actor_id: str) -> bool:
        """Check if an actor can overcharge."""
        economy = self.get_action_economy(actor_id)
        return economy is not None and economy.can_overcharge

    def can_take_reaction(self, actor_id: str) -> bool:
        """Check if an actor can take a reaction."""
        economy = self.get_action_economy(actor_id)
        return economy is not None and economy.reactions_remaining_this_turn > 0

    def is_round_complete(self) -> bool:
        """Check if all actors have taken a turn this round."""
        return all(self.has_acted_this_round(a) for a in self.all_combatants)


def start_tactical_combat_with_phases(
    combatants: dict[str, CombatSide],
    turn_order_rules: TurnOrderRules | None = None,
    actor_priorities: dict[str, int] | None = None,
) -> PhasedTacticalTracker:
    """Initialize tactical combat with phase tracking for all combatants.

    Per PR2: Players always take the very first turn in tactical combat.
    All actors start at "start" phase ready for protocol activation.

    Args:
        combatants: Map of actor_id to side
        turn_order_rules: Turn order configuration (uses defaults if None)
        actor_priorities: Priority overrides for specific actors

    Returns:
        Initialized PhasedTacticalTracker ready for first turn
    """
    rules = turn_order_rules or TurnOrderRules()

    actors_by_side: dict[CombatSide, list[str]] = {}
    for actor_id, side in combatants.items():
        if side not in actors_by_side:
            actors_by_side[side] = []
        actors_by_side[side].append(actor_id)

    priorities = actor_priorities or {}

    def sort_by_priority(actors: list[str]) -> list[str]:
        return sorted(actors, key=lambda a: (-priorities.get(a, 0), a))

    players = sort_by_priority(actors_by_side.get("players", []))
    hostiles = sort_by_priority(actors_by_side.get("hostiles", []))
    neutral = sort_by_priority(actors_by_side.get("neutral", []))

    first_actor: str | None = None
    first_side: CombatSide | None = None

    all_actors_by_priority = sort_by_priority(list(combatants.keys()))
    highest_priority_actor = (
        all_actors_by_priority[0] if all_actors_by_priority else None
    )
    highest_priority_side = (
        combatants.get(highest_priority_actor) if highest_priority_actor else None
    )

    if rules.players_act_first and players:
        first_actor = players[0]
        first_side = "players"
    elif hostiles:
        first_actor = hostiles[0]
        first_side = "hostiles"
    elif neutral:
        first_actor = neutral[0]
        first_side = "neutral"

    if highest_priority_actor and priorities.get(highest_priority_actor, 0) > 0:
        if highest_priority_actor != first_actor:
            first_actor = highest_priority_actor
            first_side = highest_priority_side

    all_actors = list(combatants.keys())

    return PhasedTacticalTracker(
        turn_order_rules=rules,
        current_actor_id=first_actor,
        current_side=first_side,
        all_combatants=combatants,
        actor_priorities=priorities,
        actor_phases={actor: "start" for actor in all_actors},
        actor_prepared_actions={actor: None for actor in all_actors},
        actor_protocols={actor: None for actor in all_actors},
        per_round_reactions={actor: {} for actor in all_actors},
        actor_action_economy={actor: ActionEconomyState() for actor in all_actors},
    )


def start_actor_turn(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> PhasedTacticalTracker:
    """Begin an actor's turn at the "start" phase.

    Per PR2: "The most common type of Free Action is a protocol, which can be
    activated or deactivated only at the start of a turn" (line 4406).

    Also clears previous turn's end-of-turn effects and resets per-turn states.

    Args:
        tracker: Current tracker state
        actor_id: Actor beginning their turn

    Returns:
        Updated tracker with actor at "start" phase
    """
    updates: dict[str, object] = {
        "current_actor_id": actor_id,
        "current_side": tracker.get_side(actor_id),
        "last_actor_id": tracker.current_actor_id,
        "actor_phases": {
            **tracker.actor_phases,
            actor_id: "start",
        },
        "actor_protocols": {
            **tracker.actor_protocols,
            actor_id: None,
        },
    }

    prepared = tracker.actor_prepared_actions.get(actor_id)
    if prepared is not None:
        prepared_expired = prepared.expires_on_turn <= tracker.round_index
        if prepared_expired:
            updates["actor_prepared_actions"] = {
                **tracker.actor_prepared_actions,
                actor_id: None,
            }

    return tracker.model_copy(update=updates)


def advance_phase(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> PhasedTacticalTracker:
    """Advance an actor's turn phase: start -> normal -> end.

    Per PR2 turn structure:
    1. Start of turn: Protocols, system activations
    2. Normal: Movement, quick/full actions, free actions
    3. End of turn: End-of-turn effects trigger, pass to next actor

    Args:
        tracker: Current tracker state
        actor_id: Actor advancing phases

    Returns:
        Updated tracker with actor at next phase

    Raises:
        ValueError: If already at "end" phase or invalid transition
    """
    current_phase = tracker.actor_phases.get(actor_id)

    if current_phase is None:
        raise ValueError(f"Actor {actor_id} has no phase set")

    if current_phase == "start":
        new_phase: TurnPhase = "normal"
    elif current_phase == "normal":
        new_phase = "end"
    else:
        raise ValueError(
            f"Cannot advance from end phase. Use end_actor_turn() to complete turn."
        )

    return tracker.model_copy(
        update={
            "actor_phases": {
                **tracker.actor_phases,
                actor_id: new_phase,
            }
        }
    )


def end_actor_turn(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> tuple[PhasedTacticalTracker, bool]:
    """Complete an actor's turn and advance to next actor/round.

    Marks the actor as having acted this round. If all actors have acted,
    starts a new round (resets reaction tracking).

    Args:
        tracker: Current tracker state
        actor_id: Actor ending their turn

    Returns:
        Tuple of (updated_tracker, is_new_round)
    """
    was_round_complete = tracker.is_round_complete()

    updated = tracker.model_copy(
        update={
            "actors_who_acted_this_round": {
                **tracker.actors_who_acted_this_round,
                actor_id: True,
            }
        }
    )

    is_new_round = not was_round_complete and updated.is_round_complete()

    if is_new_round:
        updated = updated.model_copy(
            update={
                "round_index": updated.round_index + 1,
                "actors_who_acted_this_round": {},
                "per_round_reactions": {actor: {} for actor in updated.all_combatants},
            }
        )

    return (updated, is_new_round)


def activate_protocol(
    tracker: PhasedTacticalTracker,
    actor_id: str,
    protocol_id: str,
    settings: ActionTimingValidationSettings | None = None,
) -> tuple[PhasedTacticalTracker, TimingValidationResult]:
    """Activate a protocol for an actor.

    Per PR2 line 4406: "The most common type of Free Action is a protocol,
    which can be activated or deactivated only at the start of a turn."

    Args:
        tracker: Current tracker state
        actor_id: Actor activating the protocol
        protocol_id: ID of the protocol to activate
        settings: Validation settings (uses defaults if None)

    Returns:
        Tuple of (updated_tracker, validation_result)
    """
    if settings is None:
        settings = DEFAULT_TIMING_VALIDATION

    current_phase = tracker.actor_phases.get(actor_id)
    validation = validate_protocol_timing(
        action_id=protocol_id,
        is_protocol=True,
        current_phase=current_phase or "start",
        settings=settings,
    )

    if not validation.valid:
        return tracker, validation

    updated = tracker.model_copy(
        update={
            "actor_protocols": {
                **tracker.actor_protocols,
                actor_id: protocol_id,
            }
        }
    )

    return updated, TimingValidationResult(valid=True)


def prepare_action(
    tracker: PhasedTacticalTracker,
    actor_id: str,
    held_action_id: str,
    held_action_type: ActionType,
    trigger_condition: str,
    expires_on_turn: int,
) -> PhasedTacticalTracker:
    """Prepare an action to be held for a trigger.

    Per PR2 4338-4356:
    - Can only prepare a quick action
    - Costs a quick action to prepare
    - Cannot take any other actions (quick, full, free, etc), reactions, or
      regular move until the start of next turn or when expended
    - Can be dropped as a free action to take reactions

    Args:
        tracker: Current tracker state
        actor_id: Actor preparing the action
        held_action_id: ID of the action to execute when triggered
        held_action_type: Type of the prepared action
        trigger_condition: Description of the trigger condition
        expires_on_turn: Turn number when prepared action expires

    Returns:
        Updated tracker with prepared action set
    """
    prepared = PreparedActionState(
        held_action_id=held_action_id,
        held_action_type=held_action_type,
        trigger_condition=trigger_condition,
        created_on_turn=tracker.round_index,
        expires_on_turn=expires_on_turn,
        blocks_actions=True,
        blocks_reactions=True,
        blocks_movement=True,
    )

    return tracker.model_copy(
        update={
            "actor_prepared_actions": {
                **tracker.actor_prepared_actions,
                actor_id: prepared,
            }
        }
    )


def drop_prepared_action(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> PhasedTacticalTracker:
    """Drop a prepared action, clearing the lockout.

    Per PR2 4354-4355: "If you want to take a reaction and drop your prepared
    reaction, you can also do so."

    Args:
        tracker: Current tracker state
        actor_id: Actor dropping their prepared action

    Returns:
        Updated tracker with prepared action cleared
    """
    return tracker.model_copy(
        update={
            "actor_prepared_actions": {
                **tracker.actor_prepared_actions,
                actor_id: None,
            }
        }
    )


def use_reaction(
    tracker: PhasedTacticalTracker,
    actor_id: str,
    reaction_id: str,
    max_per_round: int = 1,
    settings: ActionTimingValidationSettings | None = None,
) -> tuple[PhasedTacticalTracker, TimingValidationResult]:
    """Use a reaction and track its usage.

    Per PR2 4379-4384: "You can only make one reaction per turn (your turn or
    another actor's), but any number per round... All mechs can use the Brace
    and Overwatch reactions once per round by default."

    Args:
        tracker: Current tracker state
        actor_id: Actor using the reaction
        reaction_id: ID of the reaction being used
        max_per_round: Maximum uses per round for this reaction
        settings: Validation settings (uses defaults if None)

    Returns:
        Tuple of (updated_tracker, validation_result)
    """
    if settings is None:
        settings = DEFAULT_TIMING_VALIDATION

    current_count = tracker.get_reaction_count(actor_id, reaction_id)
    reaction_counts = tracker.per_round_reactions.get(actor_id, {})

    validation = validate_per_round_reaction(
        action_id=reaction_id,
        current_round=tracker.round_index,
        actor_id=actor_id,
        reaction_counts_by_actor={actor_id: reaction_counts},
        max_per_round=max_per_round,
    )

    if not validation.valid:
        return tracker, validation

    new_count = current_count + 1
    updated_reaction_counts = {**reaction_counts, reaction_id: new_count}

    updated = tracker.model_copy(
        update={
            "per_round_reactions": {
                **tracker.per_round_reactions,
                actor_id: updated_reaction_counts,
            }
        }
    )

    return updated, TimingValidationResult(valid=True)


def validate_action_timing(
    tracker: PhasedTacticalTracker,
    actor_id: str,
    action_id: str,
    action_type: ActionType,
    is_protocol: bool = False,
    settings: ActionTimingValidationSettings | None = None,
) -> TimingValidationResult:
    """Validate that an action can be taken at the current time.

    Combines protocol timing, prepared action lockout, and per-round reaction
    validation into a single check.

    Args:
        tracker: Current tracker state
        actor_id: Actor attempting the action
        action_id: ID of the action
        action_type: Type of action being taken
        is_protocol: Whether this is a protocol activation
        settings: Validation settings (uses defaults if None)

    Returns:
        Validation result indicating if timing is valid
    """
    if settings is None:
        settings = DEFAULT_TIMING_VALIDATION

    current_phase = tracker.actor_phases.get(actor_id, "normal")

    protocol_result = validate_protocol_timing(
        action_id=action_id,
        is_protocol=is_protocol,
        current_phase=current_phase,
        settings=settings,
    )

    if not protocol_result.valid:
        return protocol_result

    prepared = tracker.actor_prepared_actions.get(actor_id)
    prepared_result = validate_action_while_prepared(
        action_id=action_id,
        action_type=action_type,
        prepared_state=prepared,
        settings=settings,
    )

    if not prepared_result.valid:
        return prepared_result

    return TimingValidationResult(valid=True)


def get_phase_state(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> TurnPhaseState | None:
    """Get the turn phase state for an actor.

    Useful for integration with existing combat state that uses TurnPhaseState.

    Args:
        tracker: Current tracker state
        actor_id: Actor to get state for

    Returns:
        TurnPhaseState or None if actor not found
    """
    phase = tracker.actor_phases.get(actor_id)
    if phase is None:
        return None

    protocol = tracker.actor_protocols.get(actor_id)

    return TurnPhaseState(
        current_phase=phase,
        protocol_activated=protocol is not None,
        protocol_id=protocol,
    )


def get_eligible_actions(
    tracker: PhasedTacticalTracker,
    actor_id: str,
    available_actions: list[tuple[str, ActionType, bool]],
) -> list[tuple[str, ActionType, bool]]:
    """Get actions that are eligible given current phase and lockouts.

    Filters actions based on:
    - Protocol timing (only in "start" phase)
    - Prepared action lockout (blocks actions/reactions/movement)
    - Phase restrictions (if any)

    Args:
        tracker: Current tracker state
        actor_id: Actor attempting actions
        available_actions: List of (action_id, action_type, is_protocol) tuples

    Returns:
        Filtered list of eligible actions
    """
    current_phase = tracker.actor_phases.get(actor_id, "start")
    prepared = tracker.actor_prepared_actions.get(actor_id)

    eligible: list[tuple[str, ActionType, bool]] = []

    for action_id, action_type, is_protocol in available_actions:
        if is_protocol and current_phase != "start":
            continue

        if prepared is not None:
            if prepared.blocks_actions and action_type not in ("reaction",):
                continue
            if prepared.blocks_movement and action_type == "move":
                continue
            if prepared.blocks_reactions and action_type == "reaction":
                continue

        eligible.append((action_id, action_type, is_protocol))

    return eligible


def nominate_next_phase(
    tracker: PhasedTacticalTracker,
    nominator_id: str,
    nominee_id: str,
) -> tuple[PhasedTacticalTracker, bool, str | None]:
    """Nominate the next actor and advance to their turn.

    Combines nomination with phase reset (starts at "start" phase).

    Args:
        tracker: Current tracker state
        nominator_id: Actor making the nomination
        nominee_id: Actor being nominated

    Returns:
        Tuple of (updated_tracker, is_valid, error_message)
    """
    nominator_side = tracker.get_side(nominator_id)
    nominee_side = tracker.get_side(nominee_id)

    if nominator_side is None:
        return tracker, False, f"Nominator {nominator_id} not in combat"
    if nominee_side is None:
        return tracker, False, f"Nominee {nominee_id} not in combat"

    if tracker.turn_order_rules.nomination_required and nominator_side != nominee_side:
        unacted_on_nominator_side = [
            a
            for a, s in tracker.all_combatants.items()
            if s == nominator_side and not tracker.has_acted_this_round(a)
        ]
        unacted_on_nominee_side = [
            a
            for a, s in tracker.all_combatants.items()
            if s == nominee_side and not tracker.has_acted_this_round(a)
        ]
        nominator_side_exhausted = len(unacted_on_nominator_side) == 0
        nominee_side_exhausted = len(unacted_on_nominee_side) == 0

        if not (nominator_side_exhausted or nominee_side_exhausted):
            return (
                tracker,
                False,
                f"Cannot nominate across sides: {nominator_id} ({nominator_side}) "
                f"cannot nominate {nominee_id} ({nominee_side})",
            )

    if tracker.has_acted_this_round(nominee_id):
        return tracker, False, f"{nominee_id} has already acted this round"

    updated = tracker.model_copy(
        update={
            "current_actor_id": nominee_id,
            "current_side": nominee_side,
            "last_actor_id": nominator_id,
            "actor_phases": {
                **tracker.actor_phases,
                nominee_id: "start",
            },
            "actor_protocols": {
                **tracker.actor_protocols,
                nominee_id: None,
            },
        }
    )

    prepared = updated.actor_prepared_actions.get(nominee_id)
    if prepared is not None:
        if prepared.expires_on_turn <= updated.round_index:
            updated = updated.model_copy(
                update={
                    "actor_prepared_actions": {
                        **updated.actor_prepared_actions,
                        nominee_id: None,
                    }
                }
            )

    return updated, True, None


def get_actors_who_havent_acted(self) -> list[str]:
    """Get all actors who haven't acted this round."""
    return [a for a in self.all_combatants if not self.has_acted_this_round(a)]


def get_turn_order_for_display(
    tracker: PhasedTacticalTracker,
) -> list[tuple[str, int, str, TurnPhase]]:
    """Get turn order for display purposes with phase info.

    Returns list of (actor_id, round_index, side, phase) tuples.

    Args:
        tracker: Current tracker state

    Returns:
        List of tuples for display
    """
    result: list[tuple[str, int, str, TurnPhase]] = []

    for actor_id, side in tracker.all_combatants.items():
        phase = tracker.actor_phases.get(actor_id, "start")
        result.append((actor_id, tracker.round_index, side, phase))

    return result


def use_full_action(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> PhasedTacticalTracker:
    """Use a full action for an actor, updating action economy state.

    Per PR2: A turn consists of 1 full action OR 2 quick actions.

    Args:
        tracker: Current tracker state
        actor_id: Actor using the full action

    Returns:
        Updated tracker with full action counter incremented
    """
    current_economy = tracker.actor_action_economy.get(actor_id, ActionEconomyState())
    updated_economy = use_full_action_economy(current_economy)

    return tracker.model_copy(
        update={
            "actor_action_economy": {
                **tracker.actor_action_economy,
                actor_id: updated_economy,
            }
        }
    )


def use_quick_action(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> PhasedTacticalTracker:
    """Use a quick action for an actor, updating action economy state.

    Per PR2: A turn consists of 2 quick actions OR 1 full action.

    Args:
        tracker: Current tracker state
        actor_id: Actor using the quick action

    Returns:
        Updated tracker with quick action counter incremented
    """
    current_economy = tracker.actor_action_economy.get(actor_id, ActionEconomyState())
    updated_economy = use_quick_action_economy(current_economy)

    return tracker.model_copy(
        update={
            "actor_action_economy": {
                **tracker.actor_action_economy,
                actor_id: updated_economy,
            }
        }
    )


def use_overcharge(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> PhasedTacticalTracker:
    """Mark overcharge as used for an actor.

    Per PR2: Overcharge can only be used once per turn.

    Args:
        tracker: Current tracker state
        actor_id: Actor overcharging

    Returns:
        Updated tracker with overcharge marked as used
    """
    current_economy = tracker.actor_action_economy.get(actor_id, ActionEconomyState())
    updated_economy = use_overcharge_economy(current_economy)

    return tracker.model_copy(
        update={
            "actor_action_economy": {
                **tracker.actor_action_economy,
                actor_id: updated_economy,
            }
        }
    )


def use_reaction_economy(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> PhasedTacticalTracker:
    """Use a reaction for an actor, updating turn-based reaction counter.

    Per PR2: Can only make one reaction per turn.

    Args:
        tracker: Current tracker state
        actor_id: Actor using the reaction

    Returns:
        Updated tracker with reaction counter incremented
    """
    current_economy = tracker.actor_action_economy.get(actor_id, ActionEconomyState())
    updated_economy = use_reaction_economy(current_economy)

    return tracker.model_copy(
        update={
            "actor_action_economy": {
                **tracker.actor_action_economy,
                actor_id: updated_economy,
            }
        }
    )


def reset_action_economy_for_new_round(
    tracker: PhasedTacticalTracker,
) -> PhasedTacticalTracker:
    """Reset action economy for all actors at the start of a new round.

    Per PR2: Reactions reset at round boundary, but per-turn limits
    (overcharge, 1/turn reactions) reset at the start of each actor's turn.

    This function resets per-round reaction tracking while preserving
    per-turn economy state that resets naturally at turn start.

    Args:
        tracker: Current tracker state

    Returns:
        Updated tracker with per-round reaction tracking reset
    """
    return tracker.model_copy(
        update={
            "per_round_reactions": {actor: {} for actor in tracker.all_combatants},
        }
    )


def get_action_economy_summary_for_actor(
    tracker: PhasedTacticalTracker,
    actor_id: str,
) -> dict:
    """Get a summary of action economy for a specific actor.

    Args:
        tracker: Current tracker state
        actor_id: Actor to get summary for

    Returns:
        Summary dict with action economy details
    """
    economy = tracker.actor_action_economy.get(actor_id, ActionEconomyState())
    return get_action_economy_summary(economy)


def validate_action_with_economy(
    tracker: PhasedTacticalTracker,
    actor_id: str,
    action_type: ActionType,
    is_overcharge: bool = False,
) -> ActionEconomyResult:
    """Validate if an action can be taken given current economy state.

    Args:
        tracker: Current tracker state
        actor_id: Actor attempting the action
        action_type: Type of action being taken
        is_overcharge: Whether this is an overcharge action

    Returns:
        ActionEconomyResult with validation outcome
    """
    economy = tracker.actor_action_economy.get(actor_id, ActionEconomyState())
    return validate_action_economy(economy, action_type, is_overcharge)
