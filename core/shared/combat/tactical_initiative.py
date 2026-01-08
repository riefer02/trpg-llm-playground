"""Tactical combat primitives for Lancer mech combat.

Provides turn order management for tactical combat following PR2 3703-3725:
- Nomination-based turn order (players always first, alternate sides)
- Priority overrides for veteran NPCs (e.g., Viper's Speed)
- Integration with existing CombatRound and TurnOrderRules
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field

from core.shared.models import FrozenModel

CombatSide = Literal["players", "hostiles", "neutral"]

from core.mech.combat_rules import TurnOrderRules


class ActorPriority(FrozenModel):
    """Priority override for actor turn order.

    Higher priority actors go first, potentially breaking alternation.
    Used for veteran NPCs with traits like Viper's Speed.
    """

    actor_id: str
    priority: int = Field(..., ge=0, description="Higher = earlier in turn order")
    reason: str | None = Field(default=None, description="e.g., Viper's Speed trait")


class TacticalInitiativeTracker(FrozenModel):
    """Manages tactical combat turn order per PR2 3703-3725.

    Implements the nomination-based alternating system:
    - Players always act first (no opposed checks)
    - Previous actor nominates next from their side
    - Sides alternate strictly
    - Higher priority actors jump ahead (absolute priority)
    - Remaining actors on exhausted side go any order

    Attributes:
        turn_order_rules: Configuration for turn order behavior
        current_actor_id: Actor currently taking their turn
        current_side: Side of current actor ("players" or "hostiles")
        round_index: Current round number (starts at 1)
        actors_who_acted_this_round: Track who has acted this round {actor_id: True}
        all_combatants: All combatants in combat {actor_id: side}
        last_actor_id: Previous actor (for nomination validation)
        actor_priorities: Priority overrides {actor_id: priority_value}
    """

    turn_order_rules: TurnOrderRules = Field(default_factory=TurnOrderRules)
    current_actor_id: str | None = Field(default=None)
    current_side: CombatSide | None = Field(default=None)
    round_index: int = Field(default=1, ge=1)
    actors_who_acted_this_round: dict[str, bool] = Field(default_factory=dict)
    all_combatants: dict[str, CombatSide] = Field(default_factory=dict)
    last_actor_id: str | None = Field(default=None)
    actor_priorities: dict[str, int] = Field(default_factory=dict)

    def get_side(self, actor_id: str) -> CombatSide | None:
        """Get the side for an actor."""
        return self.all_combatants.get(actor_id)

    def get_actors_by_side(self) -> dict[CombatSide, list[str]]:
        """Get all actors grouped by side, sorted by priority."""
        players = [a for a, s in self.all_combatants.items() if s == "players"]
        hostiles = [a for a, s in self.all_combatants.items() if s == "hostiles"]
        neutral = [a for a, s in self.all_combatants.items() if s == "neutral"]

        def sort_by_priority(actors: list[str]) -> list[str]:
            return sorted(
                actors,
                key=lambda a: (-self.actor_priorities.get(a, 0), a),
            )

        result: dict[CombatSide, list[str]] = {}
        if players:
            result["players"] = sort_by_priority(players)
        if hostiles:
            result["hostiles"] = sort_by_priority(hostiles)
        if neutral:
            result["neutral"] = sort_by_priority(neutral)

        return result

    def has_acted_this_round(self, actor_id: str) -> bool:
        """Check if an actor has already acted this round."""
        return self.actors_who_acted_this_round.get(actor_id, False)

    def get_actors_who_havent_acted(self) -> list[str]:
        """Get all actors who haven't acted this round."""
        return [a for a in self.all_combatants if not self.has_acted_this_round(a)]

    def get_turn_order(self) -> list[str]:
        """Calculate the full turn order for the current round.

        Returns actors in the order they should take turns,
        respecting priorities and alternation rules.
        """
        return self.get_next_turn_order()

    def get_next_turn_order(self) -> list[str]:
        """Calculate the full turn order for the current round.

        Returns actors in the order they should take turns,
        respecting priorities and alternation rules.

        Per PR2 3703-3725:
        - Players act first (by default)
        - Sides alternate: player turn, then NPC turn
        - Remaining actors on exhausted side go any order
        """
        actors_by_side = self.get_actors_by_side()
        players = actors_by_side.get("players", [])
        hostiles = actors_by_side.get("hostiles", [])
        neutral = actors_by_side.get("neutral", [])

        if self.turn_order_rules.players_act_first:
            first_side: CombatSide = "players"
            second_side: CombatSide = "hostiles"
        else:
            first_side = "hostiles"
            second_side = "players"

        first_list = actors_by_side.get(first_side, [])
        second_list = actors_by_side.get(second_side, [])

        turn_order: list[str] = []
        first_idx = 0
        second_idx = 0
        first_exhausted = len(first_list) == 0
        second_exhausted = len(second_list) == 0
        current_side = first_side

        while not (first_exhausted and second_exhausted):
            if current_side == first_side and not first_exhausted:
                turn_order.append(first_list[first_idx])
                first_idx += 1
                if first_idx >= len(first_list):
                    first_exhausted = True
                if not second_exhausted:
                    current_side = second_side
            elif current_side == second_side and not second_exhausted:
                turn_order.append(second_list[second_idx])
                second_idx += 1
                if second_idx >= len(second_list):
                    second_exhausted = True
                if not first_exhausted:
                    current_side = first_side
            else:
                break

        return turn_order

    def get_current_round_progress(self) -> tuple[int, int]:
        """Get progress through current round: (acted_count, total_count)."""
        acted = len(
            [
                a
                for a in self.actors_who_acted_this_round
                if self.actors_who_acted_this_round[a]
            ]
        )
        total = len(self.all_combatants)
        return (acted, total)

    def is_round_complete(self) -> bool:
        """Check if all actors have taken a turn this round."""
        return all(self.has_acted_this_round(a) for a in self.all_combatants)


def start_tactical_combat(
    combatants: dict[str, CombatSide],
    turn_order_rules: TurnOrderRules | None = None,
    actor_priorities: dict[str, int] | None = None,
) -> TacticalInitiativeTracker:
    """Initialize tactical combat with all combatants.

    Per PR2: Players always take the very first turn in tactical combat.
    The first actor is selected from player side, respecting priorities.

    Args:
        combatants: Map of actor_id to side
        turn_order_rules: Turn order configuration (uses defaults if None)
        actor_priorities: Priority overrides for specific actors

    Returns:
        Initialized TacticalInitiativeTracker ready for first nomination
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

    return TacticalInitiativeTracker(
        turn_order_rules=rules,
        current_actor_id=first_actor,
        current_side=first_side,
        all_combatants=combatants,
        actor_priorities=priorities,
    )


def nominate_next(
    tracker: TacticalInitiativeTracker,
    nominator_id: str,
    nominee_id: str,
) -> TacticalInitiativeTracker:
    """Nominate the next actor to take a turn.

    Per PR2: The previous acting player nominates the next player/friendly NPC.
    Sides alternate: player turn, then NPC turn, then player turn, etc.

    Cross-side nomination is allowed when one side's actors are all exhausted.

    Args:
        tracker: Current initiative tracker
        nominator_id: Actor who is making the nomination
        nominee_id: Actor being nominated

    Returns:
        Updated tracker with new current actor

    Raises:
        ValueError: If nomination violates alternation rules
    """
    nominator_side = tracker.get_side(nominator_id)
    nominee_side = tracker.get_side(nominee_id)

    if nominator_side is None:
        raise ValueError(f"Nominator {nominator_id} not in combat")
    if nominee_side is None:
        raise ValueError(f"Nominee {nominee_id} not in combat")

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
            raise ValueError(
                f"Cannot nominate across sides: {nominator_id} ({nominator_side}) cannot nominate "
                f"{nominee_id} ({nominee_side}). Sides must alternate."
            )

    if tracker.has_acted_this_round(nominee_id):
        raise ValueError(f"{nominee_id} has already acted this round")

    return tracker.model_copy(
        update={
            "current_actor_id": nominee_id,
            "current_side": nominee_side,
            "last_actor_id": nominator_id,
        }
    )


def complete_turn(
    tracker: TacticalInitiativeTracker,
    actor_id: str,
) -> tuple[TacticalInitiativeTracker, bool]:
    """Mark a turn as complete and advance to next.

    Records that the actor has acted this round and returns whether
    a new round has begun.

    Args:
        tracker: Current initiative tracker
        actor_id: Actor whose turn is ending

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
            }
        )

    return (updated, is_new_round)


def get_eligible_nominees(
    tracker: TacticalInitiativeTracker,
    nominator_id: str,
) -> list[str]:
    """Get list of actors eligible to be nominated by the given nominator.

    Returns actors from the same side who haven't acted this round yet,
    excluding the nominator themselves.

    Args:
        tracker: Current initiative tracker
        nominator_id: The actor making the nomination

    Returns:
        List of actor IDs eligible for nomination
    """
    nominator_side = tracker.get_side(nominator_id)
    if nominator_side is None:
        return []

    return [
        a
        for a, side in tracker.all_combatants.items()
        if side == nominator_side
        and not tracker.has_acted_this_round(a)
        and a != nominator_id
    ]


def get_remaining_actors_on_side(
    tracker: TacticalInitiativeTracker,
    side: CombatSide,
) -> list[str]:
    """Get actors on the given side who haven't acted this round.

    Per PR2: Remaining actors on exhausted side go in any order.

    Args:
        tracker: Current initiative tracker
        side: The side to check

    Returns:
        List of actors on that side who haven't acted
    """
    return [
        a
        for a, s in tracker.all_combatants.items()
        if s == side and not tracker.has_acted_this_round(a)
    ]


def get_turn_order_for_display(
    tracker: TacticalInitiativeTracker,
) -> list[tuple[str, int, str]]:
    """Get turn order for display purposes.

    Returns list of (actor_id, round_index, side) tuples showing
    the full turn order for the current round.

    Args:
        tracker: Current initiative tracker

    Returns:
        List of tuples for display
    """
    actors_by_side = tracker.get_actors_by_side()
    players = actors_by_side.get("players", [])
    hostiles = actors_by_side.get("hostiles", [])

    if tracker.turn_order_rules.players_act_first:
        first_list, second_list = players, hostiles
    else:
        first_list, second_list = hostiles, players

    turn_order: list[tuple[str, int, str]] = []
    first_idx = 0
    second_idx = 0
    first_exhausted = len(first_list) == 0
    second_exhausted = len(second_list) == 0

    while not (first_exhausted and second_exhausted):
        if not first_exhausted and first_idx < len(first_list):
            actor = first_list[first_idx]
            turn_order.append((actor, tracker.round_index, "players"))
            first_idx += 1
            if first_idx >= len(first_list):
                first_exhausted = True
        elif not second_exhausted and second_idx < len(second_list):
            actor = second_list[second_idx]
            turn_order.append((actor, tracker.round_index, "hostiles"))
            second_idx += 1
            if second_idx >= len(second_list):
                second_exhausted = True
        else:
            break

    return turn_order


def is_valid_nomination(
    tracker: TacticalInitiativeTracker,
    nominator_id: str,
    nominee_id: str,
) -> tuple[bool, str | None]:
    """Check if a nomination would be valid.

    Cross-side nomination is allowed when one side's actors are all exhausted.

    Args:
        tracker: Current initiative tracker
        nominator_id: Actor making the nomination
        nominee_id: Actor being nominated

    Returns:
        Tuple of (is_valid, error_message_if_not_valid)
    """
    if not tracker.turn_order_rules.nomination_required:
        return True, None

    nominator_side = tracker.get_side(nominator_id)
    nominee_side = tracker.get_side(nominee_id)

    if nominator_side is None:
        return False, f"Nominator {nominator_id} not in combat"
    if nominee_side is None:
        return False, f"Nominee {nominee_id} not in combat"

    if nominator_side != nominee_side:
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
            return False, (
                f"Cannot nominate across sides: {nominator_id} ({nominator_side}) "
                f"cannot nominate {nominee_id} ({nominee_side})"
            )

    if tracker.has_acted_this_round(nominee_id):
        return False, f"{nominee_id} has already acted this round"

    return True, None


def advance_to_next_actor(
    tracker: TacticalInitiativeTracker,
) -> TacticalInitiativeTracker | None:
    """Advance to the next actor in turn order after current completes.

    Used when actors should automatically flow through turn order
    without explicit nomination (e.g., remaining actors on exhausted side).

    Args:
        tracker: Current initiative tracker

    Returns:
        Updated tracker with next actor, or None if round is complete
    """
    if tracker.is_round_complete():
        return None

    remaining = tracker.get_actors_who_havent_acted()
    if not remaining:
        return None

    current_side = tracker.current_side
    remaining_on_current = [a for a in remaining if tracker.get_side(a) == current_side]

    next_actor: str | None = None

    if remaining_on_current and tracker.turn_order_rules.remaining_side_any_order:
        next_actor = remaining_on_current[0]
    elif remaining:
        next_side = "players" if current_side == "hostiles" else "hostiles"
        remaining_on_next = [a for a in remaining if tracker.get_side(a) == next_side]
        if remaining_on_next:
            next_actor = remaining_on_next[0]

    if next_actor is None and remaining:
        next_actor = remaining[0]

    if next_actor is None:
        return None

    return tracker.model_copy(
        update={
            "current_actor_id": next_actor,
            "current_side": tracker.get_side(next_actor),
            "last_actor_id": tracker.current_actor_id,
        }
    )
