"""NPC AI behavior helpers for automated decision-making.

This module provides simple role-based behavior patterns for NPC decision-making,
matching the NPC class descriptions from PR2 (lines 13293-13309).

NPC Roles (per book):
- Striker: "damage dealers... mobile and durable... close or midrange"
- Defender: "very resilient... protect allies or an area"
- Controller: "inflict low/no damage... focus on denying areas"
- Supporter: "focus on aiding allies... increase damage or resilience"
- Artillery: "long ranged, accurate, high damage... fragile"

Target awareness: Can read target stats for smarter decisions.
"""

from __future__ import annotations

from typing import Literal, NamedTuple
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import ActionType, SizeClass
from core.shared.id_helpers import CombatantIdField
from core.npc.models import NPCRole
from core.npc.state import NPCState, NPCTemplate
from core.mech.grid import HexCoord


TargetPriority = Literal["low_hp", "high_threat", "nearest", "objective", "low_hp_ally"]


class NPCBehaviorPattern(FrozenModel):
    """Role-based targeting and action preferences for NPCs.

    Args:
        role: NPC role classification (striker, defender, controller, supporter)
        priority_targets: Order of target priority for this role
        preferred_actions: Actions this NPC prefers to take
        target_awareness: Whether to consider target stats in decisions
    """

    role: NPCRole
    priority_targets: list[TargetPriority] = Field(default_factory=list)
    preferred_actions: list[ActionType] = Field(default_factory=list)
    target_awareness: bool = True


class TargetInfo(FrozenModel):
    """Information about a potential target for NPC decision-making."""

    id: str
    distance: int
    hp_current: int
    hp_max: int
    is_objective_holder: bool = False
    is_ally: bool = False


class ActionScore(FrozenModel):
    """Score for a potential action-target combination."""

    action: ActionType
    target_id: CombatantIdField | None
    score: float
    reasoning: str = ""


class NPCActionDecision(FrozenModel):
    """Result of AI decision-making for an NPC turn."""

    action: ActionType
    target_id: CombatantIdField | None
    reasoning: str
    fallback_used: bool = False


STRIKER_PATTERN = NPCBehaviorPattern(
    role="striker",
    priority_targets=["low_hp", "high_threat", "nearest"],
    preferred_actions=["full", "quick"],
    target_awareness=True,
)

DEFENDER_PATTERN = NPCBehaviorPattern(
    role="defender",
    priority_targets=["nearest", "objective"],
    preferred_actions=["full", "reaction"],
    target_awareness=True,
)

CONTROLLER_PATTERN = NPCBehaviorPattern(
    role="controller",
    priority_targets=["nearest", "objective"],
    preferred_actions=["quick", "full"],
    target_awareness=True,
)

SUPPORTER_PATTERN = NPCBehaviorPattern(
    role="supporter",
    priority_targets=["low_hp_ally", "objective"],
    preferred_actions=["quick", "full"],
    target_awareness=True,
)

NPC_BEHAVIOR_PATTERNS: dict[NPCRole, NPCBehaviorPattern] = {
    "striker": STRIKER_PATTERN,
    "defender": DEFENDER_PATTERN,
    "controller": CONTROLLER_PATTERN,
    "supporter": SUPPORTER_PATTERN,
}


def get_role_from_template(template: NPCTemplate) -> NPCRole:
    """Get the role from an NPC template.

    Args:
        template: The NPC template

    Returns:
        The NPC's role
    """
    return template.role


def get_behavior_pattern(role: NPCRole) -> NPCBehaviorPattern:
    """Get the default behavior pattern for an NPC role.

    Args:
        role: The NPC role to get a pattern for

    Returns:
        Default behavior pattern for this role
    """
    return NPC_BEHAVIOR_PATTERNS.get(role, STRIKER_PATTERN)


def compute_target_score(
    target: TargetInfo,
    priorities: list[TargetPriority],
    npc_hp: int,
) -> float:
    """Compute a priority score for a target based on NPC role priorities.

    Args:
        target: Target information
        priorities: Order of priorities for this NPC role
        npc_hp: Current HP of the NPC making the decision

    Returns:
        Score (higher = more attractive target)
    """
    score = 0.0

    for priority in priorities:
        if priority == "low_hp":
            hp_pct = target.hp_current / max(target.hp_max, 1)
            score += (1.0 - hp_pct) * 10
        elif priority == "high_threat":
            if target.hp_current > npc_hp:
                score += 5
        elif priority == "nearest":
            score += max(0, 10 - target.distance)
        elif priority == "objective":
            if target.is_objective_holder:
                score += 15
        elif priority == "low_hp_ally":
            if target.is_ally and target.hp_current < target.hp_max * 0.5:
                score += 15

    return score


def score_available_actions(
    npc: NPCState,
    available_actions: list[ActionType],
    visible_targets: list[TargetInfo],
    pattern: NPCBehaviorPattern,
) -> list[ActionScore]:
    """Score available actions against visible targets.

    Args:
        npc: The NPC making the decision
        available_actions: Actions the NPC can take this turn
        visible_targets: Targets visible to the NPC
        pattern: Behavior pattern to use for scoring

    Returns:
        List of scored action-target combinations, sorted by score
    """
    scored_actions: list[ActionScore] = []

    for action in available_actions:
        if not visible_targets:
            scored_actions.append(
                ActionScore(
                    action=action,
                    target_id=None,
                    score=0.0,
                    reasoning="No targets visible",
                )
            )
            continue

        best_target = None
        best_score = float("-inf")

        for target in visible_targets:
            target_score = compute_target_score(
                target, pattern.priority_targets, npc.stats.hp_max
            )

            if pattern.target_awareness:
                if target.hp_current <= 0:
                    continue

            if target_score > best_score:
                best_score = target_score
                best_target = target

        if best_target:
            scored_actions.append(
                ActionScore(
                    action=action,
                    target_id=best_target.id,
                    score=best_score,
                    reasoning=f"Target {best_target.id} (score: {best_score:.1f})",
                )
            )
        else:
            scored_actions.append(
                ActionScore(
                    action=action,
                    target_id=None,
                    score=0.0,
                    reasoning="No valid targets",
                )
            )

    return sorted(scored_actions, key=lambda x: x.score, reverse=True)


def select_npc_action_with_role(
    npc: NPCState,
    role: NPCRole,
    available_actions: list[ActionType],
    visible_targets: list[TargetInfo],
) -> NPCActionDecision:
    """Select the best action for an NPC based on its role.

    This is a simple heuristic-based decision maker. For more complex
    behavior, implement custom decision functions.

    Args:
        npc: The NPC making the decision
        role: The NPC's role for behavior pattern selection
        available_actions: Actions the NPC can take this turn
        visible_targets: Currently visible targets

    Returns:
        NPCActionDecision with selected action and reasoning
    """
    pattern = get_behavior_pattern(role)

    scored = score_available_actions(npc, available_actions, visible_targets, pattern)

    if not scored:
        return NPCActionDecision(
            action="full",
            target_id=None,
            reasoning="No actions or targets available",
            fallback_used=True,
        )

    best = scored[0]

    if not pattern.preferred_actions:
        return NPCActionDecision(
            action=best.action,
            target_id=best.target_id,
            reasoning=f"Fallback: {best.reasoning}",
            fallback_used=True,
        )

    for action in pattern.preferred_actions:
        if action in available_actions:
            return NPCActionDecision(
                action=action,
                target_id=best.target_id,
                reasoning=f"Preferred action: {action} → {best.reasoning}",
            )

    return NPCActionDecision(
        action=best.action,
        target_id=best.target_id,
        reasoning=f"Fallback: {best.reasoning}",
        fallback_used=True,
    )


def is_adjacent(
    pos1: HexCoord, pos2: HexCoord, size1: SizeClass, size2: SizeClass
) -> bool:
    """Check if two units are adjacent considering size.

    Adjacent means orthogonal (not diagonal) and within adjacency distance.
    Same position is not adjacent.
    """
    if pos1 == pos2:
        return False

    dx = abs(pos1.q - pos2.q)
    dy = abs(pos1.r - pos2.r)

    max_dist = max(dx, dy)
    min_dist = min(dx, dy)

    if min_dist != 0:
        return False

    size_bonus = 0
    if size1 == "size_2" or size2 == "size_2":
        size_bonus = 1
    return max_dist <= (1 + size_bonus)
