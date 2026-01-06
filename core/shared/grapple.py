"""Grapple and Ram resolution helpers for Lancer combat.

This module provides type-safe helpers for Grapple and Ram actions including:
- Grapple attempt resolution (hit detection, size comparison, condition application)
- Contested HULL checks for same-size grapples
- Group grapple size calculation
- Ram resolution (prone application, knockback direction)
- Disarm mechanics on successful grapple

Grapple Rules (per PR2 ~4157-4177):
- Grapple is a quick action melee attack
- On hit: both parties become engaged, neither can boost/react
- Smaller party is immobilized, moves when larger party moves
- Same size = contested HULL check at start of turn
- Grapple breaks if adjacency breaks (e.g., knockback)
- Attacker ends free action, defender ends via quick action contested check

Ram Rules (per PR2 ~4152-4155):
- Ram is a quick action melee attack against adjacent target
- On hit: target becomes prone, may knock back up to 1 space directly away

Size Rules (per PR2 ~3823-3836):
- Size 1/2: humans, small mechs, EVA suits
- Size 1: typical light/assault/line mechs
- Size 2: battle tanks, vehicles, heavy mechs
- Size 3: siege mechs
- Size 4-5: titanic mechs, mech-oriented flyers
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import SizeClass, StatusType
from core.shared.dice import roll_dice
from core.mech.grid import HexCoord
from core.mech.statuses import get_status_definition


SIZE_ORDER: dict[SizeClass, int] = {
    "size_half": 1,
    "size_1": 1,
    "size_2": 2,
    "size_3": 3,
    "size_4": 4,
    "size_5": 5,
}


class GrappleAttempt(FrozenModel):
    """Input for grapple attempt resolution."""

    attacker_size: SizeClass
    target_size: SizeClass
    attacker_has_mount: bool = True
    target_has_mount: bool = True
    hit: bool = True
    attacker_hull_bonus: int = 0
    target_hull_bonus: int = 0


class GrappleResult(FrozenModel):
    """Result of a grapple attempt."""

    hit: bool
    grapple_initiated: bool
    attacker_engaged: bool = False
    target_engaged: bool = False
    smaller_party: Literal["attacker", "target", "tie"]
    target_becomes_immobilized: bool = False
    attacker_becomes_immobilized: bool = False
    contested_check_required: bool = False
    contested_check_winner: Literal["attacker", "target", "tie"] | None = None
    contested_check_roll: int | None = None
    reason: str = ""


class GrappleStatus(FrozenModel):
    """Status of an ongoing grapple."""

    active: bool
    attacker_grappling: bool = False
    target_grappled: bool = False
    smaller_party: Literal["attacker", "target"] | None = None
    immobilized_party: Literal["attacker", "target", "none"] = "none"
    can_boost: bool = True
    can_take_reactions: bool = True
    contested_check_pending: bool = False


class GrappleEndResult(FrozenModel):
    """Result of ending a grapple."""

    ended: bool
    by_initiator: bool
    new_engagement_state: Literal["still_engaged", "disengaged", "none"] = "none"
    reason: str = ""


class RamAttempt(FrozenModel):
    """Input for ram attempt resolution."""

    attacker_size: SizeClass
    target_size: SizeClass
    attacker_has_mount: bool = True
    hit: bool = True
    knockback_bonus: int = 0


class RamResult(FrozenModel):
    """Result of a ram attempt."""

    hit: bool
    target_becomes_prone: bool = False
    knockback_spaces: int = 0
    knockback_direction: HexCoord | None = None
    knockback_blocked: bool = False
    reason: str = ""


class GroupGrappleSize(FrozenModel):
    """Group grapple size calculation result."""

    participant_sizes: list[SizeClass]
    total_size: int
    largest_size: SizeClass
    participant_count: int


class DisarmResult(FrozenModel):
    """Result of disarm attempt during grapple."""

    attempted: bool
    successful: bool
    target_lost_mount: bool = False
    reason: str = ""


def _compare_sizes(size_a: SizeClass, size_b: SizeClass) -> Literal["a", "b", "tie"]:
    """Compare two sizes, returning which is larger or if tied.

    Args:
        size_a: First size to compare
        size_b: Second size to compare

    Returns:
        "a" if size_a > size_b, "b" if size_b > size_a, "tie" if equal
    """
    val_a = SIZE_ORDER.get(size_a, 0)
    val_b = SIZE_ORDER.get(size_b, 0)
    if val_a > val_b:
        return "a"
    elif val_b > val_a:
        return "b"
    return "tie"


def can_grapple(
    attacker_size: SizeClass,
    target_size: SizeClass,
    attacker_has_mount: bool = True,
    target_conditions: list[StatusType] | None = None,
) -> tuple[bool, str]:
    """Check if a grapple can be initiated.

    Per PR2: Grapple requires adjacency and a melee attack.
    Some conditions may prevent grappling (e.g., stunned prevents actions).

    Args:
        attacker_size: Size of the grappling mech
        target_size: Size of the target mech
        attacker_has_mount: Whether attacker has appropriate mount
        target_conditions: Current conditions on target

    Returns:
        Tuple of (can_grapple, reason)
    """
    if not attacker_has_mount:
        return False, "Attacker lacks appropriate mount for grappling"

    if target_conditions and "stunned" in target_conditions:
        definition = get_status_definition("stunned")
        if definition and definition.effects.action_restrictions:
            if definition.effects.action_restrictions.disallow_actions:
                return False, "Target is stunned and cannot be grappled"

    return True, "Grapple can be initiated"


def attempt_grapple(attempt: GrappleAttempt) -> GrappleResult:
    """Resolve a grapple attempt.

    Per PR2 ~4157-4166:
    - On hit: both parties become engaged
    - Neither can boost or take reactions
    - Smaller party is immobilized, moves when larger moves
    - Same size: contested HULL check at start of turn

    Args:
        attempt: GrappleAttempt with all required parameters

    Returns:
        GrappleResult with full resolution details
    """
    if not attempt.hit:
        return GrappleResult(
            hit=False,
            grapple_initiated=False,
            smaller_party="tie",
            reason="Grapple attack missed",
        )

    size_comparison = _compare_sizes(attempt.attacker_size, attempt.target_size)

    if size_comparison == "tie":
        roll = roll_dice("1d20")
        attacker_total = roll + attempt.attacker_hull_bonus
        target_total = roll + attempt.target_hull_bonus

        if attacker_total > target_total:
            winner = "attacker"
        elif target_total > attacker_total:
            winner = "target"
        else:
            winner = "tie"

        return GrappleResult(
            hit=True,
            grapple_initiated=True,
            attacker_engaged=True,
            target_engaged=True,
            smaller_party="tie",
            contested_check_required=True,
            contested_check_winner=winner,
            contested_check_roll=roll,
            reason=f"Same-size grapple: contested HULL check (roll {roll}, attacker {attacker_total} vs target {target_total})",
        )

    smaller = "attacker" if size_comparison == "b" else "target"
    larger = "target" if size_comparison == "b" else "attacker"

    attacker_immobilized = smaller == "attacker"
    target_immobilized = smaller == "target"

    return GrappleResult(
        hit=True,
        grapple_initiated=True,
        attacker_engaged=True,
        target_engaged=True,
        smaller_party=smaller,
        target_becomes_immobilized=target_immobilized,
        attacker_becomes_immobilized=attacker_immobilized,
        contested_check_required=False,
        reason=f"Grapple successful: {smaller} party is immobilized (size {attempt.attacker_size} vs {attempt.target_size})",
    )


def resolve_grapple_status(
    attacker_size: SizeClass,
    target_size: SizeClass,
    attacker_conditions: list[StatusType] | None = None,
    target_conditions: list[StatusType] | None = None,
) -> GrappleStatus:
    """Get the current status of an ongoing grapple.

    Per PR2 ~4162:
    - While grappled/grappling, neither party can boost or take reactions
    - Smaller party is immobilized, moves when larger party moves

    Args:
        attacker_size: Size of grappling mech
        target_size: Size of grappled mech
        attacker_conditions: Current conditions on attacker
        target_conditions: Current conditions on target

    Returns:
        GrappleStatus with current grapple state
    """
    size_comparison = _compare_sizes(attacker_size, target_size)

    if size_comparison == "tie":
        smaller = None
        contested_pending = True
    elif size_comparison == "a":
        smaller = "target"
        contested_pending = False
    else:
        smaller = "attacker"
        contested_pending = False

    immobilized = smaller if smaller else "none"

    attacker_immo = attacker_conditions and "immobilized" in attacker_conditions
    target_immo = target_conditions and "immobilized" in target_conditions

    if attacker_immo and target_immo:
        can_boost = False
        can_react = False
    elif attacker_immo or target_immo:
        can_boost = False
        can_react = False
    else:
        can_boost = False
        can_react = False

    return GrappleStatus(
        active=True,
        attacker_grappling=True,
        target_grappled=True,
        smaller_party=smaller,
        immobilized_party=immobilized,
        can_boost=can_boost,
        can_take_reactions=can_react,
        contested_check_pending=contested_pending,
    )


def end_grapple(initiator: Literal["attacker", "target"]) -> GrappleEndResult:
    """End a grapple.

    Per PR2 ~4172-4173:
    - Attacker can end grapple as free action
    - Defender can end grapple as quick action by winning contested HULL check

    Args:
        initiator: Who is ending the grapple

    Returns:
        GrappleEndResult with outcome details
    """
    if initiator == "attacker":
        return GrappleEndResult(
            ended=True,
            by_initiator=True,
            new_engagement_state="disengaged",
            reason="Attacker ended grapple as free action",
        )

    return GrappleEndResult(
        ended=True,
        by_initiator=False,
        new_engagement_state="disengaged",
        reason="Defender wins contested HULL check and ends grapple as quick action",
    )


def contest_grapple_check(
    attacker_hull_bonus: int,
    target_hull_bonus: int,
) -> tuple[Literal["attacker", "target", "tie"], int]:
    """Resolve a contested HULL check for grapple.

    Used when both parties are same size, or when defender attempts to escape.

    Per PR2 ~4164-4165:
    "If both parties are the same size, they can make a contested hull check at the
    start of their turn, counting as the larger party until this contest is repeated."

    Args:
        attacker_hull_bonus: Attacker's HULL skill bonus
        target_hull_bonus: Target's HULL skill bonus

    Returns:
        Tuple of (winner, roll_result)
    """
    roll = roll_dice("1d20")
    attacker_total = roll + attacker_hull_bonus
    target_total = roll + target_hull_bonus

    if attacker_total > target_total:
        return "attacker", roll
    elif target_total > attacker_total:
        return "target", roll
    return "tie", roll


def calculate_group_grapple_size(
    participant_sizes: list[SizeClass],
) -> GroupGrappleSize:
    """Calculate total size for a group grapple.

    Per PR2 ~4174-4177:
    "If there are multiple parties involved in a grapple, the same rules apply, but
    when counting size, count up all opponents of a side in a grapple."

    Args:
        participant_sizes: List of sizes for all participants on one side

    Returns:
        GroupGrappleSize with calculated totals
    """
    if not participant_sizes:
        return GroupGrappleSize(
            participant_sizes=[],
            total_size=0,
            largest_size="size_1",
            participant_count=0,
        )

    total = sum(SIZE_ORDER.get(s, 0) for s in participant_sizes)
    largest = max(participant_sizes, key=lambda s: SIZE_ORDER.get(s, 0))

    return GroupGrappleSize(
        participant_sizes=participant_sizes,
        total_size=total,
        largest_size=largest,
        participant_count=len(participant_sizes),
    )


def can_ram(
    attacker_size: SizeClass,
    target_size: SizeClass,
    attacker_has_mount: bool = True,
) -> tuple[bool, str]:
    """Check if a ram can be initiated.

    Per PR2 ~4152-4155:
    - Ram is a melee attack against adjacent character
    - Requires appropriate mount

    Args:
        attacker_size: Size of the ramming mech
        target_size: Size of the target mech
        attacker_has_mount: Whether attacker has appropriate mount

    Returns:
        Tuple of (can_ram, reason)
    """
    if not attacker_has_mount:
        return False, "Attacker lacks appropriate mount for ramming"

    return True, "Ram can be initiated"


def attempt_ram(
    attempt: RamAttempt,
    terrain_occupancies: dict[HexCoord, bool] | None = None,
    attacker_position: HexCoord | None = None,
    target_position: HexCoord | None = None,
) -> RamResult:
    """Resolve a ram attempt.

    Per PR2 ~4152-4155:
    "Ramming is a melee attack made against an adjacent character with the aim of
    knocking them down or back. If your attack is successful, your target is knocked
    Prone and you may also knock your target back up to 1 space directly away from you."

    Knockback is blocked by terrain occupancies (obstructions, other mechs, etc.).

    Args:
        attempt: RamAttempt with all required parameters
        terrain_occupancies: Map of occupied hexes for knockback blocking
        attacker_position: Attacker's current hex position (for direction calc)
        target_position: Target's current hex position (for direction calc)

    Returns:
        RamResult with full resolution details
    """
    if not attempt.hit:
        return RamResult(
            hit=False,
            target_becomes_prone=False,
            reason="Ram attack missed",
        )

    knockback_total = 1 + attempt.knockback_bonus

    blocked = False
    blocked_at: HexCoord | None = None

    if (
        terrain_occupancies
        and attacker_position
        and target_position
        and knockback_total > 0
    ):
        direction = get_knockback_direction(attacker_position, target_position)
        for i in range(1, knockback_total + 1):
            check_pos = HexCoord(
                q=target_position.q + (direction.q * i),
                r=target_position.r + (direction.r * i),
            )
            if terrain_occupancies.get(check_pos, False):
                blocked = True
                blocked_at = check_pos
                break

    effective_knockback = 0 if blocked else knockback_total

    return RamResult(
        hit=True,
        target_becomes_prone=True,
        knockback_spaces=effective_knockback,
        knockback_blocked=blocked,
        reason=f"Ram successful: target becomes prone, knockback {'blocked' if blocked else f'{effective_knockback} spaces'}",
    )


def resolve_disarm_on_grapple(
    grapple_result: GrappleResult,
    target_has_mount: bool,
) -> DisarmResult:
    """Resolve disarm attempt during grapple.

    Per PR2 ~4158:
    "disarming... or damaging it so that it cannot do the same to you"

    Args:
        grapple_result: The grapple result from attempt_grapple
        target_has_mount: Whether target has a mount to disarm

    Returns:
        DisarmResult with outcome details
    """
    if not grapple_result.grapple_initiated:
        return DisarmResult(
            attempted=False,
            successful=False,
            reason="Grapple not initiated",
        )

    if not target_has_mount:
        return DisarmResult(
            attempted=True,
            successful=False,
            reason="Target has no mount to disarm",
        )

    return DisarmResult(
        attempted=True,
        successful=True,
        target_lost_mount=True,
        reason="Target disarmed during grapple",
    )


def get_knockback_direction(
    from_coord: HexCoord,
    to_coord: HexCoord,
) -> HexCoord:
    """Calculate direction from attacker to target for knockback.

    Args:
        from_coord: Attacker's position
        to_coord: Target's position

    Returns:
        HexCoord representing direction vector
    """
    dq = to_coord.q - from_coord.q
    dr = to_coord.r - from_coord.r
    return HexCoord(q=dq, r=dr)


def is_valid_grapple_size(size: SizeClass) -> bool:
    """Check if a size is valid for grappling.

    All sizes can grapple, but size affects who is "larger" in the grapple.

    Args:
        size: Size class to check

    Returns:
        True if size is valid for grappling
    """
    return size in SIZE_ORDER


def is_larger_in_grapple(
    size_a: SizeClass,
    size_b: SizeClass,
) -> tuple[bool, Literal["a", "b", "tie"]]:
    """Determine which size is larger for grapple purposes.

    Args:
        size_a: First participant's size
        size_b: Second participant's size

    Returns:
        Tuple of (is_a_larger, comparison_result)
    """
    comparison = _compare_sizes(size_a, size_b)
    return comparison == "a", comparison
