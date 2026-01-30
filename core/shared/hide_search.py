"""Hide/Search resolution helpers for Lancer combat.

This module provides type-safe helpers for Hide and Search actions including:
- Hide action: Gain hidden condition with valid cover
- Search action: Contested check to reveal hidden targets
- Hidden condition lifecycle: Breaking conditions and re-detection
- Disengage action: Ignoring engagement during movement

Hide Rules (per PR2 ~4221-4237):
- Requires hard cover OR area/zone of soft cover (smoke cloud)
- Cannot hide if engaged (even if invisible)
- Hiding is always successful → gain hidden condition
- Hidden: cannot be directly targeted, exact location unknown
- Can still be hit by area attacks
- Breaking: attack, boost, reaction, lose cover/LOS, cover destroyed, lose invisibility

Search Rules (per PR2 ~4241-4249):
- Contested check: Searcher Systems vs Hidden target Agility
- Target must be in searcher's sensor range
- Pilot search: skill check, range 5
- Success → target loses hidden immediately
- Can be located again by any character

Disengage Rules (per PR2 ~4289-4291):
- Full action
- Until end of turn: movement ignores engagement, no reactions provoked
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType
from core.shared.dice import roll_dice
from core.mech.grid import HexCoord
from core.mech.terrain import TerrainMap
from core.shared.terrain import get_terrain_at


# =============================================================================
# Soft Cover Zone Tracking (for terrain primitives integration)
# =============================================================================


class SoftCoverZoneState(FrozenModel):
    """Runtime state for a soft cover zone with duration tracking.

    Used for smoke clouds, foliage areas, and other zones that provide
    soft cover for hiding but may expire after a set number of rounds.
    """

    zone_id: str
    coords: frozenset[HexCoord]
    zone_subtype: str  # "smoke", "foliage", "mist", "darkness"
    created_round: int | None = None
    duration_rounds: int | None = None  # None = permanent


def is_in_active_soft_cover_zone(
    zones: list[SoftCoverZoneState],
    coord: HexCoord,
    current_round: int | None = None,
) -> bool:
    """Check if coord is in an active (non-expired) soft cover zone.

    Args:
        zones: List of soft cover zone states
        coord: The coordinate to check
        current_round: Current round number for expiration check

    Returns:
        True if coord is in an active zone
    """
    for zone in zones:
        if coord not in zone.coords:
            continue

        # Check if zone has expired
        if zone.duration_rounds is not None and current_round is not None:
            if zone.created_round is not None:
                elapsed = current_round - zone.created_round
                if elapsed >= zone.duration_rounds:
                    continue  # Zone expired

        return True

    return False


def check_soft_cover_for_hide(
    terrain: TerrainMap | None,
    soft_cover_zones: list[SoftCoverZoneState],
    target_coord: HexCoord,
    current_round: int | None = None,
    min_adjacent_hexes: int = 3,
) -> bool:
    """Combined check for soft cover area: terrain OR active zone.

    This integrates terrain-based soft cover with zone-based soft cover
    from terrain primitives.

    Args:
        terrain: The terrain map (for terrain-based soft cover)
        soft_cover_zones: List of active soft cover zones
        target_coord: The coordinate to check
        current_round: Current round number for zone expiration
        min_adjacent_hexes: Minimum adjacent hexes for terrain-based area

    Returns:
        True if target is in a soft cover area/zone
    """
    # Check zone-based soft cover first
    if is_in_active_soft_cover_zone(zones=soft_cover_zones, coord=target_coord, current_round=current_round):
        return True

    # Fall back to terrain-based soft cover
    return is_soft_cover_area(terrain=terrain, target_coord=target_coord, min_adjacent_hexes=min_adjacent_hexes)


# =============================================================================
# Original Hide/Search Types and Functions
# =============================================================================


class HideAttempt(FrozenModel):
    """Input for hide action resolution."""

    has_hard_cover: bool = False
    is_in_soft_cover_area: bool = False
    has_los: bool = True
    is_invisible: bool = False
    is_engaged: bool = False
    current_conditions: list[StatusType] = Field(default_factory=list)


class HideResult(FrozenModel):
    """Result of hide action resolution."""

    can_hide: bool
    hidden_condition_applied: bool = False
    reason: str = ""


class SearchAttempt(FrozenModel):
    """Input for search action resolution."""

    searcher_systems_bonus: int = 0
    target_agility_bonus: int = 0
    target_in_sensor_range: bool = False
    is_pilot_search: bool = False
    pilot_in_range_5: bool = False
    pilot_skill_bonus: int = 0


class SearchResult(FrozenModel):
    """Result of search action resolution."""

    search_successful: bool
    target_revealed: bool = False
    search_roll: int | None = None
    search_total: int | None = None
    target_roll: int | None = None
    target_total: int | None = None
    reason: str = ""


class DisengageResult(FrozenModel):
    """Result of disengage action resolution."""

    disengage_successful: bool
    ignores_engagement: bool = True
    prevents_reactions: bool = True
    duration_turn_end: bool = True
    reason: str = ""


class HiddenConditionBreakCheck(FrozenModel):
    """Check if hidden condition should be broken."""

    hidden_broken: bool
    break_reason: str = ""
    break_type: Literal[
        "attack",
        "boost",
        "reaction",
        "cover_lost",
        "cover_destroyed",
        "invisibility_lost",
        "none",
    ] = "none"


def can_hide(
    has_hard_cover: bool,
    is_in_soft_cover_area: bool,
    has_los: bool,
    is_invisible: bool,
    is_engaged: bool,
) -> tuple[bool, str]:
    """Check if a character can perform the Hide action.

    Per PR2 ~4221-4228:
    - Requires hard cover OR area/zone of soft cover (smoke cloud)
    - Cannot hide if engaged (even if invisible)
    - Lack of line of sight is always sufficient
    - If invisible, can always hide without cover

    Args:
        has_hard_cover: Whether target has adjacent hard cover
        is_in_soft_cover_area: Whether target is in a zone/area of soft cover
        has_los: Whether attacker has line of sight to target
        is_invisible: Whether target is invisible
        is_engaged: Whether target is engaged with another character

    Returns:
        Tuple of (can_hide, reason)
    """
    if is_engaged:
        return False, "Cannot hide while engaged with another character"

    if is_invisible:
        return True, "Invisible characters can always hide"

    if not has_los:
        return True, "Lack of line of sight is sufficient for hiding"

    if has_hard_cover:
        return True, "Has hard cover for hiding"

    if is_in_soft_cover_area:
        return True, "In area/zone of soft cover for hiding"

    return (
        False,
        "Need hard cover, soft cover area, lack of LOS, or invisibility to hide",
    )


def is_soft_cover_area(
    terrain: TerrainMap | None,
    target_coord: HexCoord,
    min_adjacent_hexes: int = 3,
) -> bool:
    """Check if target is in a soft cover area/zone.

    Per PR2 ~4222-4225:
    "an area or zone of soft cover such as a smoke cloud (it must be an area
    or zone - other systems or talents that grant you soft cover won't work)"

    A valid area requires multiple adjacent hexes with soft cover, not just
    a single hex. This simulates zone effects like smoke grenades.

    Args:
        terrain: The terrain map
        target_coord: The coordinate to check
        min_adjacent_hexes: Minimum adjacent hexes with soft cover (default 3)

    Returns:
        True if in a soft cover area/zone
    """
    if terrain is None:
        return False

    target_terrain = get_terrain_at(terrain, target_coord)
    if target_terrain is None or not target_terrain.provides_soft_cover:
        return False

    adjacent_coords = target_coord.neighbors()
    adjacent_soft_cover = 0

    for adj_coord in adjacent_coords:
        adj_terrain = get_terrain_at(terrain, adj_coord)
        if adj_terrain and adj_terrain.provides_soft_cover:
            adjacent_soft_cover += 1

    total_soft_cover = 1 + adjacent_soft_cover
    return total_soft_cover >= min_adjacent_hexes


def attempt_hide(
    attempt: HideAttempt,
) -> HideResult:
    """Resolve a hide action.

    Per PR2 ~4229:
    "Hiding is always successful. After you hide, you gain the hidden condition."

    Args:
        attempt: HideAttempt with all required parameters

    Returns:
        HideResult with outcome details
    """
    can_hide_result, reason = can_hide(
        has_hard_cover=attempt.has_hard_cover,
        is_in_soft_cover_area=attempt.is_in_soft_cover_area,
        has_los=attempt.has_los,
        is_invisible=attempt.is_invisible,
        is_engaged=attempt.is_engaged,
    )

    if not can_hide_result:
        return HideResult(
            can_hide=False,
            hidden_condition_applied=False,
            reason=reason,
        )

    return HideResult(
        can_hide=True,
        hidden_condition_applied=True,
        reason=f"Hide successful: {reason}",
    )


def check_hidden_broken(
    current_conditions: list[StatusType],
    action_type: Literal["attack", "boost", "reaction", "none"],
    has_los: bool,
    has_cover: bool,
    is_invisible: bool,
    cover_destroyed: bool = False,
) -> HiddenConditionBreakCheck:
    """Check if the hidden condition should be broken.

    Per PR2 ~4233-4237:
    "Performing any attack (melee, ranged, or tech) or hostile action such as
    forcing a save that originates from your mech, taking the boost action, or
    taking a reaction with your mech will break hiding. You immediately lose
    hidden if you lose the benefit of cover due to line of sight (ie, a mech
    comes around a wall and can now draw unbroken line of sight to you) or if
    your cover disappears or is destroyed. If you're hiding and invisible,
    you lose hidden if you lose invisibility."

    Args:
        current_conditions: Current conditions on the character
        action_type: Action being taken (attack, boost, reaction, none)
        has_los: Whether attacker now has line of sight to hidden character
        has_cover: Whether hidden character still has cover benefit
        is_invisible: Whether hidden character is still invisible
        cover_destroyed: Whether cover was destroyed

    Returns:
        HiddenConditionBreakCheck with break details
    """
    if "hidden" not in current_conditions:
        return HiddenConditionBreakCheck(
            hidden_broken=False,
            break_reason="Not hidden",
            break_type="none",
        )

    if action_type == "attack":
        return HiddenConditionBreakCheck(
            hidden_broken=True,
            break_reason="Attacking breaks hidden condition",
            break_type="attack",
        )

    if action_type == "boost":
        return HiddenConditionBreakCheck(
            hidden_broken=True,
            break_reason="Boosting breaks hidden condition",
            break_type="boost",
        )

    if action_type == "reaction":
        return HiddenConditionBreakCheck(
            hidden_broken=True,
            break_reason="Taking a reaction breaks hidden condition",
            break_type="reaction",
        )

    if not is_invisible and "invisible" in current_conditions:
        return HiddenConditionBreakCheck(
            hidden_broken=True,
            break_reason="Lost invisibility",
            break_type="invisibility_lost",
        )

    if has_los and not has_cover and not is_invisible:
        return HiddenConditionBreakCheck(
            hidden_broken=True,
            break_reason="Lost cover benefit due to line of sight",
            break_type="cover_lost",
        )

    if cover_destroyed:
        return HiddenConditionBreakCheck(
            hidden_broken=True,
            break_reason="Cover was destroyed",
            break_type="cover_destroyed",
        )

    return HiddenConditionBreakCheck(
        hidden_broken=False,
        break_reason="Hidden condition maintained",
        break_type="none",
    )


def can_search(
    target_in_sensor_range: bool,
    is_pilot_search: bool,
    pilot_in_range_5: bool,
) -> tuple[bool, str]:
    """Check if a character can perform the Search action.

    Per PR2 ~4243-4247:
    "The searching party needs you to be in their sensor range... A character
    must be in range 5 of you to be revealed this way [pilot search]."

    Args:
        target_in_sensor_range: Whether target is in searcher's sensor range
        is_pilot_search: Whether this is a pilot-scale search
        pilot_in_range_5: Whether pilot is within range 5 of target

    Returns:
        Tuple of (can_search, reason)
    """
    if is_pilot_search:
        if not pilot_in_range_5:
            return False, "Pilot must be in range 5 to search for hidden target"
        return True, "Pilot can attempt to search"

    if not target_in_sensor_range:
        return False, "Target must be in sensor range to search"
    return True, "Target in sensor range, can search"


def attempt_search(
    attempt: SearchAttempt,
) -> SearchResult:
    """Resolve a search action.

    Per PR2 ~4242-4249:
    "The searching party chooses a character they suspect is hidden and makes
    a contested check, revealing their target on a success. The searching party
    needs you to be in their sensor range and makes a systems check. A hidden
    mech makes an agility check. If you're searching on foot as a pilot, you
    make a skill check... A character must be in range 5 of you to be revealed
    this way. Once a hidden target is detected, it immediately loses the hidden
    condition and can be located again by any character."

    Args:
        attempt: SearchAttempt with all required parameters

    Returns:
        SearchResult with contested check outcome
    """
    can_search_result, reason = can_search(
        target_in_sensor_range=attempt.target_in_sensor_range,
        is_pilot_search=attempt.is_pilot_search,
        pilot_in_range_5=attempt.pilot_in_range_5,
    )

    if not can_search_result:
        return SearchResult(
            search_successful=False,
            target_revealed=False,
            reason=reason,
        )

    if attempt.is_pilot_search:
        search_roll = roll_dice("1d20")
        search_total = search_roll + attempt.pilot_skill_bonus
        success = search_total >= 10

        return SearchResult(
            search_successful=success,
            target_revealed=success,
            search_roll=search_roll,
            search_total=search_total,
            reason=f"Pilot search: {search_roll}+{attempt.pilot_skill_bonus} vs DC 10",
        )

    search_roll = roll_dice("1d20")
    target_roll = roll_dice("1d20")

    search_total = search_roll + attempt.searcher_systems_bonus
    target_total = target_roll + attempt.target_agility_bonus
    success = search_total >= target_total

    return SearchResult(
        search_successful=success,
        target_revealed=success,
        search_roll=search_roll,
        search_total=search_total,
        target_roll=target_roll,
        target_total=target_total,
        reason=f"Search: {search_roll}+{attempt.searcher_systems_bonus} vs {target_roll}+{attempt.target_agility_bonus}",
    )


def resolve_disengage() -> DisengageResult:
    """Resolve a disengage action.

    Per PR2 ~4289-4291:
    "When you disengage, you attempt to extricate yourself safely from a dangerous
    situation... Until the end of your current turn, your movement ignores engagement
    and does not provoke reactions, such as overwatch."

    Disengage is always successful as a full action. The caller is responsible for
    tracking when the effect expires (end of current turn).

    Returns:
        DisengageResult with effect details
    """
    return DisengageResult(
        disengage_successful=True,
        ignores_engagement=True,
        prevents_reactions=True,
        duration_turn_end=True,
        reason="Disengage successful: movement ignores engagement and reactions until end of turn",
    )


def is_hidden_target_revealable(
    was_hidden: bool,
    search_successful: bool,
    is_in_sensor_range: bool,
) -> tuple[bool, str]:
    """Check if a hidden target can be revealed/located.

    Per PR2 ~4248-4249:
    "Once a hidden target is detected, it immediately loses the hidden condition
    and can be located again by any character."

    This function helps determine if a target that was hidden is now revealable
    to characters who weren't the searcher.

    Args:
        was_hidden: Whether the target had hidden condition
        search_successful: Whether the search was successful
        is_in_sensor_range: Whether the querying character has sensor range

    Returns:
        Tuple of (can_be_located, reason)
    """
    if not was_hidden:
        return False, "Target was not hidden"

    if not search_successful:
        return False, "Search was not successful"

    return True, "Hidden target revealed and can be located by all characters"


def get_cover_for_hiding(
    terrain: TerrainMap | None,
    target_coord: HexCoord,
) -> tuple[bool, bool, str]:
    """Get comprehensive cover status for hiding purposes.

    Checks all hiding requirements and returns details about cover validity.

    Args:
        terrain: The terrain map
        target_coord: The coordinate to check

    Returns:
        Tuple of (has_hard_cover, is_in_soft_cover_area, reason)
    """
    if terrain is None:
        return False, False, "No terrain, no cover available"

    target_terrain = get_terrain_at(terrain, target_coord)

    has_hard_cover = False
    adjacent_coords = target_coord.neighbors()
    for adj_coord in adjacent_coords:
        adj_terrain = get_terrain_at(terrain, adj_coord)
        if adj_terrain and adj_terrain.provides_hard_cover:
            has_hard_cover = True
            break

    if has_hard_cover:
        return True, False, "Has adjacent hard cover"

    is_in_soft_cover_area = is_soft_cover_area(terrain, target_coord)

    if is_in_soft_cover_area:
        return False, True, "In soft cover area/zone"

    if target_terrain and target_terrain.blocks_line_of_sight:
        return False, False, "Blocks line of sight but no cover for hiding"

    return False, False, "No valid cover for hiding"
