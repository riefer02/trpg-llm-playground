"""Overwatch reaction trigger detection for Lancer combat.

This module provides type-safe helpers for detecting overwatch opportunities when
combatants start movement inside enemy weapon threat ranges, and for movement
events that enter/leave threat when granted by talents.

Overwatch Rules (per PR2 ~4395-4401, ~4005-4020):
- Trigger: Any enemy STARTS any movement (move, boost, etc) inside weapon threat range
- Threat: Default 1 for all weapons unless listed otherwise
- Reaction: Can immediately make a skirmish action as a reaction using that weapon
- Budget: Each mech can use overwatch once per round (default)
- Prevention: Disengage action prevents overwatch triggers for the turn
- Prevention: Hidden/Invisible status prevents reactions targeting the mover
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.ids import CombatantId, WeaponId
from core.shared.effects import ReactionTriggerEffect, ReactionTriggerEvent
from core.mech.grid import HexCoord

if TYPE_CHECKING:
    from core.mech.combat_state import MechCombatScenario, CombatantState


class OverwatchOpportunity(FrozenModel):
    """A detected overwatch opportunity for a single enemy.

    This represents one enemy's ability to potentially react with overwatch
    when a combatant starts movement in their weapon's threat range.
    """

    reactor_id: CombatantId = Field(
        ..., description="ID of enemy who can react"
    )
    weapon_id: WeaponId = Field(
        ..., description="Weapon with threat covering position"
    )
    weapon_threat: int = Field(
        ..., ge=1, description="Threat range of the weapon"
    )
    target_id: CombatantId = Field(
        ..., description="ID of moving combatant (target)"
    )
    target_position: HexCoord = Field(
        ..., description="Mover's starting position"
    )
    can_react: bool = Field(
        ..., description="True if reactor has available reaction budget"
    )
    prevention_reason: str | None = Field(
        default=None, description="Reason if reaction is prevented"
    )


class OverwatchTriggerResult(FrozenModel):
    """Result of checking overwatch triggers at movement start.

    Per PR2 rules, this check happens BEFORE movement occurs - the trigger
    condition is starting movement inside threat range, not entering it.
    """

    opportunities: list[OverwatchOpportunity] = Field(
        default_factory=list,
        description="List of overwatch opportunities detected"
    )
    reactions_prevented: bool = Field(
        default=False,
        description="True if mover's status prevents all overwatch reactions"
    )
    prevention_reason: str | None = Field(
        default=None,
        description="Global reason if all reactions are prevented"
    )


def _get_weapons_with_threat(
    combatant: "CombatantState",
) -> list[tuple[WeaponId, int]]:
    """Get all weapons with threat ranges from combatant inventory.

    Returns a list of (weapon_id, threat_range) tuples. Default threat is 1
    for all weapons unless specified otherwise.

    Args:
        combatant: The combatant whose weapons to check

    Returns:
        List of (weapon_id, threat_range) tuples for weapons with threat
    """
    from core.mech.compendium import get_weapon_definition
    from core.mech.weapon import resolve_weapon_profile

    weapons_with_threat: list[tuple[WeaponId, int]] = []

    if combatant.inventory is None:
        return weapons_with_threat

    for mount in combatant.inventory.mounts:
        for weapon in mount.weapons:
            # Skip destroyed weapons
            if weapon.destroyed:
                continue

            # Skip thrown weapons that haven't been retrieved
            if weapon.thrown_coord is not None:
                continue

            # Look up weapon definition to check for threat range
            weapon_def = get_weapon_definition(weapon.weapon_id)
            if weapon_def is None:
                continue

            profile = resolve_weapon_profile(weapon_def)

            # Check for threat range in the weapon's ranges
            threat_value: int | None = None
            for range_entry in profile.ranges:
                if range_entry.range_type == "threat":
                    threat_value = range_entry.value
                    break
            if threat_value is None:
                # Default threat for all weapons (PR2 ~4005-4020)
                threat_value = 1
            weapons_with_threat.append((WeaponId(weapon.weapon_id), threat_value))

    return weapons_with_threat


def _has_available_overwatch(
    combatant: "CombatantState",
) -> tuple[bool, str | None]:
    """Check if combatant has unused overwatch this round.

    Per PR2 rules, mechs can use overwatch once per round by default.

    Args:
        combatant: The combatant to check

    Returns:
        Tuple of (has_available, reason_if_not)
    """
    overwatch_uses = combatant.per_round_reactions.get("overwatch", 0)

    # Default: 1 overwatch per round
    max_overwatch = 1

    if overwatch_uses >= max_overwatch:
        return False, "Overwatch already used this round"

    return True, None


def _collect_reaction_triggers(
    combatant: "CombatantState",
) -> list[ReactionTriggerEffect]:
    """Collect reaction trigger effects from active combatant effects."""
    triggers: list[ReactionTriggerEffect] = []
    triggers.extend(combatant.reaction_triggers)
    for effect in combatant.talent_effects:
        triggers.extend(effect.reaction_triggers)
    for effect in combatant.frame_trait_effects:
        triggers.extend(effect.reaction_triggers)
    for mode in combatant.active_mode_effects:
        triggers.extend(mode.effects.reaction_triggers)
    if combatant.core_power_active and combatant.core_power_effects:
        triggers.extend(combatant.core_power_effects.reaction_triggers)
    return triggers


def _condition_allows_trigger(
    condition,
    weapon_def,
) -> bool:
    """Check if a reaction trigger condition is satisfied for a weapon."""
    if condition is None:
        return True
    if isinstance(condition, str):
        if condition == "cqb_overwatch":
            return weapon_def is not None and weapon_def.weapon_type == "cqb"
        return False
    return False


def _get_allowed_overwatch_triggers(
    combatant: "CombatantState",
    weapon_def,
) -> set[ReactionTriggerEvent]:
    """Get allowed overwatch trigger events for a given weapon."""
    allowed: set[ReactionTriggerEvent] = {"enemy_starts_movement_in_threat"}
    for trigger in _collect_reaction_triggers(combatant):
        if trigger.reaction_id != "overwatch":
            continue
        if not _condition_allows_trigger(trigger.condition, weapon_def):
            continue
        allowed.update(trigger.trigger_events)
    return allowed


def _extract_path_coords(
    mover: "CombatantState",
    movement_path: list | None,
) -> list[HexCoord]:
    """Extract movement path coordinates including the mover's start."""
    coords: list[HexCoord] = []
    if mover.position is None:
        return coords
    coords.append(mover.position.coord)
    if not movement_path:
        return coords
    for step in movement_path:
        coord = getattr(step, "coord", step)
        if isinstance(coord, HexCoord):
            coords.append(coord)
    return coords


def _detect_movement_trigger_event(
    path_coords: list[HexCoord],
    reactor_coord: HexCoord,
    threat: int,
    allowed_events: set[ReactionTriggerEvent],
) -> ReactionTriggerEvent | None:
    """Detect the first applicable overwatch trigger event along a path."""
    if not path_coords:
        return None

    start_in_threat = path_coords[0].distance_to(reactor_coord) <= threat
    if start_in_threat and "enemy_starts_movement_in_threat" in allowed_events:
        return "enemy_starts_movement_in_threat"

    for prev_coord, next_coord in zip(path_coords[:-1], path_coords[1:]):
        prev_in = prev_coord.distance_to(reactor_coord) <= threat
        next_in = next_coord.distance_to(reactor_coord) <= threat
        if not prev_in and next_in and "enemy_enters_threat" in allowed_events:
            return "enemy_enters_threat"
        if prev_in and not next_in:
            if "enemy_leaves_threat" in allowed_events:
                return "enemy_leaves_threat"
            if "enemy_exits_threat" in allowed_events:
                return "enemy_exits_threat"

    return None


def check_overwatch_triggers_for_movement(
    scenario: "MechCombatScenario",
    mover: "CombatantState",
    movement_path: list | None,
    is_disengaging: bool = False,
    is_hidden: bool = False,
    is_invisible: bool = False,
) -> OverwatchTriggerResult:
    """Check if movement triggers any overwatch opportunities."""
    from core.mech.compendium import get_weapon_definition

    if is_disengaging:
        return OverwatchTriggerResult(
            opportunities=[],
            reactions_prevented=True,
            prevention_reason="Disengage prevents overwatch triggers",
        )

    if is_hidden:
        return OverwatchTriggerResult(
            opportunities=[],
            reactions_prevented=True,
            prevention_reason="Hidden status prevents reactions targeting this mover",
        )

    if is_invisible:
        return OverwatchTriggerResult(
            opportunities=[],
            reactions_prevented=True,
            prevention_reason="Invisible status prevents reactions targeting this mover",
        )

    if mover.position is None:
        return OverwatchTriggerResult(
            opportunities=[],
            reactions_prevented=False,
            prevention_reason="Mover has no position",
        )

    path_coords = _extract_path_coords(mover, movement_path)
    opportunities: list[OverwatchOpportunity] = []

    for combatant in scenario.combatants:
        if combatant.id == mover.id:
            continue
        if combatant.side == mover.side:
            continue
        if combatant.position is None:
            continue
        if combatant.resources.hp_current <= 0:
            continue
        if "stunned" in combatant.statuses:
            continue
        if "shutdown" in combatant.statuses:
            continue

        for weapon_id, weapon_threat in _get_weapons_with_threat(combatant):
            weapon_def = get_weapon_definition(weapon_id)
            if weapon_def is None:
                continue

            allowed_events = _get_allowed_overwatch_triggers(combatant, weapon_def)
            trigger_event = _detect_movement_trigger_event(
                path_coords,
                combatant.position.coord,
                weapon_threat,
                allowed_events,
            )
            if trigger_event is None:
                continue

            can_react, prevention_reason = _has_available_overwatch(combatant)
            opportunities.append(
                OverwatchOpportunity(
                    reactor_id=CombatantId(combatant.id),
                    weapon_id=weapon_id,
                    weapon_threat=weapon_threat,
                    target_id=CombatantId(mover.id),
                    target_position=mover.position.coord,
                    can_react=can_react,
                    prevention_reason=prevention_reason,
                )
            )

    return OverwatchTriggerResult(
        opportunities=opportunities,
        reactions_prevented=False,
        prevention_reason=None,
    )


def check_overwatch_triggers_at_movement_start(
    scenario: "MechCombatScenario",
    mover: "CombatantState",
    is_disengaging: bool = False,
    is_hidden: bool = False,
    is_invisible: bool = False,
) -> OverwatchTriggerResult:
    """Check if movement triggers any overwatch opportunities.

    Per PR2 4395-4401: "If any enemy STARTS any movement (move, boost, etc)
    inside the threat of one of your weapons, you can immediately make a
    skirmish action as a reaction against that target using that weapon."

    This function checks:
    1. If mover's status prevents reactions (Disengage, Hidden, Invisible)
    2. For each enemy combatant with melee weapons:
       - If enemy has available overwatch reaction
       - If mover is within any weapon's threat range

    Args:
        scenario: Current combat scenario with all combatants
        mover: The combatant who is starting movement
        is_disengaging: Whether the Disengage action was used this turn
        is_hidden: Whether mover has the hidden status
        is_invisible: Whether mover has the invisible status

    Returns:
        OverwatchTriggerResult with all detected opportunities
    """
    return check_overwatch_triggers_for_movement(
        scenario=scenario,
        mover=mover,
        movement_path=None,
        is_disengaging=is_disengaging,
        is_hidden=is_hidden,
        is_invisible=is_invisible,
    )


__all__ = [
    "OverwatchOpportunity",
    "OverwatchTriggerResult",
    "check_overwatch_triggers_at_movement_start",
    "check_overwatch_triggers_for_movement",
]
