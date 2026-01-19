"""Overwatch reaction trigger detection for Lancer combat.

This module provides type-safe helpers for detecting overwatch opportunities when
combatants start movement inside enemy weapon threat ranges.

Overwatch Rules (per PR2 ~4395-4401):
- Trigger: Any enemy STARTS any movement (move, boost, etc) inside weapon threat range
- Threat: Default 1 for all weapons unless listed otherwise (melee weapons have threat)
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

    Only melee weapons have threat ranges. Returns a list of (weapon_id, threat_range)
    tuples. Default threat is 1 for melee weapons unless specified otherwise.

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
            for range_entry in profile.ranges:
                if range_entry.range_type == "threat":
                    threat = range_entry.value
                    weapons_with_threat.append((WeaponId(weapon.weapon_id), threat))
                    break

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
    # Check global prevention conditions
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

    # Mover must have a position
    if mover.position is None:
        return OverwatchTriggerResult(
            opportunities=[],
            reactions_prevented=False,
            prevention_reason="Mover has no position",
        )

    mover_coord = mover.position.coord
    opportunities: list[OverwatchOpportunity] = []

    # Check each combatant
    for combatant in scenario.combatants:
        # Skip self
        if combatant.id == mover.id:
            continue

        # Skip allies (same side)
        if combatant.side == mover.side:
            continue

        # Skip combatants without positions
        if combatant.position is None:
            continue

        # Skip destroyed/incapacitated combatants
        if combatant.resources.hp_current <= 0:
            continue

        # Skip stunned combatants (cannot take reactions)
        if "stunned" in combatant.statuses:
            continue

        # Skip shutdown combatants (cannot take reactions)
        if "shutdown" in combatant.statuses:
            continue

        # Get weapons with threat ranges
        weapons_with_threat = _get_weapons_with_threat(combatant)
        if not weapons_with_threat:
            continue

        # Calculate distance from mover to this combatant
        distance = mover_coord.distance_to(combatant.position.coord)

        # Check if mover is within threat range of any weapon
        for weapon_id, threat_range in weapons_with_threat:
            if distance <= threat_range:
                # Check if combatant can react
                can_react, prevention_reason = _has_available_overwatch(combatant)

                opportunities.append(
                    OverwatchOpportunity(
                        reactor_id=CombatantId(combatant.id),
                        weapon_id=weapon_id,
                        weapon_threat=threat_range,
                        target_id=CombatantId(mover.id),
                        target_position=mover_coord,
                        can_react=can_react,
                        prevention_reason=prevention_reason,
                    )
                )

    return OverwatchTriggerResult(
        opportunities=opportunities,
        reactions_prevented=False,
        prevention_reason=None,
    )


__all__ = [
    "OverwatchOpportunity",
    "OverwatchTriggerResult",
    "check_overwatch_triggers_at_movement_start",
]
