"""Tech action resolution helpers for mech combat.

Implements resolution logic for quick tech actions:
- Scan: Reveals information about target (not contested)
- Bolster: Grants accuracy bonus to target's next check/save
- Lock On: Grants status that boosts ally accuracy
- Invade: Tech attack vs E-defense, deals heat, applies conditions
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.enums import StatusType
from core.shared.dice import DiceExpression
from core.mech.combat_resolution import DiceRollResult, ResolutionSettings


ScanInfoType = Literal["stats", "hidden_info", "public_info"]


class ScanResult(FrozenModel):
    """Result of a Scan action.

    Scan reveals information about the target. This is not contested -
    it simply reveals the requested information categories.
    """

    action_id: str = "scan"
    actor_id: str
    target_id: str
    success: bool = True
    revealed_info: list[ScanInfoType] = Field(
        default_factory=list, description="Information categories revealed by the scan"
    )


class BolsterResult(FrozenModel):
    """Result of a Bolster action.

    Bolster requires a Systems check (not contested against target).
    On success, the target gains +2 accuracy on their next check or save
    before the end of their next turn.
    """

    action_id: str = "bolster"
    actor_id: str
    target_id: str
    success: bool = True
    accuracy_bonus: int = Field(default=2, ge=0, description="Accuracy bonus granted")
    duration: Literal["end_of_next_turn"] = "end_of_next_turn"
    systems_roll: DiceRollResult | None = Field(
        default=None, description="Systems check result"
    )
    check_total: int | None = Field(
        default=None, description="Total systems check value"
    )


class LockOnResult(FrozenModel):
    """Result of a Lock On action.

    Lock On is not contested. It grants the target a lock_on status
    that remains until consumed by a hostile attack. Allies attacking
    the locked target gain +1 accuracy.
    """

    action_id: str = "lock_on"
    actor_id: str
    target_id: str
    success: bool = True
    accuracy_bonus: int = Field(default=1, ge=0, description="Accuracy bonus to allies")
    duration: Literal["until_consumed"] = "until_consumed"
    status_granted: StatusType = "lock_on"


class InvadeResult(FrozenModel):
    """Result of an Invade action.

    Invade is a tech attack: Systems check vs target's E-defense.
    On hit: target takes 2 heat and becomes impaired and slowed
    until the end of their next turn.
    """

    action_id: str = "invade"
    actor_id: str
    target_id: str
    success: bool
    hit: bool
    systems_roll: DiceRollResult | None = Field(
        default=None, description="Systems check result"
    )
    check_total: int | None = Field(
        default=None, description="Total systems check value"
    )
    target_e_defense: int | None = Field(
        default=None, description="Target's E-defense at time of check"
    )
    heat_applied: int | None = Field(
        default=None, ge=0, description="Heat dealt to target"
    )
    conditions_applied: list[StatusType] = Field(
        default_factory=list, description="Conditions inflicted on target"
    )
    duration: Literal["end_of_next_turn"] = "end_of_next_turn"


def _roll_systems_check(
    attacker_systems: int,
    settings: ResolutionSettings | None = None,
) -> tuple[list[int], int]:
    """Roll a systems check for tech actions.

    Args:
        attacker_systems: The actor's systems score
        settings: Optional resolution settings for forced rolls

    Returns:
        Tuple of (roll results, total value)
    """
    base_bonus = max(attacker_systems, 0)
    dice_count = max(1 + base_bonus, 1)

    if settings and settings.forced_rolls:
        rolls = list(settings.forced_rolls[:dice_count])
    else:
        rolls = DiceExpression.parse(f"{dice_count}d6").roll()

    total = sum(rolls) + base_bonus
    return rolls, total


def _create_roll_result(rolls: list[int]) -> DiceRollResult:
    """Create a DiceRollResult from roll values."""
    return DiceRollResult(rolls=rolls, chosen=sorted(rolls)[:1] if rolls else [])


def resolve_scan(
    *,
    actor_id: str,
    target_id: str,
    scan_options: list[ScanInfoType],
) -> ScanResult:
    """Resolve a Scan action.

    Scan reveals information about the target. It is not contested -
    the information is simply revealed based on the scan options.

    Args:
        actor_id: ID of the actor performing the scan
        target_id: ID of the target being scanned
        scan_options: Categories of information to reveal

    Returns:
        ScanResult with the revealed information
    """
    return ScanResult(
        action_id="scan",
        actor_id=actor_id,
        target_id=target_id,
        success=True,
        revealed_info=list(scan_options),
    )


def resolve_bolster(
    *,
    actor_id: str,
    target_id: str,
    attacker_systems: int,
    accuracy_bonus: int = 2,
    settings: ResolutionSettings | None = None,
) -> BolsterResult:
    """Resolve a Bolster action.

    Bolster requires a Systems check. On success, the target gains
    +accuracy_bonus accuracy on their next check or save before the
    end of their next turn.

    In Lancer rules, Bolster is typically not contested - it just requires
    the pilot/mech to make a systems check. Success is automatic on any roll.

    Args:
        actor_id: ID of the actor performing the bolster
        target_id: ID of the target receiving the bonus
        attacker_systems: The actor's systems score
        accuracy_bonus: Accuracy bonus to grant (default 2)
        settings: Optional resolution settings for forced rolls

    Returns:
        BolsterResult with the action outcome
    """
    rolls, total = _roll_systems_check(attacker_systems, settings)

    return BolsterResult(
        action_id="bolster",
        actor_id=actor_id,
        target_id=target_id,
        success=True,
        accuracy_bonus=accuracy_bonus,
        systems_roll=_create_roll_result(rolls),
        check_total=total,
    )


def resolve_lock_on(
    *,
    actor_id: str,
    target_id: str,
    accuracy_bonus: int = 1,
) -> LockOnResult:
    """Resolve a Lock On action.

    Lock On is not contested. It grants the target a lock_on status
    that remains until consumed by a hostile attack. Allies attacking
    the locked target gain the specified accuracy bonus.

    Args:
        actor_id: ID of the actor performing lock on
        target_id: ID of the target to lock onto
        accuracy_bonus: Accuracy bonus to grant to allies (default 1)

    Returns:
        LockOnResult with the action outcome
    """
    return LockOnResult(
        action_id="lock_on",
        actor_id=actor_id,
        target_id=target_id,
        success=True,
        accuracy_bonus=accuracy_bonus,
    )


def resolve_invade(
    *,
    actor_id: str,
    target_id: str,
    attacker_systems: int,
    target_e_defense: int,
    heat_on_hit: int = 2,
    conditions: list[StatusType] | None = None,
    settings: ResolutionSettings | None = None,
) -> InvadeResult:
    """Resolve an Invade action.

    Invade is a tech attack: Systems check vs target's E-defense.
    On hit: target takes heat_on_hit heat and becomes impaired and slowed
    until the end of their next turn.

    Args:
        actor_id: ID of the actor performing invade
        target_id: ID of the target being invaded
        attacker_systems: The actor's systems score for the tech attack
        target_e_defense: The target's E-defense
        heat_on_hit: Heat to deal on hit (default 2)
        conditions: Conditions to inflict (default: impaired, slowed)
        settings: Optional resolution settings for forced rolls

    Returns:
        InvadeResult with the attack outcome
    """
    if conditions is None:
        conditions = ["impaired", "slowed"]

    rolls, total = _roll_systems_check(attacker_systems, settings)
    hit = total >= target_e_defense

    heat_applied = None
    conditions_applied = []

    if hit:
        heat_applied = heat_on_hit
        conditions_applied = list(conditions)

    return InvadeResult(
        action_id="invade",
        actor_id=actor_id,
        target_id=target_id,
        success=hit,
        hit=hit,
        systems_roll=_create_roll_result(rolls),
        check_total=total,
        target_e_defense=target_e_defense,
        heat_applied=heat_applied,
        conditions_applied=conditions_applied,
    )
