"""Self Destruct action resolution primitives for Lancer TTRPG.

Implements resolution logic for Self Destruct per PR2 4329-4336:
- Quick action to initiate reactor meltdown
- 1-2 turn delay (player choice)
- Explodes like reactor meltdown (4d6 explosive, burst 2)
- Agility save halves damage
- Annihilates mech, kills pilot/passenger
- Creates wreckage

Resolution Pattern:
1. resolve_self_destruct_initiation() - Start the countdown
2. apply_self_destruct_initiation() - Apply countdown state
3. resolve_self_destruct_explosion() - Resolve the explosion
4. apply_self_destruct_explosion() - Apply destruction effects
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import DamageType, SizeClass
from core.shared.dice import DiceExpression
from core.shared.saves import SaveRequest, SaveResult, resolve_save
from core.shared.heat import MeltdownState, trigger_meltdown
from core.shared.battlefield_objects import WreckageState
from core.mech.grid import HexPosition, HexCoord, hex_add, hexes_in_radius
from core.mech.combat_state import CombatantState

if TYPE_CHECKING:
    from core.mech.combat_state import CombatStats, CombatResources


class SelfDestructRule(FrozenModel):
    """Rule configuration for Self Destruct action."""

    burst_radius: int = Field(default=2, ge=0, description="Burst radius for explosion")
    damage: DiceExpression = Field(
        default_factory=lambda: DiceExpression.parse("4d6"),
        description="Damage expression for explosion",
    )
    damage_type: DamageType = "explosive"
    min_delay_turns: int = Field(default=1, ge=1, description="Minimum delay turns")
    max_delay_turns: int = Field(default=2, ge=1, description="Maximum delay turns")
    save_skill: Literal["agility"] = "agility"
    save_halves_damage: bool = True


DEFAULT_SELF_DESTRUCT_RULES = SelfDestructRule()


class SelfDestructInput(FrozenModel):
    """Input for Self Destruct initiation."""

    actor_id: str = Field(..., description="ID of the pilot initiating self destruct")
    mech_id: str = Field(..., description="ID of the mech to self destruct")
    delay_turns: int = Field(..., ge=1, description="Turns until explosion (1-2)")
    rules: SelfDestructRule | None = Field(
        default=None, description="Override resolution rules"
    )


class SelfDestructExplosionInput(FrozenModel):
    """Input for Self Destruct explosion resolution."""

    mech_id: str = Field(..., description="ID of the exploding mech")
    mech_position: HexPosition = Field(..., description="Position of the mech")
    rules: SelfDestructRule | None = Field(
        default=None, description="Override resolution rules"
    )


class SelfDestructResolutionResult(FrozenModel):
    """Complete result of Self Destruct initiation (pure logic)."""

    actor_id: str = Field(..., description="ID of the pilot")
    mech_id: str = Field(..., description="ID of the mech")
    delay_turns: int = Field(..., description="Turns until explosion")
    countdown_started: bool = Field(
        default=False, description="Whether the countdown was started"
    )
    meltdown_state: MeltdownState | None = Field(
        default=None, description="Meltdown countdown state"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class TargetExplosionResult(FrozenModel):
    """Result of explosion affecting a single target."""

    target_id: str = Field(..., description="ID of the affected target")
    target_position: HexPosition | None = Field(
        default=None, description="Target's position"
    )
    in_burst_radius: bool = Field(
        default=False, description="Whether target is in burst radius"
    )
    distance: int | None = Field(
        default=None, description="Distance from explosion center"
    )
    damage_dealt: int = Field(default=0, ge=0, description="Damage dealt")
    save_result: SaveResult | None = Field(
        default=None, description="Save result if applicable"
    )
    damage_halved_by_save: bool = Field(
        default=False, description="Whether damage was halved by save"
    )


class SelfDestructExplosionResult(FrozenModel):
    """Complete result of Self Destruct explosion (pure logic)."""

    mech_id: str = Field(..., description="ID of the exploding mech")
    mech_position: HexPosition | None = Field(
        default=None, description="Mech's position at explosion"
    )
    burst_radius: int = Field(..., description="Burst radius")
    damage_expression: DiceExpression = Field(
        ..., description="Damage expression rolled"
    )
    damage_rolls: list[int] = Field(
        default_factory=list, description="Individual damage dice rolls"
    )
    total_damage: int = Field(..., description="Total damage before saves")
    target_results: list[TargetExplosionResult] = Field(
        default_factory=list, description="Results for each affected target"
    )
    mech_destroyed: bool = Field(
        default=False, description="Whether the mech was destroyed"
    )
    pilot_killed: bool = Field(
        default=False, description="Whether the pilot was killed"
    )
    wreckage: WreckageState | None = Field(
        default=None, description="Wreckage object created"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class SelfDestructExplosionApplicationResult(FrozenModel):
    """Result of applying Self Destruct explosion to combatant states."""

    updated_mech: CombatantState | None = Field(
        default=None, description="Destroyed mech state"
    )
    updated_pilot: CombatantState | None = Field(
        default=None, description="Killed pilot state"
    )
    wreckage_created: bool = Field(
        default=False, description="Whether wreckage was created"
    )
    affected_targets: list[TargetExplosionResult] = Field(
        default_factory=list, description="Results for each affected target"
    )


def resolve_self_destruct_initiation(
    input: SelfDestructInput,
    rules: SelfDestructRule | None = None,
) -> SelfDestructResolutionResult:
    """Resolve Self Destruct initiation per PR2 4329-4336.

    Self Destruct is a Quick Action that starts a reactor meltdown countdown.
    The player chooses 1-2 turns delay. At the end of the specified turn,
    the mech explodes.

    Args:
        input: Self Destruct input with mech and delay information
        rules: Optional rule configuration (overrides input.rules if provided)

    Returns:
        Detailed breakdown of the self destruct initiation
    """
    if rules is None:
        rules = input.rules if input.rules else DEFAULT_SELF_DESTRUCT_RULES

    errors: list[str] = []

    if input.delay_turns < rules.min_delay_turns:
        errors.append(f"Delay must be at least {rules.min_delay_turns} turn(s)")
    if input.delay_turns > rules.max_delay_turns:
        errors.append(f"Delay cannot exceed {rules.max_delay_turns} turn(s)")

    if errors:
        return SelfDestructResolutionResult(
            actor_id=input.actor_id,
            mech_id=input.mech_id,
            delay_turns=input.delay_turns,
            countdown_started=False,
            meltdown_state=None,
            validation_errors=errors,
        )

    meltdown_state = MeltdownState(
        turns_remaining=input.delay_turns,
        triggered_by_overheat=False,
        exposed_applied=False,
        is_immediate=True,
    )

    return SelfDestructResolutionResult(
        actor_id=input.actor_id,
        mech_id=input.mech_id,
        delay_turns=input.delay_turns,
        countdown_started=True,
        meltdown_state=meltdown_state,
        validation_errors=errors,
    )


def apply_self_destruct_initiation(
    mech: CombatantState,
    result: SelfDestructResolutionResult,
) -> CombatantState:
    """Apply Self Destruct initiation to mech state.

    Updates mech with the meltdown countdown state.

    Args:
        mech: Current mech state
        result: Resolution result to apply

    Returns:
        Updated mech with countdown state
    """
    if not result.countdown_started or result.meltdown_state is None:
        return mech

    updated_mech = mech.model_copy(update={"meltdown_state": result.meltdown_state})
    return updated_mech


def resolve_self_destruct_explosion(
    input: SelfDestructExplosionInput,
    all_combatants: list[CombatantState],
    rules: SelfDestructRule | None = None,
) -> SelfDestructExplosionResult:
    """Resolve Self Destruct explosion effects per PR2 4329-4336.

    When the countdown expires, the mech explodes dealing 4d6 explosive
    damage in a burst 2 radius. Targets can make an agility save to
    halve damage.

    Args:
        input: Explosion input with mech position
        all_combatants: List of all combatants to check for affected targets
        rules: Optional rule configuration

    Returns:
        Detailed breakdown of explosion effects
    """
    if rules is None:
        rules = input.rules if input.rules else DEFAULT_SELF_DESTRUCT_RULES

    errors: list[str] = []

    if input.mech_position is None:
        errors.append("Mech has no position - cannot resolve explosion")

    if errors:
        return SelfDestructExplosionResult(
            mech_id=input.mech_id,
            mech_position=input.mech_position,
            burst_radius=rules.burst_radius,
            damage_expression=rules.damage,
            damage_rolls=[],
            total_damage=0,
            target_results=[],
            mech_destroyed=False,
            pilot_killed=False,
            wreckage=None,
            validation_errors=errors,
        )

    damage_dice = rules.damage.roll()
    total_damage = sum(damage_dice)

    center_coord = input.mech_position.coord
    affected_hexes = hexes_in_radius(center_coord, rules.burst_radius)

    target_results: list[TargetExplosionResult] = []

    for combatant in all_combatants:
        if combatant.id == input.mech_id:
            continue

        if combatant.position is None:
            continue

        distance = center_coord.distance_to(combatant.position.coord)

        if distance > rules.burst_radius:
            continue

        damage_dealt = total_damage
        save_result: SaveResult | None = None
        damage_halved = False

        if rules.save_skill == "agility":
            save_req = SaveRequest(
                save_type="agility",
                save_target=10,
            )
            save_result = resolve_save(save_req)
            if save_result.success and rules.save_halves_damage:
                damage_dealt = (damage_dealt + 1) // 2
                damage_halved = True

        target_results.append(
            TargetExplosionResult(
                target_id=combatant.id,
                target_position=combatant.position,
                in_burst_radius=True,
                distance=distance,
                damage_dealt=damage_dealt,
                save_result=save_result,
                damage_halved_by_save=damage_halved,
            )
        )

    return SelfDestructExplosionResult(
        mech_id=input.mech_id,
        mech_position=input.mech_position,
        burst_radius=rules.burst_radius,
        damage_expression=rules.damage,
        damage_rolls=damage_dice,
        total_damage=total_damage,
        target_results=target_results,
        mech_destroyed=True,
        pilot_killed=True,
        wreckage=None,
        validation_errors=errors,
    )


def apply_self_destruct_explosion(
    mech: CombatantState,
    pilot: CombatantState | None,
    result: SelfDestructExplosionResult,
) -> SelfDestructExplosionApplicationResult:
    """Apply Self Destruct explosion to combatant states.

    Destroys the mech, kills the pilot, creates wreckage, and applies
    damage to all affected targets.

    Args:
        mech: Current mech state
        pilot: Current pilot state (if inside mech)
        result: Explosion result to apply

    Returns:
        Updated combatant states with explosion effects
    """
    from core.shared.enums import SizeClass

    updated_mech = mech
    updated_pilot = pilot

    wreckage = None
    if result.mech_destroyed:
        size_value = _get_size_value(mech.stats.size)
        wreckage = WreckageState.from_meltdown(
            combatant_id=mech.id,
            combatant_name=mech.name,
            position=mech.position,
            size_value=size_value,
            object_id=f"wreckage_{mech.id}",
        )

        statuses_after = [
            s for s in mech.statuses if s not in ["impaired", "exposed", "stunned"]
        ]
        statuses_after.append("out")  # type: ignore[arg-type]

        updated_mech = mech.model_copy(
            update={
                "statuses": statuses_after,
                "resources": mech.resources.model_copy(
                    update={
                        "hp_current": 0,
                        "structure_current": 0,
                        "stress_current": 0,
                    }
                ),
                "meltdown_state": None,
            }
        )

    if result.pilot_killed and updated_pilot is not None:
        pilot_statuses = list(updated_pilot.statuses)
        if "out" not in pilot_statuses:
            pilot_statuses.append("out")  # type: ignore[arg-type]

        updated_pilot = updated_pilot.model_copy(
            update={
                "statuses": pilot_statuses,
                "resources": updated_pilot.resources.model_copy(
                    update={"hp_current": 0}
                ),
            }
        )

    updated_targets: list[TargetExplosionResult] = []
    for target_result in result.target_results:
        updated_damage = target_result
        if target_result.damage_dealt > 0:
            updated_damage = target_result.model_copy(
                update={"damage_dealt": target_result.damage_dealt}
            )
        updated_targets.append(updated_damage)

    return SelfDestructExplosionApplicationResult(
        updated_mech=updated_mech,
        updated_pilot=updated_pilot,
        wreckage_created=result.wreckage is not None,
        affected_targets=updated_targets,
    )


def _get_size_value(size: SizeClass | int) -> int:
    """Extract integer size value from SizeClass or return as-is."""
    if isinstance(size, int):
        return size
    size_mapping = {
        "size_half": 1,
        "size_1": 1,
        "size_2": 2,
        "size_3": 3,
        "size_4": 4,
        "size_5": 5,
    }
    return size_mapping.get(str(size), 1)


try:
    from core.shared.heat import MeltdownState
    from core.mech.combat_state import CombatantState

    CombatantState.model_rebuild(
        _types_namespace={"MeltdownState": MeltdownState, "StabilizeState": type(None)}
    )
except ImportError:
    pass
