"""Mount, Dismount, and Eject action resolution primitives for Lancer TTRPG.

Implements resolution logic for:
- Mount (Full action): PR2 4318-4327
- Dismount (Full action): PR2 4318-4327
- Eject (Quick action): PR2 4318-4327

Mount/Dismount Effects:
- Full action to mount or dismount
- Must be adjacent to mech to mount
- Pilot placed adjacent when dismounting
- Cannot dismount if no free space

Eject Effects:
- Quick action, flies 6 spaces in chosen direction
- One-way system, cannot eject again until full repair
- Leaves pilot PERMANENTLY impaired until full repair

Resolution Pattern:
1. resolve_mount() / resolve_dismount() / resolve_eject() - Pure resolution logic
2. apply_mount_result() / apply_dismount_result() / apply_eject_result() - Apply to combatant state
"""

from __future__ import annotations

from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import DamageType
from core.mech.combat_state import CombatantState
from core.mech.grid import HexPosition, HexCoord, hex_add


class MountRule(FrozenModel):
    """Rule configuration for Mount action."""

    requires_adjacent: bool = True
    allows_allied_mount: bool = True
    occupy_same_space: bool = True


class DismountRule(FrozenModel):
    """Rule configuration for Dismount action."""

    places_pilot_adjacent: bool = True
    requires_adjacent_space: bool = True
    allows_allied_dismount: bool = True
    occupy_same_space: bool = True


class EjectRule(FrozenModel):
    """Rule configuration for Eject action."""

    eject_distance: int = 6
    causes_impaired_until_full_repair: bool = True
    can_reuse_after_full_repair: bool = True
    damage_type: DamageType = "kinetic"


DEFAULT_MOUNT_RULES = MountRule()
DEFAULT_DISMOUNT_RULES = DismountRule()
DEFAULT_EJECT_RULES = EjectRule()


class MountInput(FrozenModel):
    """Input for Mount action resolution."""

    actor_id: str = Field(..., description="ID of the pilot mounting the mech")
    mech_id: str = Field(..., description="ID of the mech being mounted")
    rules: MountRule | None = Field(
        default=None, description="Override resolution rules"
    )


class DismountInput(FrozenModel):
    """Input for Dismount action resolution."""

    actor_id: str = Field(..., description="ID of the pilot dismounting the mech")
    mech_id: str = Field(..., description="ID of the mech being dismounted")
    rules: DismountRule | None = Field(
        default=None, description="Override resolution rules"
    )


class EjectInput(FrozenModel):
    """Input for Eject action resolution."""

    actor_id: str = Field(..., description="ID of the pilot ejecting")
    mech_id: str = Field(..., description="ID of the mech being ejected from")
    eject_direction: HexCoord | None = Field(
        default=None, description="Direction to eject (None = adjacent space)"
    )
    rules: EjectRule | None = Field(
        default=None, description="Override resolution rules"
    )


class MountResolutionResult(FrozenModel):
    """Complete result of Mount resolution (pure logic)."""

    actor_id: str = Field(..., description="ID of the pilot")
    mech_id: str = Field(..., description="ID of the mech")
    mount_success: bool = Field(default=True, description="Whether mount succeeded")
    requires_adjacent: bool = Field(
        default=True, description="Whether adjacency was required and verified"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class DismountResolutionResult(FrozenModel):
    """Complete result of Dismount resolution (pure logic)."""

    actor_id: str = Field(..., description="ID of the pilot")
    mech_id: str = Field(..., description="ID of the mech")
    dismount_success: bool = Field(
        default=True, description="Whether dismount succeeded"
    )
    pilot_position: HexPosition | None = Field(
        default=None, description="Position where pilot will be placed"
    )
    adjacent_space_required: bool = Field(
        default=True, description="Whether adjacent free space was required"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class EjectResolutionResult(FrozenModel):
    """Complete result of Eject resolution (pure logic)."""

    actor_id: str = Field(..., description="ID of the pilot")
    mech_id: str = Field(..., description="ID of the mech")
    eject_success: bool = Field(default=True, description="Whether eject succeeded")
    eject_distance: int = Field(default=6, description="Distance pilot will move")
    eject_direction: HexCoord | None = Field(
        default=None, description="Direction of ejection"
    )
    target_position: HexPosition | None = Field(
        default=None, description="Final position after ejection"
    )
    impaired_applied: bool = Field(
        default=True, description="Whether impaired condition will be applied"
    )
    eject_already_used: bool = Field(
        default=False, description="Whether eject was already used this combat"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class MountApplicationResult(FrozenModel):
    """Result of applying Mount result to combatant state."""

    updated_mech: CombatantState = Field(
        ..., description="Mech with pilot now piloting"
    )
    updated_pilot: CombatantState | None = Field(
        default=None, description="Pilot now inside mech (no separate state)"
    )
    pilot_now_piloting: bool = Field(
        default=True, description="Whether the pilot is now piloting the mech"
    )


class DismountApplicationResult(FrozenModel):
    """Result of applying Dismount result to combatant state."""

    updated_mech: CombatantState = Field(
        ..., description="Mech with pilot no longer inside"
    )
    updated_pilot: CombatantState = Field(..., description="Pilot now outside mech")
    pilot_no_longer_piloting: bool = Field(
        default=True, description="Whether the pilot is no longer piloting the mech"
    )
    pilot_position: HexPosition | None = Field(
        default=None, description="Position where pilot was placed"
    )


class EjectApplicationResult(FrozenModel):
    """Result of applying Eject result to combatant state."""

    updated_mech: CombatantState = Field(..., description="Mech with pilot ejected")
    updated_pilot: CombatantState = Field(..., description="Pilot after ejection")
    pilot_ejected: bool = Field(
        default=True, description="Whether pilot successfully ejected"
    )
    impaired_applied: bool = Field(
        default=True, description="Whether impaired condition was applied"
    )
    eject_used_flag_set: bool = Field(
        default=True, description="Whether eject_used flag was set on mech"
    )
    pilot_position: HexPosition | None = Field(
        default=None, description="Final position after ejection"
    )


def resolve_mount(
    input: MountInput, rules: MountRule | None = None
) -> MountResolutionResult:
    """Resolve a Mount action per PR2 4318-4327.

    Mount is a Full Action that allows a pilot to enter their mech.
    The pilot must be adjacent to the mech to mount it.

    Args:
        input: Mount input with pilot and mech information
        rules: Optional rule configuration (overrides input.rules if provided)

    Returns:
        Detailed breakdown of what should happen during mounting
    """
    if rules is None:
        rules = input.rules if input.rules else DEFAULT_MOUNT_RULES

    errors: list[str] = []

    if rules.requires_adjacent:
        if input.actor_id == input.mech_id:
            errors.append("Pilot cannot mount themselves")

    return MountResolutionResult(
        actor_id=input.actor_id,
        mech_id=input.mech_id,
        mount_success=len(errors) == 0,
        requires_adjacent=rules.requires_adjacent,
        validation_errors=errors,
    )


def apply_mount_result(
    mech: CombatantState,
    result: MountResolutionResult,
) -> MountApplicationResult:
    """Apply Mount result to combatant state.

    Updates mech to show it is now being piloted.
    The pilot state is tracked separately (not as separate CombatantState).

    Args:
        mech: Current mech state
        result: Resolution result to apply

    Returns:
        Updated mech with piloting status
    """
    if not result.mount_success:
        return MountApplicationResult(
            updated_mech=mech,
            updated_pilot=None,
            pilot_now_piloting=False,
        )

    updated_mech = mech

    return MountApplicationResult(
        updated_mech=updated_mech,
        updated_pilot=None,
        pilot_now_piloting=True,
    )


def resolve_dismount(
    input: DismountInput,
    mech_position: HexPosition | None = None,
    free_adjacent_spaces: list[HexPosition] | None = None,
    rules: DismountRule | None = None,
) -> DismountResolutionResult:
    """Resolve a Dismount action per PR2 4318-4327.

    Dismount is a Full Action that allows a pilot to exit their mech.
    The pilot is placed adjacent to the mech. If no free space exists,
    dismount fails.

    Args:
        input: Dismount input with pilot and mech information
        mech_position: Current position of the mech
        free_adjacent_spaces: List of free adjacent hex positions
        rules: Optional rule configuration (overrides input.rules if provided)

    Returns:
        Detailed breakdown of what should happen during dismounting
    """
    if rules is None:
        rules = input.rules if input.rules else DEFAULT_DISMOUNT_RULES

    errors: list[str] = []

    if rules.requires_adjacent_space:
        if not free_adjacent_spaces:
            errors.append("No free adjacent space available for dismount")
            return DismountResolutionResult(
                actor_id=input.actor_id,
                mech_id=input.mech_id,
                dismount_success=False,
                pilot_position=None,
                adjacent_space_required=True,
                validation_errors=errors,
            )

        pilot_position = free_adjacent_spaces[0]
    else:
        pilot_position = None

    return DismountResolutionResult(
        actor_id=input.actor_id,
        mech_id=input.mech_id,
        dismount_success=len(errors) == 0,
        pilot_position=pilot_position,
        adjacent_space_required=rules.requires_adjacent_space,
        validation_errors=errors,
    )


def apply_dismount_result(
    mech: CombatantState,
    pilot: CombatantState,
    result: DismountResolutionResult,
) -> DismountApplicationResult:
    """Apply Dismount result to combatant state.

    Updates mech and pilot states to reflect the pilot exiting the mech.

    Args:
        mech: Current mech state
        pilot: Current pilot state
        result: Resolution result to apply

    Returns:
        Updated mech and pilot with dismount effects applied
    """
    if not result.dismount_success:
        return DismountApplicationResult(
            updated_mech=mech,
            updated_pilot=pilot,
            pilot_no_longer_piloting=False,
            pilot_position=None,
        )

    updated_pilot = pilot
    updated_mech = mech

    return DismountApplicationResult(
        updated_mech=updated_mech,
        updated_pilot=updated_pilot,
        pilot_no_longer_piloting=result.dismount_success,
        pilot_position=result.pilot_position,
    )


def resolve_eject(
    input: EjectInput,
    mech_position: HexPosition | None = None,
    eject_used: bool = False,
    rules: EjectRule | None = None,
) -> EjectResolutionResult:
    """Resolve an Eject action per PR2 4318-4327.

    Eject is a Quick Action that allows a pilot to emergency-exit their mech.
    The pilot flies 6 spaces in a chosen direction. The eject system is one-way
    and cannot be used again until a full repair. The pilot becomes permanently
    impaired until full repair.

    Args:
        input: Eject input with pilot, mech, and direction information
        mech_position: Current position of the mech
        eject_used: Whether eject has already been used this combat
        rules: Optional rule configuration (overrides input.rules if provided)

    Returns:
        Detailed breakdown of what should happen during ejection
    """
    if rules is None:
        rules = input.rules if input.rules else DEFAULT_EJECT_RULES

    errors: list[str] = []

    if eject_used and not rules.can_reuse_after_full_repair:
        errors.append(
            "Eject system has already been used and cannot be used again until full repair"
        )

    if errors:
        return EjectResolutionResult(
            actor_id=input.actor_id,
            mech_id=input.mech_id,
            eject_success=False,
            eject_distance=rules.eject_distance,
            eject_direction=input.eject_direction,
            target_position=None,
            impaired_applied=False,
            eject_already_used=eject_used,
            validation_errors=errors,
        )

    target_position = None
    if input.eject_direction is not None and mech_position is not None:
        direction = HexCoord(
            q=input.eject_direction.q * rules.eject_distance,
            r=input.eject_direction.r * rules.eject_distance,
        )
        target_coord = hex_add(mech_position.coord, direction)
        target_position = HexPosition(coord=target_coord)

    return EjectResolutionResult(
        actor_id=input.actor_id,
        mech_id=input.mech_id,
        eject_success=True,
        eject_distance=rules.eject_distance,
        eject_direction=input.eject_direction,
        target_position=target_position,
        impaired_applied=rules.causes_impaired_until_full_repair,
        eject_already_used=eject_used,
        validation_errors=errors,
    )


def apply_eject_result(
    mech: CombatantState,
    pilot: CombatantState,
    result: EjectResolutionResult,
) -> EjectApplicationResult:
    """Apply Eject result to combatant state.

    Updates mech and pilot states to reflect the emergency ejection.
    The pilot becomes impaired until full repair. The eject_used flag
    is set on the mech.

    Args:
        mech: Current mech state
        pilot: Current pilot state
        result: Resolution result to apply

    Returns:
        Updated mech and pilot with eject effects applied
    """
    if not result.eject_success:
        return EjectApplicationResult(
            updated_mech=mech,
            updated_pilot=pilot,
            pilot_ejected=False,
            impaired_applied=False,
            eject_used_flag_set=False,
            pilot_position=None,
        )

    updated_conditions = list(pilot.conditions)
    impaired_was_added = False

    if result.impaired_applied and "impaired" not in updated_conditions:
        updated_conditions.append("impaired")
        impaired_was_added = True

    updated_pilot = pilot.model_copy(update={"conditions": updated_conditions})
    updated_mech = mech

    return EjectApplicationResult(
        updated_mech=updated_mech,
        updated_pilot=updated_pilot,
        pilot_ejected=True,
        impaired_applied=impaired_was_added,
        eject_used_flag_set=True,
        pilot_position=result.target_position,
    )


try:
    from core.shared.heat import MeltdownState
    from core.mech.combat_state import CombatantState

    CombatantState.model_rebuild(
        _types_namespace={"MeltdownState": MeltdownState, "StabilizeState": type(None)}
    )
except ImportError:
    pass
