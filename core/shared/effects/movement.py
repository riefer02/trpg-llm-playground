"""Movement and positioning effects.

Effects for granting movement, teleportation, forced movement, and movement restrictions.

Effects:
    - MovementGrant: Grants movement or teleportation actions
    - MoveAdjacentEffect: Moves source adjacent to a target
    - PositionSwapEffect: Swaps positions between two targets
    - ForcedMovement: Push or pull effects on targets
    - MovementRestrictionEffect: Limits movement types or distance
    - MovementSurfaceEffect: Changes terrain movement rules
    - MovementModeAccessEffect: Grants flight, burrow, or other movement modes
    - JumpDistanceEffect: Modifies jump distances
    - MovementOverrideEffect: Overrides movement mechanics

See Also:
    - PR2 3729-3930: Movement rules
    - PR2 4132-4151: Movement and forced movement effects
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.enums import ActionType
from core.shared.effects.types import (
    EffectDuration,
    EffectTarget,
    EffectTargetNoAll,
    ForcedMovementDistanceType,
    MovementDistanceType,
    MovementMode,
    TechRangeType,
    TriggerType,
    UsesPer,
)
from core.shared.effects.conditions import EffectCondition

if TYPE_CHECKING:
    from core.shared.effects.core import MechanicalEffect

__all__ = [
    "MovementGrant",
    "MoveAdjacentEffect",
    "PositionSwapEffect",
    "ForcedMovement",
    "MovementRestrictionEffect",
    "MovementSurfaceEffect",
    "MovementModeAccessEffect",
    "JumpDistanceEffect",
    "MovementOverrideEffect",
]


class MovementGrant(FrozenModel):
    """Grants movement or teleportation.

    Per PR2 4132-4145: Movement includes walking, flying, and teleporting.
    Movement grants can be triggered by actions, reactions, or conditions.

    Examples:
        MovementGrant(spaces=2, movement_type="fly", trigger="on_successful_save")
        MovementGrant(spaces=3, movement_type="teleport", trigger="after_boost")
        MovementGrant(spaces="speed", movement_type="walk", trigger="brace")
    """

    spaces: MovementDistanceType
    movement_type: Literal["walk", "fly", "teleport", "boost"] = "walk"
    trigger: TriggerType | str | None = Field(default=None)
    target: EffectTargetNoAll = "self"
    distance_is_maximum: bool = False
    requires_line_of_sight: bool = True
    requires_free_space: bool = False
    fails_if_occupied: bool = False
    ignores_engagement: bool = False
    provokes_reactions: bool = True
    condition: EffectCondition | None = None


class MoveAdjacentEffect(FrozenModel):
    """
    Moves the source adjacent to a target under specific conditions.

    Examples:
        MoveAdjacentEffect(target="enemy", movement_type="fly", trigger="on_turn_end")
    """

    target: Literal["enemy", "ally", "any"] = "enemy"
    movement_type: Literal["walk", "fly", "teleport"] = "walk"
    trigger: TriggerType | None = None
    action_type: ActionType | None = None
    requires_line_of_sight: bool = True
    requires_free_space: bool = True
    fails_if_occupied: bool = True
    ignores_engagement: bool = False
    provokes_reactions: bool = True
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class PositionSwapEffect(FrozenModel):
    """
    Swaps positions between the source and a target.

    Examples:
        PositionSwapEffect(action_type="quick", range=10, range_type="sensors", uses_per="scene")
    """

    action_type: ActionType
    target: EffectTargetNoAll = "ally"
    range: int | None = Field(default=None, ge=0)
    range_type: TechRangeType = "sensors"
    requires_line_of_sight: bool = True
    uses_per: UsesPer = "scene"
    condition: EffectCondition | None = None


class ForcedMovement(FrozenModel):
    """Forced push/pull movement applied to a target.

    Per PR2 4146-4151: Forced movement moves targets against their will.
    Push moves away from source, pull moves toward source or zone.

    Examples:
        ForcedMovement(direction="pull", distance=5, ignores_engagement=True)
    """

    direction: Literal["pull", "push"]
    distance: ForcedMovementDistanceType
    target: EffectTarget = "enemy"
    toward: Literal["source", "zone_center"] = "source"
    ignores_engagement: bool = False
    provokes_reactions: bool = True
    must_obey_obstructions: bool = True
    on_collision: "MechanicalEffect | None" = None
    condition: EffectCondition | None = None


class MovementRestrictionEffect(FrozenModel):
    """
    Movement restriction applied to a target.

    Examples:
        MovementRestrictionEffect(target="enemy", cannot_move_closer_to_source=True)
    """

    target: EffectTarget = "enemy"
    movement_modes: list[MovementMode] = Field(default_factory=list)
    max_voluntary_speed: int | None = Field(default=None, ge=0)
    cannot_move_closer_to_source: bool = False
    cannot_move_further_from_source: bool = False
    must_move_straight_line: bool = False
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class MovementSurfaceEffect(FrozenModel):
    """
    Modifies terrain/surface rules for movement.

    Examples:
        MovementSurfaceEffect(ignore_difficult_terrain=True)
        MovementSurfaceEffect(treat_vertical_as_ground=True, fall_if_prone=True)
    """

    target: EffectTarget = "self"
    ignore_difficult_terrain: bool = False
    treat_vertical_as_ground: bool = False
    fall_if_prone: bool = False
    duration: EffectDuration = "scene"
    condition: EffectCondition | None = None


class MovementModeAccessEffect(FrozenModel):
    """
    Grants access to movement modes or improves traversal.

    Examples:
        MovementModeAccessEffect(climb_at_full_speed=True, swim_at_full_speed=True)
    """

    target: EffectTargetNoAll = "self"
    climb_at_full_speed: bool = False
    swim_at_full_speed: bool = False
    condition: EffectCondition | None = None


class JumpDistanceEffect(FrozenModel):
    """
    Modifies jump distances relative to speed.

    Examples:
        JumpDistanceEffect(horizontal_multiplier=1.0, vertical_multiplier=0.5)
    """

    target: EffectTargetNoAll = "self"
    horizontal_multiplier: float | None = Field(default=None, ge=0)
    vertical_multiplier: float | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class MovementOverrideEffect(FrozenModel):
    """
    Replaces movement modes with another type.

    Examples:
        MovementOverrideEffect(movement_modes=["move", "boost"], override_type="teleport")
    """

    target: EffectTargetNoAll = "self"
    movement_modes: list[MovementMode] = Field(default_factory=list)
    override_type: Literal["walk", "fly", "teleport"] = "teleport"
    same_distance: bool = True
    must_end_on_surface: bool = False
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None
