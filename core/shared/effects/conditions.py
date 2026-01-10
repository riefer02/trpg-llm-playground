"""Condition primitives for effect targeting.

Conditions define when and how effects apply. They can be combined
using ConditionGroup for complex AND/OR/NOT logic.

Effects:
    - SpatialCondition: Spatial relationship predicates (adjacent, range, zone)
    - AttackContextCondition: Attack context predicates (types, ranges, targeting)
    - SizeCondition: Size comparison predicates for source/target/self
    - CheckContextCondition: Non-attack check/save predicates
    - ReactionCondition: Reaction-specific predicates (brace, overwatch)
    - ConditionGroup: Combine conditions with AND/OR/NOT semantics

See Also:
    - PR2 3985-4012: Conditions reference
    - PR2 3965-3969: Attack roll context
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import Field, model_validator

from core.shared.models import FrozenModel
from core.shared.enums import (
    AttackType,
    CoverType,
    RangeType,
    SaveType,
    SizeClass,
)
from core.shared.id_helpers import ReactionIdField
from core.shared.effects.types import (
    AttackAreaShape,
    ConditionType,
    EffectTarget,
    EffectTargetNoAll,
    SpatialRelation,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "SpatialCondition",
    "AttackContextCondition",
    "SizeCondition",
    "CheckContextCondition",
    "ReactionCondition",
    "ConditionGroup",
    "EffectCondition",
]


class SpatialCondition(FrozenModel):
    """
    Spatial relationship predicates (adjacent, range, zone entry, cover sourcing).

    Examples:
        SpatialCondition(relation="adjacent", target="ally")
        SpatialCondition(relation="outside_range", range=5, range_type="range")
    """

    relation: SpatialRelation
    target: EffectTarget = "enemy"
    source: EffectTargetNoAll = "self"
    range: int | None = Field(default=None, ge=0)
    range_type: RangeType | None = None
    cover: CoverType | None = None
    requires_line_of_sight: bool | None = None
    requires_free_space: bool | None = None


class AttackContextCondition(FrozenModel):
    """
    Attack context predicates (attack types, ranges, targeting scope).

    Examples:
        AttackContextCondition(attack_types=["ranged"], applies_to="incoming")
        AttackContextCondition(range_comparator="gt", range=5, applies_to="outgoing")
    """

    attack_types: list[AttackType] = Field(default_factory=list)
    area_shapes: list[AttackAreaShape] = Field(default_factory=list)
    weapon_ids: list[str] = Field(default_factory=list)
    applies_to: Literal["incoming", "outgoing", "mutual"] = "incoming"
    target: EffectTarget = "enemy"
    range_comparator: Literal["gt", "lt", "within"] | None = None
    range: int | None = Field(default=None, ge=0)
    range_type: RangeType | None = None
    requires_line_of_sight: bool | None = None
    requires_line_of_effect_crossing_zone: bool | None = None


class SizeCondition(FrozenModel):
    """
    Size comparison predicates for source/target/self.

    Examples:
        SizeCondition(subject="source", comparator="lt", size="size_5")
    """

    subject: Literal["source", "target", "self"] = "source"
    comparator: Literal["lt", "lte", "gt", "gte", "eq"] = "lt"
    size: SizeClass


class CheckContextCondition(FrozenModel):
    """
    Non-attack check/save predicates and equipment gating.

    Examples:
        CheckContextCondition(check_kinds=["save"], saves=["hull"])
        CheckContextCondition(equipment_materials=["metal"])
    """

    check_kinds: list[Literal["check", "save", "contested", "search"]] = Field(
        default_factory=list
    )
    saves: list[SaveType] = Field(default_factory=list)
    target: EffectTarget = "self"
    requires_line_of_sight: bool | None = None
    equipment_materials: list[str] = Field(default_factory=list)
    equipment_tags: list[str] = Field(default_factory=list)


class ReactionCondition(FrozenModel):
    """
    Reaction-specific predicates (brace, overwatch).

    Examples:
        ReactionCondition(reaction_id="brace")
        ReactionCondition(reaction_id="overwatch", is_attack=True)
    """

    reaction_id: ReactionIdField | None = None
    is_attack: bool | None = None

    @model_validator(mode="after")
    def _validate_reaction_condition(self) -> "ReactionCondition":
        """Ensure at least one condition parameter is set.

        Raises:
            ValueError: If both reaction_id and is_attack are None.
        """
        if self.reaction_id is None and self.is_attack is None:
            raise ValueError(
                "ReactionCondition must define 'reaction_id' or 'is_attack'"
            )
        return self


class ConditionGroup(FrozenModel):
    """
    Combine multiple conditions with AND/OR/NOT semantics.

    Examples:
        ConditionGroup(all_of=[SpatialCondition(relation="adjacent", target="ally")])
        ConditionGroup(
            all_of=[SpatialCondition(relation="using_cover_from_source", cover="hard")],
            any_of=[AttackContextCondition(area_shapes=["blast", "line", "cone"])],
        )
    """

    all_of: list["EffectCondition"] = Field(default_factory=list)
    any_of: list["EffectCondition"] = Field(default_factory=list)
    none_of: list["EffectCondition"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_group(self) -> "ConditionGroup":
        """Ensure at least one condition list is defined.

        Raises:
            ValueError: If all_of, any_of, and none_of are all empty.
        """
        if not self.all_of and not self.any_of and not self.none_of:
            raise ValueError("ConditionGroup must define at least one condition list")
        return self


# Type alias for any condition type - must be defined after all condition classes
EffectCondition = (
    ConditionType
    | str
    | SpatialCondition
    | AttackContextCondition
    | CheckContextCondition
    | ReactionCondition
    | SizeCondition
    | ConditionGroup
)
