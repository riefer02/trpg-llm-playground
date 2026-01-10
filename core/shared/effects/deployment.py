"""Deployment and attachment effects.

Effects for deploying devices, attachments, and system links.

Effects:
    - DeploymentEffect: Deploying devices that prime and activate
    - AttachmentEffect: Attaching devices to targets
    - SystemLinkEffect: Persistent links between characters

See Also:
    - PR2 5070-5088: Deployables and drones
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.enums import ActionType, CoverType, SaveType, StatusType
from core.shared.effects.types import (
    CheckKind,
    DeploymentActivationCondition,
    EffectDuration,
    EffectTargetNoAll,
    StatType,
    TechRangeType,
)
from core.shared.effects.conditions import EffectCondition

if TYPE_CHECKING:
    from core.shared.effects.core import MechanicalEffect

__all__ = [
    "DeploymentEffect",
    "AttachmentEffect",
    "SystemLinkEffect",
]


class DeploymentEffect(FrozenModel):
    """
    Represents deploying a device that primes and can be activated later.

    Examples:
        DeploymentEffect(
            action_type="quick",
            placement_range=1,
            placement_relation="adjacent",
            primes_after="turn_end",
            activation_condition="adjacent_start_or_move",
        )
    """

    action_type: ActionType
    placement_range: int = Field(..., ge=0)
    placement_relation: Literal["adjacent", "range"] = "adjacent"
    placement_requires_free_space: bool = False
    primes_after: Literal["turn_end", "immediate"] = "immediate"
    activation_condition: DeploymentActivationCondition
    activation_action: ActionType | None = None
    activation_target: EffectTargetNoAll = "self"
    activation_effect: "MechanicalEffect | None" = None
    consumes_on_activation: bool = True
    becomes_object_on_activation: bool = False
    open_topped: bool = False
    immobile: bool = False
    can_deactivate: bool = True


class AttachmentEffect(FrozenModel):
    """
    Attaches a device or effect to a target.

    Examples:
        AttachmentEffect(action_type="quick", range=5, target="ally")
    """

    action_type: ActionType
    range: int = Field(..., ge=0)
    target: EffectTargetNoAll = "ally"
    requires_line_of_sight: bool = True
    max_instances_per_target: int | None = Field(default=None, ge=1)
    duration: EffectDuration | None = None
    condition: EffectCondition | None = None


class SystemLinkEffect(FrozenModel):
    """
    Persistent link between two characters for shared systems/conditions.

    Examples:
        SystemLinkEffect(
            action_type="quick",
            range=10,
            range_type="sensors",
            target="ally",
            duration="scene",
        )
    """

    action_type: ActionType
    range: int | None = Field(default=None, ge=0)
    range_type: TechRangeType = "sensors"
    target: EffectTargetNoAll = "ally"
    requires_line_of_sight: bool = True
    duration: EffectDuration | None = None
    max_links_per_source: int | None = Field(default=None, ge=1)
    breaks_on_out_of_range: bool = False
    break_if_statuses: list[StatusType] = Field(default_factory=list)
    shares_space: bool = False
    moves_with_target: bool = False
    cover_from_target: CoverType | None = None
    share_conditions: bool = False
    share_heat_from_tech: bool = False
    share_heat_all: bool = False
    tech_action_uses_target_sensors: bool = False
    tech_action_uses_target_los: bool = False
    stat_proxy_to_target: list[StatType] = Field(default_factory=list)
    stat_proxy_to_source: list[StatType] = Field(default_factory=list)
    check_stat_proxy_to_target: list[SaveType] = Field(default_factory=list)
    check_stat_proxy_to_source: list[SaveType] = Field(default_factory=list)
    check_kinds: list[CheckKind] = Field(default_factory=list)
    condition: EffectCondition | None = None
