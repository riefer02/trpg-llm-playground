"""Stat and mount slot modification effects.

Effects for modifying mech stats, mount slots, and integrated weapons.

Effects:
    - StatModifier: Numeric modifier to a stat
    - CompanionStatModifierEffect: Stat modifiers for drones/deployables
    - StatOverrideEffect: Sets a stat to a specific value
    - MountSlotGrant: Adds one or more mount slots
    - MountSlotReplacement: Replaces existing mount slots
    - MountSizeUpgradeEffect: Increases weapon size a mount can accept
    - IntegratedWeaponEffect: Grants integrated weapon mount

See Also:
    - PR2 4411-4450: Mech stats reference
    - PR2 4500-4550: Mount system rules
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.enums import ActionType, MountSlotType
from core.shared.effects.types import (
    EffectDuration,
    EffectTargetNoAll,
    StatType,
    TriggerType,
    UsesPer,
    WeaponSizeType,
)
from core.shared.effects.conditions import EffectCondition

__all__ = [
    "StatModifier",
    "CompanionStatModifierEffect",
    "StatOverrideEffect",
    "MountSlotGrant",
    "MountSlotReplacement",
    "MountSizeUpgradeEffect",
    "IntegratedWeaponEffect",
]


class StatModifier(FrozenModel):
    """
    Numeric modifier to a stat.

    Examples:
        StatModifier(stat="hp", value=5)  # +5 HP
        StatModifier(stat="size", value=1)  # +1 Size
    """

    stat: StatType
    value: int


class CompanionStatModifierEffect(FrozenModel):
    """
    Stat modifiers that apply to deployables, drones, or other companion entities.

    Examples:
        CompanionStatModifierEffect(stat="hp", value=5, applies_to=["drone", "deployable"])
    """

    stat: StatType
    value: int
    applies_to: list[Literal["drone", "deployable", "object"]] = Field(
        default_factory=list
    )
    duration: EffectDuration | None = None
    condition: EffectCondition | None = None


class StatOverrideEffect(FrozenModel):
    """
    Sets a stat to a specific value while a condition is met.

    Examples:
        StatOverrideEffect(stat="evasion", value=5, condition="reserve_power_mode")
    """

    stat: StatType
    value: int
    target: EffectTargetNoAll = "self"
    duration: EffectDuration | None = None
    condition: EffectCondition | None = None


class MountSlotGrant(FrozenModel):
    """
    Adds one or more mount slots.

    Examples:
        MountSlotGrant(slot_type="flexible", count=1, requires_mount_count_lt=3)
    """

    slot_type: MountSlotType
    count: int = Field(default=1, ge=1)
    requires_mount_count_lt: int | None = Field(default=None, ge=0)
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class MountSlotReplacement(FrozenModel):
    """
    Replaces existing mount slots with a new slot type.

    Examples:
        MountSlotReplacement(new_slot_type="main_aux", count=1)
    """

    new_slot_type: MountSlotType
    count: int = Field(default=1, ge=1)
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class MountSizeUpgradeEffect(FrozenModel):
    """
    Increases the weapon size that a mount can accept.

    Examples:
        MountSizeUpgradeEffect(increase_by=1, count=1)
    """

    increase_by: int = Field(default=1, ge=1)
    count: int = Field(default=1, ge=1)
    max_size: WeaponSizeType | None = None
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class IntegratedWeaponEffect(FrozenModel):
    """
    Grants an integrated weapon mount with linked free-fire rules.

    Examples:
        IntegratedWeaponEffect(weapon_size="aux", free_attack_uses_per="round")
    """

    weapon_size: WeaponSizeType = "aux"
    free_attack_action_type: ActionType = "free"
    free_attack_uses_per: UsesPer = "round"
    free_attack_trigger: TriggerType = "on_attack_roll"
    requires_other_weapon_attack: bool = True
    cannot_be_modified: bool = True
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None
