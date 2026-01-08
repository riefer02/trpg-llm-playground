"""Mech weapon types for Lancer TTRPG."""

from typing import Literal
from pydantic import Field, model_validator
from core.shared.models import FrozenModel

from core.shared.enums import ActionType, DamageType
from core.shared.dice import DiceExpression
from core.shared.enums import RangeType
from core.shared.effects import (
    EffectTargetWithObjectNoAll,
    MechanicalEffect,
    UsesPer,
)
from core.shared.id_helpers import WeaponIdField, LicenseIdField, FrameIdField


WeaponSize = Literal["aux", "main", "heavy", "superheavy"]
WeaponType = Literal["cqb", "rifle", "launcher", "cannon", "melee", "nexus"]
WeaponDamageType = DamageType | Literal["heat"]
WeaponTagType = Literal[
    "accurate",
    "inaccurate",
    "loading",
    "ordnance",
    "arcing",
    "seeking",
    "smart",
    "ap",
    "overkill",
    "reliable",
    "heat_self",
    "heat_target",
    "burn",
    "knockback",
    "limited",
    "threat",
    "thrown",
    "line",
    "cone",
    "blast",
    "burst",
]


class WeaponRange(FrozenModel):
    """Range profile for a weapon (including threat, blast, etc)."""

    range_type: RangeType
    value: int = Field(..., ge=0)


class WeaponDamage(FrozenModel):
    """Damage component for a weapon."""

    damage_type: WeaponDamageType
    dice: DiceExpression | None = None
    flat: int = 0


class WeaponTag(FrozenModel):
    """A weapon tag with an optional numeric value."""

    tag: WeaponTagType
    value: int | None = None


class WeaponProfile(FrozenModel):
    """Alternate profile for a weapon (e.g., selectable damage type)."""

    profile_id: WeaponIdField = Field(..., description="Unique profile identifier")
    name: str = Field(..., description="Display name")
    damage_type: WeaponDamageType | None = Field(
        default=None,
        description="Primary damage type for the profile (if applicable)",
    )
    ranges: list[WeaponRange] = Field(default_factory=list)
    damage: list[WeaponDamage] = Field(default_factory=list)
    tags: list[WeaponTag] = Field(default_factory=list)
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)


class WeaponProfileChoice(FrozenModel):
    """Profile selection rules for weapons with multiple profiles."""

    profiles: list[WeaponProfile] = Field(default_factory=list)
    selection: Literal["attack", "turn_start", "scene_start"] = "attack"
    default_profile_id: str | None = None


class MimicGunProfileRule(FrozenModel):
    """Dynamic profile rule for the Mimic Gun."""

    roll_expression: DiceExpression = Field(
        default_factory=lambda: DiceExpression.parse("1d20"),
        description="Roll used for each mimic value",
    )
    roll_count: int = Field(default=3, ge=1)
    cycle_on: Literal["turn_start"] = "turn_start"
    range_from_roll: bool = True
    damage_divisor: int = Field(default=2, ge=1)
    damage_rounding: Literal["up", "down"] = "up"
    damage_bonus: int = Field(default=1, ge=0)
    reroll_action_type: Literal["full"] = "full"


class DynamicWeaponDefinition(FrozenModel):
    """Dynamic weapon behavior (profile choices or roll-based cycling)."""

    profile_choice: WeaponProfileChoice | None = None
    mimic_gun: MimicGunProfileRule | None = None

    @model_validator(mode="after")
    def _validate_dynamic_mode(self) -> "DynamicWeaponDefinition":
        mode_count = sum(
            1 for mode in (self.profile_choice, self.mimic_gun) if mode is not None
        )
        if mode_count != 1:
            raise ValueError(
                "DynamicWeaponDefinition must set exactly one dynamic mode."
            )
        return self


class MountlessWeaponDefinition(FrozenModel):
    """Weapon-like profile that does not count as a weapon or occupy a mount."""

    id: WeaponIdField = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    profile: WeaponProfile
    action_type: ActionType = "free"
    target: EffectTargetWithObjectNoAll = "enemy"
    uses_per: UsesPer = "round"
    requires_line_of_sight: bool = True
    counts_as_attack: bool = False
    auto_hit: bool = True
    ignores_cover: bool = True
    damage_unreducible: bool = True
    counts_as_weapon: bool = False
    modifiable: bool = False
    benefits_from_talents: bool = False

    @model_validator(mode="after")
    def _validate_profile_identity(self) -> "MountlessWeaponDefinition":
        if self.profile.profile_id != self.id:
            raise ValueError(
                "Mountless weapon profile_id must match its definition id."
            )
        if self.profile.name != self.name:
            raise ValueError(
                "Mountless weapon profile name must match its definition name."
            )
        return self


class MechWeaponDefinition(FrozenModel):
    """Definition for a mech weapon."""

    id: WeaponIdField = Field(..., description="Unique weapon identifier")
    name: str = Field(..., description="Display name")
    size: WeaponSize
    weapon_type: WeaponType
    damage_type: WeaponDamageType | None = Field(
        default=None,
        description="Base damage type (None for non-damaging weapons)",
    )
    license_id: LicenseIdField | None = Field(
        default=None,
        description="License ID required to use this weapon (None for GMS/general)",
    )
    license_rank: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="Required license rank if gated by a specific license",
    )
    integrated_only: bool = False
    integrated_frame_id: FrameIdField | None = Field(
        default=None,
        description="Frame ID that provides this weapon as an integrated mount",
    )
    unique: bool = False
    ranges: list[WeaponRange] = Field(default_factory=list)
    damage: list[WeaponDamage] = Field(default_factory=list)
    tags: list[WeaponTag] = Field(default_factory=list)
    limited_uses: int | None = Field(default=None, ge=0)
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)
    dynamic: DynamicWeaponDefinition | None = None


def resolve_weapon_profile(
    weapon: MechWeaponDefinition,
    profile_id: WeaponIdField | None = None,
) -> WeaponProfile:
    """Resolve a weapon profile, falling back to the base weapon definition."""
    if weapon.dynamic and weapon.dynamic.profile_choice:
        choice = weapon.dynamic.profile_choice
        chosen_id = profile_id or choice.default_profile_id
        if chosen_id:
            for profile in choice.profiles:
                if profile.profile_id == chosen_id:
                    return profile
            raise ValueError(f"Unknown profile '{chosen_id}' for weapon '{weapon.id}'.")
        if choice.profiles:
            return choice.profiles[0]

    return WeaponProfile(
        profile_id=weapon.id,
        name=weapon.name,
        damage_type=weapon.damage_type,
        ranges=weapon.ranges,
        damage=weapon.damage,
        tags=weapon.tags,
        effects=weapon.effects,
    )
