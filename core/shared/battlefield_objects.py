"""Battlefield object states for destroyed mechs and wreckage.

Provides type-safe models for battlefield objects per PR2 rules:
- Destroyed mechs become objects with hard cover and difficult terrain
- 10 HP per size, evasion 5, armor 0
- Can be moved/dragged as terrain
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.mech.grid import HexPosition


class ObjectKind(FrozenModel):
    """Classification of battlefield object types."""

    kind: Literal["destroyed_mech", "wreckage", "debris", "ruins", "other"]


class ObjectState(FrozenModel):
    """Generic battlefield object state.

    Per PR2 rules:
    - Objects have Evasion 5, 10 HP/size, armor 0
    - Can be targeted by attacks
    - Can be destroyed (at 0 HP)
    - Can be moved/dragged as difficult terrain
    """

    id: str = Field(..., description="Unique identifier for this object")
    name: str = Field(..., description="Display name")
    position: HexPosition | None = Field(
        default=None, description="Grid position (None if not on map)"
    )
    size: int = Field(..., ge=1, description="Size category (1 = size 1)")
    hp: int = Field(..., ge=0, description="Current HP")
    max_hp: int = Field(..., ge=1, description="Maximum HP")
    evasion: int = Field(default=5, ge=0, description="Evasion value")
    armor: int = Field(default=0, ge=0, description="Armor value")
    e_defense: int = Field(default=10, ge=0, description="E-Defense value")
    is_destroyed: bool = Field(default=False, description="Whether object is destroyed")
    provides_soft_cover: bool = Field(
        default=False, description="Provides soft cover (+1 difficulty)"
    )
    provides_hard_cover: bool = Field(
        default=False, description="Provides hard cover (+2 difficulty)"
    )
    is_difficult_terrain: bool = Field(
        default=False, description="Is difficult terrain"
    )
    is_passable: bool = Field(default=True, description="Can be moved through")
    terrain_type: str | None = Field(
        default=None, description="Associated terrain type identifier"
    )
    kind: ObjectKind = Field(
        default_factory=lambda: ObjectKind(kind="other"),
        description="Object classification",
    )


class DestroyedMechObject(ObjectState):
    """Destroyed mech that becomes a battlefield object per PR2.

    PR2 Rules (PR2 4717-4725):
    - When destroyed, a mech becomes an object on the battlefield
    - Provides hard cover
    - Climbing over or moving through is difficult terrain
    - Can be moved and dragged around
    - Can be repaired during rest (4 repairs)
    """

    original_mech_id: str = Field(
        ..., description="ID of the original mech before destruction"
    )
    original_mech_name: str = Field(..., description="Name of the original mech")
    original_mech_size: int = Field(..., ge=1, description="Size of the original mech")
    original_owner_id: str | None = Field(
        default=None, description="Pilot/mech that owned this"
    )
    is_wreckage: bool = Field(
        default=False, description="True if this is wreckage from reactor meltdown"
    )
    provides_hard_cover: bool = Field(
        default=True, description="Destroyed mechs provide hard cover"
    )
    is_difficult_terrain: bool = Field(
        default=True, description="Moving through is difficult terrain"
    )
    is_passable: bool = Field(
        default=False, description="Cannot move through without penalty"
    )
    kind: ObjectKind = Field(
        default_factory=lambda: ObjectKind(kind="destroyed_mech"),
        description="Classification as destroyed mech",
    )

    @classmethod
    def from_combatant(
        cls,
        combatant_id: str,
        combatant_name: str,
        position: HexPosition | None,
        size_value: int,
        object_id: str,
        is_wreckage: bool = False,
        owner_id: str | None = None,
    ) -> "DestroyedMechObject":
        """Create DestroyedMechObject from a destroyed combatant.

        Args:
            combatant_id: ID of the destroyed combatant
            combatant_name: Name of the destroyed combatant
            position: Position on the battlefield
            size_value: Size of the mech (1-4 typically)
            object_id: Unique ID for the new object
            is_wreckage: True if destroyed by reactor meltdown
            owner_id: Pilot ID if applicable

        Returns:
            New DestroyedMechObject instance
        """
        return cls(
            id=object_id,
            name=f"Wreckage of {combatant_name}"
            if not is_wreckage
            else f"Remains of {combatant_name}",
            position=position,
            size=size_value,
            hp=10 * size_value,
            max_hp=10 * size_value,
            evasion=5,
            armor=0,
            original_mech_id=combatant_id,
            original_mech_name=combatant_name,
            original_mech_size=size_value,
            original_owner_id=owner_id,
            is_wreckage=is_wreckage,
            provides_hard_cover=True,
            is_difficult_terrain=True,
            is_passable=False,
            kind=ObjectKind(kind="destroyed_mech" if not is_wreckage else "wreckage"),
        )


class WreckageState(DestroyedMechObject):
    """Wreckage from reactor meltdown (annihilated wreck).

    Per PR2: Reactor meltdown annihilates the wreck - no remains to repair.
    This is distinct from normal destruction where wreckage can be repaired.
    """

    is_annihilated: bool = Field(
        default=False, description="True if annihilated by reactor meltdown"
    )

    @classmethod
    def from_meltdown(
        cls,
        combatant_id: str,
        combatant_name: str,
        position: HexPosition | None,
        size_value: int,
        object_id: str,
    ) -> "WreckageState":
        """Create WreckageState from a mech that suffered reactor meltdown.

        Args:
            combatant_id: ID of the destroyed combatant
            combatant_name: Name of the destroyed combatant
            position: Position on the battlefield
            size_value: Size of the mech
            object_id: Unique ID for the new object

        Returns:
            WreckageState representing annihilated mech
        """
        return cls(
            id=object_id,
            name=f"Scorched earth where {combatant_name} stood",
            position=position,
            size=size_value,
            hp=0,
            max_hp=10 * size_value,
            is_destroyed=True,
            is_annihilated=True,
            original_mech_id=combatant_id,
            original_mech_name=combatant_name,
            original_mech_size=size_value,
            provides_hard_cover=False,
            is_difficult_terrain=False,
            is_passable=True,
            kind=ObjectKind(kind="wreckage"),
        )


def damage_object(
    obj: ObjectState,
    damage: int,
    armor_piercing: int = 0,
) -> tuple[ObjectState, bool]:
    """Apply damage to a battlefield object.

    Args:
        obj: The object to damage
        damage: Amount of damage
        armor_piercing: Armor piercing value

    Returns:
        Tuple of (damaged object, whether object was destroyed)
    """
    effective_armor = max(0, obj.armor - armor_piercing)
    net_damage = max(0, damage - effective_armor)
    new_hp = max(0, obj.hp - net_damage)
    is_destroyed = new_hp == 0

    updated_obj = obj.model_copy(update={"hp": new_hp, "is_destroyed": is_destroyed})

    return updated_obj, is_destroyed


def get_object_defense(
    obj: ObjectState,
    attack_type: Literal["ranged", "melee", "tech"],
) -> int:
    """Get the defense value for targeting an object.

    Args:
        obj: The object being targeted
        attack_type: Type of attack being made

    Returns:
        Defense value to beat
    """
    if attack_type == "tech":
        return obj.e_defense
    return obj.evasion
