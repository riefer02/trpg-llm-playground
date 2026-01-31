"""Mech damage state tracking for downtime repairs.

Tracks current HP, structure, stress, and destroyed weapons/systems
between combats. Used for salvage spending and repair mechanics.
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.id_helpers import WeaponIdField, SystemIdField


class MechDamageState(FrozenModel):
    """Damage state for a mech between combats.

    Tracks current resource levels and destroyed components.
    Default values represent a fully healthy mech (full HP, structure, stress,
    no destroyed components).
    """

    hp_current: int = Field(..., ge=0, description="Current HP")
    structure_current: int = Field(
        ..., ge=0, le=4, description="Current structure (0-4)"
    )
    stress_current: int = Field(..., ge=0, description="Current reactor stress")
    destroyed_weapons: list[WeaponIdField] = Field(
        default_factory=list,
        description="Weapon IDs that are destroyed and need repair",
    )
    destroyed_systems: list[SystemIdField] = Field(
        default_factory=list,
        description="System IDs that are destroyed and need repair",
    )
    is_destroyed: bool = Field(
        default=False,
        description="Whether the mech is completely destroyed (wreck)",
    )

    @classmethod
    def full_health(
        cls, hp_max: int, structure_max: int = 4, stress_capacity: int = 0
    ) -> "MechDamageState":
        """Create a damage state representing a fully healthy mech.

        Args:
            hp_max: Maximum HP for this mech
            structure_max: Maximum structure (default 4 for mechs)
            stress_capacity: Maximum stress capacity (heat cap)

        Returns:
            A damage state with current values set to max HP/structure, zero stress
        """
        return cls(
            hp_current=hp_max,
            structure_current=structure_max,
            stress_current=0,
            destroyed_weapons=[],
            destroyed_systems=[],
            is_destroyed=False,
        )

    def is_full_health(
        self, hp_max: int, structure_max: int = 4, stress_capacity: int = 0
    ) -> bool:
        """Check if the mech is at full health.

        Args:
            hp_max: Maximum HP for this mech
            structure_max: Maximum structure (default 4)
            stress_capacity: Maximum stress capacity (heat cap)

        Returns:
            True if all resources are at max and no destroyed components
        """
        return (
            self.hp_current == hp_max
            and self.structure_current == structure_max
            and self.stress_current == 0
            and not self.destroyed_weapons
            and not self.destroyed_systems
            and not self.is_destroyed
        )

    def apply_repair(
        self,
        repair_type: Literal[
            "hp",
            "structure",
            "stress",
            "destroyed_weapon",
            "destroyed_system",
            "destroyed_mech",
        ],
        hp_max: int,
        structure_max: int = 4,
        stress_capacity: int = 0,
        weapon_id: WeaponIdField | None = None,
        system_id: SystemIdField | None = None,
    ) -> "MechDamageState":
        """Apply a repair effect to this damage state.

        Args:
            repair_type: Type of repair to apply
            hp_max: Maximum HP for this mech
            structure_max: Maximum structure (default 4)
            stress_capacity: Maximum stress capacity (heat cap)
            weapon_id: Required for destroyed_weapon repair
            system_id: Required for destroyed_system repair

        Returns:
            Updated damage state
        """
        updates: dict[str, object] = {}

        if repair_type == "hp":
            updates["hp_current"] = hp_max
        elif repair_type == "structure":
            updates["structure_current"] = min(
                self.structure_current + 1, structure_max
            )
        elif repair_type == "stress":
            updates["stress_current"] = max(self.stress_current - 1, 0)
        elif repair_type == "destroyed_weapon":
            if not weapon_id:
                raise ValueError("weapon_id required for destroyed_weapon repair")
            updates["destroyed_weapons"] = [
                w for w in self.destroyed_weapons if w != weapon_id
            ]
        elif repair_type == "destroyed_system":
            if not system_id:
                raise ValueError("system_id required for destroyed_system repair")
            updates["destroyed_systems"] = [
                s for s in self.destroyed_systems if s != system_id
            ]
        elif repair_type == "destroyed_mech":
            # Restore destroyed mech to 1 structure, 1 stress, full HP
            updates.update(
                {
                    "is_destroyed": False,
                    "hp_current": hp_max,
                    "structure_current": 1,
                    "stress_current": 1,
                }
            )
        else:
            raise ValueError(f"Unknown repair type: {repair_type}")

        return self.model_copy(update=updates)
