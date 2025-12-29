"""Mech mount slots and compatibility for Lancer TTRPG."""
from pydantic import Field
from core.shared.enums import MountSlotType
from core.shared.models import FrozenModel

from core.mech.weapon import WeaponSize


class MountSlot(FrozenModel):
    """A mount slot on a mech frame."""

    slot_type: MountSlotType
    integrated_weapon_id: str | None = Field(
        default=None,
        description="Integrated weapon ID (only for integrated mounts)",
    )



def allowed_weapon_sizes(slot_type: MountSlotType) -> set[WeaponSize]:
    """Return allowed weapon sizes for a given mount slot."""
    if slot_type == "main":
        return {"main", "aux"}
    if slot_type == "heavy":
        return {"superheavy", "heavy", "main", "aux"}
    if slot_type == "aux_aux":
        return {"aux"}
    if slot_type == "main_aux":
        return {"main", "aux"}
    if slot_type == "flexible":
        return {"main", "aux"}
    if slot_type == "integrated":
        return {"aux", "main", "heavy", "superheavy"}
    raise ValueError(f"Unknown mount slot type: {slot_type}")
