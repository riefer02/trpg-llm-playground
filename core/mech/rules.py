"""Mech rules and penalties for Lancer TTRPG."""

from pydantic import BaseModel, Field

from core.shared.effects import AccuracyModifier
from core.shared.enums import StatusType


class MechPilotingRules(BaseModel):
    """Rules for piloting a mech."""

    unlicensed_accuracy_penalty: AccuracyModifier = AccuracyModifier(value=-1)
    unlicensed_status_penalties: list[StatusType] = Field(
        default_factory=lambda: ["impaired", "slowed"],
    )

    model_config = {"frozen": True}


class CorePowerRules(BaseModel):
    """Core power availability rules."""

    starts_with_core_power: bool = True
    restores_on_mission_start: bool = True
    restores_on_full_repair: bool = True

    model_config = {"frozen": True}


class SystemPointRules(BaseModel):
    """System point bonus rules."""

    grit_bonus: bool = True
    systems_per_bonus_sp: int = Field(default=2, ge=1)

    model_config = {"frozen": True}


DEFAULT_MECH_PILOTING_RULES = MechPilotingRules()
DEFAULT_CORE_POWER_RULES = CorePowerRules()
DEFAULT_SYSTEM_POINT_RULES = SystemPointRules()
