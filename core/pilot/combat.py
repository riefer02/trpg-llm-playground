"""Pilot combat rules and stats for Lancer TTRPG."""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.enums import SizeClass


class PilotCombatBaseStats(FrozenModel):
    """Base combat stats for a pilot."""

    size: SizeClass = "size_half"
    hp: int = Field(default=6, ge=0)
    evasion: int = Field(default=10, ge=0)
    e_defense: int = Field(default=10, ge=0)
    speed: int = Field(default=4, ge=0)


DEFAULT_PILOT_COMBAT_STATS = PilotCombatBaseStats()


class PilotDamageSeverity(FrozenModel):
    """Damage severity bands for pilots."""

    name: Literal["minor", "major", "lethal"]
    min_damage: int = Field(..., ge=0)
    max_damage: int | None = Field(default=None, ge=0)


PILOT_DAMAGE_SEVERITY_BANDS: list[PilotDamageSeverity] = [
    PilotDamageSeverity(name="minor", min_damage=1, max_damage=2),
    PilotDamageSeverity(name="major", min_damage=3, max_damage=5),
    PilotDamageSeverity(name="lethal", min_damage=6, max_damage=None),
]


class DownAndOutRule(FrozenModel):
    """Down and out resolution when a pilot hits 0 HP."""

    roll_die: Literal["1d6"] = "1d6"
    recover_roll: int = Field(default=6, ge=1, le=6)
    death_roll: int = Field(default=1, ge=1, le=6)
    down_and_out_range_min: int = Field(default=2, ge=1, le=6)
    down_and_out_range_max: int = Field(default=5, ge=1, le=6)
    down_and_out_evasion: int = Field(default=5, ge=0)
    additional_damage_causes_death: bool = True
    voluntary_death_allowed: bool = True


DEFAULT_DOWN_AND_OUT_RULE = DownAndOutRule()


class PilotRestRule(FrozenModel):
    """Rest and recovery rules for pilots."""

    short_rest_hours: int = Field(default=1, ge=1)
    short_rest_hp_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    full_rest_hours: int = Field(default=10, ge=1)


DEFAULT_PILOT_REST_RULE = PilotRestRule()


class PilotCombatRules(FrozenModel):
    """Combined pilot combat rule references."""

    base_stats: PilotCombatBaseStats = DEFAULT_PILOT_COMBAT_STATS
    damage_bands: list[PilotDamageSeverity] = Field(
        default_factory=lambda: PILOT_DAMAGE_SEVERITY_BANDS
    )
    down_and_out: DownAndOutRule = DEFAULT_DOWN_AND_OUT_RULE
    rest: PilotRestRule = DEFAULT_PILOT_REST_RULE
    max_armor: int = Field(default=2, ge=0)
    armor_piercing_ignores_armor: bool = True


DEFAULT_PILOT_COMBAT_RULES = PilotCombatRules()


class PilotInMechCombatRules(FrozenModel):
    """Overrides for pilots while fighting in mech combat."""

    use_grit_for_attacks_and_saves: bool = True
    immune_to_tech_actions: bool = True
    heat_converts_to_energy_damage: bool = True
    cannot_engage_mechs: bool = True


DEFAULT_PILOT_IN_MECH_RULES = PilotInMechCombatRules()
