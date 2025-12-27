"""Mech build models and stat calculations for Lancer TTRPG."""

from pydantic import BaseModel, Field

from core.pilot.skill import SkillSet
from core.mech.frame import MechFrameDefinition
from core.mech.weapon import WeaponSize
from core.shared.enums import SizeClass
from core.shared.effects import MechanicalEffect, StatModifier


class MountedWeapon(BaseModel):
    """Weapon installed on a specific mount slot."""

    mount_index: int = Field(..., ge=0)
    weapon_id: str
    weapon_size: WeaponSize

    model_config = {"frozen": True}


class InstalledSystem(BaseModel):
    """System installed on a mech."""

    system_id: str
    sp_cost: int | None = Field(default=None, ge=0)

    model_config = {"frozen": True}


class MechBuild(BaseModel):
    """A mech build derived from a frame and loadout."""

    frame_id: str
    weapons: list[MountedWeapon] = Field(default_factory=list)
    systems: list[InstalledSystem] = Field(default_factory=list)

    model_config = {"frozen": True}

    def total_sp(self) -> int:
        """Total system points spent (if provided)."""
        return sum(system.sp_cost or 0 for system in self.systems)


class MechDerivedStats(BaseModel):
    """Final mech stats after pilot bonuses are applied."""

    size: SizeClass
    armor: int
    hp: int
    evasion: int
    e_defense: int
    speed: int
    sensor_range: int
    tech_attack: int
    heat_cap: int
    repair_cap: int
    save_target: int
    structure: int
    system_points: int
    attack_bonus: int
    limited_bonus: int

    model_config = {"frozen": True}


def compute_mech_stats(
    frame: MechFrameDefinition,
    skills: SkillSet,
    grit: int,
    bonus_effects: list[MechanicalEffect] | None = None,
) -> MechDerivedStats:
    """Compute final mech stats from a frame, skill bonuses, grit, and effects."""
    base = frame.base_stats
    hull = skills.hull
    agility = skills.agility
    systems = skills.systems
    engineering = skills.engineering

    armor = base.armor
    hp = base.hp + grit + (2 * hull)
    repair_cap = base.repair_cap + (hull // 2)
    evasion = base.evasion + agility
    speed = base.speed + (agility // 2)
    sensor_range = base.sensor_range
    tech_attack = base.tech_attack + systems
    e_defense = base.e_defense + systems
    heat_cap = base.heat_cap + engineering
    system_points = frame.system_points + grit + (systems // 2)
    save_target = base.save_target + grit
    limited_bonus = engineering // 2

    if bonus_effects:
        for effect in bonus_effects:
            for mod in effect.stat_mods:
                hp, armor, evasion, e_defense, speed, sensor_range, tech_attack, heat_cap, repair_cap, save_target, limited_bonus, system_points = _apply_stat_modifier(
                    mod,
                    hp,
                    armor,
                    evasion,
                    e_defense,
                    speed,
                    sensor_range,
                    tech_attack,
                    heat_cap,
                    repair_cap,
                    save_target,
                    limited_bonus,
                    system_points,
                )

    return MechDerivedStats(
        size=base.size,
        armor=armor,
        hp=hp,
        evasion=evasion,
        e_defense=e_defense,
        speed=speed,
        sensor_range=sensor_range,
        tech_attack=tech_attack,
        heat_cap=heat_cap,
        repair_cap=repair_cap,
        save_target=save_target,
        structure=base.structure,
        system_points=system_points,
        attack_bonus=grit,
        limited_bonus=limited_bonus,
    )


def _apply_stat_modifier(
    mod: StatModifier,
    hp: int,
    armor: int,
    evasion: int,
    e_defense: int,
    speed: int,
    sensor_range: int,
    tech_attack: int,
    heat_cap: int,
    repair_cap: int,
    save_target: int,
    limited_bonus: int,
    system_points: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
    """Apply a stat modifier to mech stats."""
    if mod.stat == "hp":
        hp += mod.value
    elif mod.stat == "armor":
        armor += mod.value
    elif mod.stat == "evasion":
        evasion += mod.value
    elif mod.stat == "e_defense":
        e_defense += mod.value
    elif mod.stat == "speed":
        speed += mod.value
    elif mod.stat == "sensor_range":
        sensor_range += mod.value
    elif mod.stat == "tech_attack":
        tech_attack += mod.value
    elif mod.stat == "heat_cap":
        heat_cap += mod.value
    elif mod.stat == "repair_cap":
        repair_cap += mod.value
    elif mod.stat == "save_target":
        save_target += mod.value
    elif mod.stat == "limited_bonus":
        limited_bonus += mod.value
    elif mod.stat == "system_points":
        system_points += mod.value

    return (
        hp,
        armor,
        evasion,
        e_defense,
        speed,
        sensor_range,
        tech_attack,
        heat_cap,
        repair_cap,
        save_target,
        limited_bonus,
        system_points,
    )
