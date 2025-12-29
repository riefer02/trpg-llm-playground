"""Mech build models and stat calculations for Lancer TTRPG."""

import re

from pydantic import Field
from core.shared.models import FrozenModel

from core.pilot.skill import SkillSet
from core.mech.frame import MechFrameDefinition
from core.mech.weapon import (
    MechWeaponDefinition,
    WeaponSize,
    WeaponRange,
    WeaponDamage,
    WeaponTag,
)
from core.mech.system import MechSystemDefinition
from core.mech.mounts import MountSlot
from core.shared.enums import SizeClass
from core.shared.effects import (
    MechanicalEffect,
    StatModifier,
    LimitedUseBonusEffect,
    OverchargeCostCapEffect,
    AISystemLimitEffect,
    WeaponGrantEffect,
)


class MountedWeapon(FrozenModel):
    """Weapon installed on a specific mount slot."""

    mount_index: int = Field(..., ge=0)
    weapon_id: str
    weapon_size: WeaponSize



class InstalledSystem(FrozenModel):
    """System installed on a mech."""

    system_id: str
    sp_cost: int | None = Field(default=None, ge=0)



class MechBuild(FrozenModel):
    """A mech build derived from a frame and loadout."""

    frame_id: str
    weapons: list[MountedWeapon] = Field(default_factory=list)
    systems: list[InstalledSystem] = Field(default_factory=list)


    def total_sp(self, system_definitions: dict[str, MechSystemDefinition] | None = None) -> int:
        """Total system points spent (uses definitions if needed)."""
        total = 0
        for system in self.systems:
            if system.sp_cost is not None:
                total += system.sp_cost
                continue
            if system_definitions:
                definition = system_definitions.get(system.system_id)
                if definition:
                    total += definition.sp_cost
        return total


def build_mounted_weapon(
    mount_index: int,
    weapon_id: str,
    weapon_definitions: dict[str, MechWeaponDefinition] | None = None,
) -> MountedWeapon:
    """Create a mounted weapon using compendium definitions."""
    if weapon_definitions is None:
        from core.mech.compendium import WEAPON_DEFINITIONS_BY_ID

        weapon_definitions = WEAPON_DEFINITIONS_BY_ID
    weapon_def = weapon_definitions.get(weapon_id)
    if not weapon_def:
        raise ValueError(f"Unknown weapon ID: {weapon_id}")
    return MountedWeapon(
        mount_index=mount_index,
        weapon_id=weapon_id,
        weapon_size=weapon_def.size,
    )


def build_installed_system(
    system_id: str,
    system_definitions: dict[str, MechSystemDefinition] | None = None,
) -> InstalledSystem:
    """Create an installed system using compendium definitions."""
    if system_definitions is None:
        from core.mech.compendium import SYSTEM_DEFINITIONS_BY_ID

        system_definitions = SYSTEM_DEFINITIONS_BY_ID
    system_def = system_definitions.get(system_id)
    if not system_def:
        raise ValueError(f"Unknown system ID: {system_id}")
    return InstalledSystem(system_id=system_id, sp_cost=system_def.sp_cost)


def _slugify_weapon_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def _resolve_weapon_grant_id(
    grant: WeaponGrantEffect,
    index: int,
    existing_ids: set[str],
) -> str:
    if grant.weapon_id:
        if grant.weapon_id in existing_ids:
            raise ValueError(f"Weapon grant ID '{grant.weapon_id}' already exists.")
        return grant.weapon_id
    base = _slugify_weapon_name(grant.name)
    if not base:
        base = f"weapon_grant_{index + 1}"
    else:
        base = f"grant_{base}"
    candidate = base
    suffix = 1
    while candidate in existing_ids:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def build_weapon_definition_from_grant(
    grant: WeaponGrantEffect,
    weapon_id: str,
) -> MechWeaponDefinition:
    """Build a weapon definition from a weapon grant effect."""
    ranges = [
        WeaponRange(range_type=range_spec.range_type, value=range_spec.value)
        for range_spec in grant.ranges
    ]
    tags = [WeaponTag(tag=tag.tag, value=tag.value) for tag in grant.tags]
    if any(spec.ap for spec in grant.damage) and not any(tag.tag == "ap" for tag in tags):
        tags.append(WeaponTag(tag="ap"))

    damage: list[WeaponDamage] = []
    for spec in grant.damage:
        if spec.damage_type == "burn":
            if spec.dice is not None:
                raise ValueError(
                    "Weapon grant burn damage with dice is unsupported; use burn tags instead."
                )
            if spec.flat > 0 and not any(tag.tag == "burn" for tag in tags):
                tags.append(WeaponTag(tag="burn", value=spec.flat))
            continue
        damage.append(
            WeaponDamage(damage_type=spec.damage_type, dice=spec.dice, flat=spec.flat)
        )

    return MechWeaponDefinition(
        id=weapon_id,
        name=grant.name,
        size=grant.size,
        weapon_type=grant.weapon_type,
        damage_type=damage[0].damage_type if damage else None,
        ranges=ranges,
        damage=damage,
        tags=tags,
        limited_uses=grant.limited_uses,
        unique=grant.unique,
        integrated_only=grant.integrated_mount,
    )


def resolve_weapon_grants(
    bonus_effects: list[MechanicalEffect] | None,
    weapon_definitions: dict[str, MechWeaponDefinition] | None = None,
) -> tuple[dict[str, MechWeaponDefinition], list[MountSlot]]:
    """Resolve weapon grant effects into weapon definitions and integrated mounts."""
    if not bonus_effects:
        return {}, []
    grants: list[WeaponGrantEffect] = []
    for effect in bonus_effects:
        grants.extend(effect.weapon_grants)
    if not grants:
        return {}, []
    existing_ids = set(weapon_definitions or {})
    definitions: dict[str, MechWeaponDefinition] = {}
    integrated_mounts: list[MountSlot] = []
    for index, grant in enumerate(grants):
        weapon_id = _resolve_weapon_grant_id(grant, index, existing_ids)
        existing_ids.add(weapon_id)
        definitions[weapon_id] = build_weapon_definition_from_grant(grant, weapon_id)
        if grant.integrated_mount:
            integrated_mounts.append(
                MountSlot(slot_type="integrated", integrated_weapon_id=weapon_id)
            )
    return definitions, integrated_mounts


def build_weapon_definitions_with_grants(
    weapon_definitions: dict[str, MechWeaponDefinition],
    bonus_effects: list[MechanicalEffect] | None = None,
) -> dict[str, MechWeaponDefinition]:
    """Extend weapon definitions with any weapon grants from effects."""
    grant_definitions, _ = resolve_weapon_grants(bonus_effects, weapon_definitions)
    if not grant_definitions:
        return weapon_definitions
    return {**weapon_definitions, **grant_definitions}


def build_mech_from_compendium(
    frame_id: str,
    weapon_mounts: list[tuple[int, str]],
    system_ids: list[str],
    weapon_definitions: dict[str, MechWeaponDefinition] | None = None,
    system_definitions: dict[str, MechSystemDefinition] | None = None,
    bonus_effects: list[MechanicalEffect] | None = None,
) -> MechBuild:
    """Build a mech loadout from compendium IDs."""
    if weapon_definitions is None:
        from core.mech.compendium import WEAPON_DEFINITIONS_BY_ID

        weapon_definitions = WEAPON_DEFINITIONS_BY_ID
    weapon_definitions = build_weapon_definitions_with_grants(
        weapon_definitions,
        bonus_effects,
    )
    weapons = [
        build_mounted_weapon(mount_index, weapon_id, weapon_definitions)
        for mount_index, weapon_id in weapon_mounts
    ]
    systems = [
        build_installed_system(system_id, system_definitions)
        for system_id in system_ids
    ]
    return MechBuild(frame_id=frame_id, weapons=weapons, systems=systems)


class MechDerivedStats(FrozenModel):
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
    overcharge_cost_caps: list[OverchargeCostCapEffect] = Field(default_factory=list)
    ai_system_limit: int = Field(default=1, ge=0)



class LimitedUseEntry(FrozenModel):
    """Effective limited uses for a single item instance."""

    item_id: str
    uses: int



class LimitedUseSummary(FrozenModel):
    """Effective limited uses for weapons and systems."""

    weapons: list[LimitedUseEntry] = Field(default_factory=list)
    systems: list[LimitedUseEntry] = Field(default_factory=list)



def compute_limited_uses(
    build: MechBuild,
    stats: MechDerivedStats,
    weapon_definitions: dict[str, MechWeaponDefinition],
    system_definitions: dict[str, MechSystemDefinition],
    bonus_effects: list[MechanicalEffect] | None = None,
) -> LimitedUseSummary:
    """Compute effective limited uses after applying engineering bonuses."""
    limited_bonus = stats.limited_bonus
    limited_use_bonuses = _collect_limited_use_bonuses(bonus_effects)
    weapon_entries: list[LimitedUseEntry] = []
    system_entries: list[LimitedUseEntry] = []

    for mounted in build.weapons:
        definition = weapon_definitions.get(mounted.weapon_id)
        if definition and definition.limited_uses is not None:
            extra_uses = _limited_use_bonus_for_weapon(
                definition,
                limited_use_bonuses,
            )
            weapon_entries.append(
                LimitedUseEntry(
                    item_id=mounted.weapon_id,
                    uses=definition.limited_uses + limited_bonus + extra_uses,
                )
            )

    for system in build.systems:
        definition = system_definitions.get(system.system_id)
        if definition and definition.limited_uses is not None:
            extra_uses = _limited_use_bonus_for_system(
                definition,
                limited_use_bonuses,
            )
            system_entries.append(
                LimitedUseEntry(
                    item_id=system.system_id,
                    uses=definition.limited_uses + limited_bonus + extra_uses,
                )
            )

    return LimitedUseSummary(weapons=weapon_entries, systems=system_entries)


def compute_mech_stats(
    frame: MechFrameDefinition,
    skills: SkillSet,
    grit: int,
    bonus_effects: list[MechanicalEffect] | None = None,
) -> MechDerivedStats:
    """Compute final mech stats from a frame, skill bonuses, grit, and effects."""
    base = frame.base_stats
    overcharge_cost_caps = _collect_overcharge_cost_caps(bonus_effects)
    ai_system_limit = compute_ai_system_limit(bonus_effects)
    size = base.size
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
                size, hp, armor, evasion, e_defense, speed, sensor_range, tech_attack, heat_cap, repair_cap, save_target, limited_bonus, system_points = _apply_stat_modifier(
                    mod,
                    size,
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
        size=size,
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
        overcharge_cost_caps=overcharge_cost_caps,
        ai_system_limit=ai_system_limit,
    )


def _apply_stat_modifier(
    mod: StatModifier,
    size: SizeClass,
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
) -> tuple[SizeClass, int, int, int, int, int, int, int, int, int, int, int, int]:
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
    elif mod.stat == "size":
        size = _adjust_size(size, mod.value)

    return (
        size,
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


def _adjust_size(size: SizeClass, delta: int) -> SizeClass:
    order: list[SizeClass] = ["size_half", "size_1", "size_2", "size_3", "size_4", "size_5"]
    index = order.index(size) + delta
    if index < 0 or index >= len(order):
        raise ValueError(f"Size adjustment {delta} out of bounds for {size}")
    return order[index]


def _collect_limited_use_bonuses(
    bonus_effects: list[MechanicalEffect] | None,
) -> list[LimitedUseBonusEffect]:
    if not bonus_effects:
        return []
    bonuses: list[LimitedUseBonusEffect] = []
    for effect in bonus_effects:
        bonuses.extend(effect.limited_use_bonuses)
    return bonuses


def _collect_overcharge_cost_caps(
    bonus_effects: list[MechanicalEffect] | None,
) -> list[OverchargeCostCapEffect]:
    if not bonus_effects:
        return []
    caps: list[OverchargeCostCapEffect] = []
    for effect in bonus_effects:
        caps.extend(effect.overcharge_cost_caps)
    return caps


def _collect_ai_system_limits(
    bonus_effects: list[MechanicalEffect] | None,
) -> list[AISystemLimitEffect]:
    if not bonus_effects:
        return []
    limits: list[AISystemLimitEffect] = []
    for effect in bonus_effects:
        limits.extend(effect.ai_system_limits)
    return limits


def compute_ai_system_limit(
    bonus_effects: list[MechanicalEffect] | None,
    base_limit: int = 1,
) -> int:
    """Resolve the maximum number of AI systems allowed."""
    limits = _collect_ai_system_limits(bonus_effects)
    if not limits:
        return base_limit
    bonus_total = sum(limit.bonus_systems for limit in limits)
    effective = base_limit + bonus_total
    explicit_limits = [
        limit.max_ai_systems for limit in limits if limit.max_ai_systems is not None
    ]
    if explicit_limits:
        effective = max(effective, max(explicit_limits))
    return effective


def _limited_use_bonus_for_weapon(
    definition: MechWeaponDefinition,
    bonuses: list[LimitedUseBonusEffect],
) -> int:
    tags = {tag.tag for tag in definition.tags}
    return _apply_limited_use_bonuses(
        bonuses,
        tags=tags,
        limited_uses=definition.limited_uses,
        item_type="weapon",
        is_deployable=False,
    )


def _limited_use_bonus_for_system(
    definition: MechSystemDefinition,
    bonuses: list[LimitedUseBonusEffect],
) -> int:
    tags = {tag.tag for tag in definition.tags}
    is_deployable = definition.deployable is not None or "deployable" in tags
    return _apply_limited_use_bonuses(
        bonuses,
        tags=tags,
        limited_uses=definition.limited_uses,
        item_type="system",
        is_deployable=is_deployable,
    )


def _apply_limited_use_bonuses(
    bonuses: list[LimitedUseBonusEffect],
    tags: set[str],
    limited_uses: int | None,
    item_type: str,
    is_deployable: bool,
) -> int:
    if limited_uses is None:
        return 0
    extra = 0
    for bonus in bonuses:
        applies_to = bonus.applies_to or ["weapon", "system", "deployable"]
        if item_type == "weapon" and "weapon" not in applies_to:
            continue
        if item_type == "system":
            type_match = "system" in applies_to or (is_deployable and "deployable" in applies_to)
            if not type_match:
                continue
        if not _limited_use_tag_match(bonus, tags, limited_uses):
            continue
        extra += bonus.bonus_uses
    return extra


def _limited_use_tag_match(
    bonus: LimitedUseBonusEffect,
    tags: set[str],
    limited_uses: int | None,
) -> bool:
    if not bonus.requires_tag:
        return True
    if bonus.requires_tag in tags:
        return True
    return bonus.requires_tag == "limited" and limited_uses is not None
