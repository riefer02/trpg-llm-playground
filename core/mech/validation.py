"""Validation helpers for mech builds."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING
from pydantic import Field
from core.shared.validation import ValidationIssue, ValidationResult
from core.shared.effects import MechanicalEffect, OverchargeCostCapEffect, ModeEffect
from core.shared.effects_validation import validate_mechanical_effects

from core.mech.build import (
    MechBuild,
    MountedWeapon,
    compute_mech_stats,
    compute_limited_uses,
    LimitedUseSummary,
    resolve_weapon_grants,
)
from core.mech.frame import MechFrameDefinition
from core.mech.mounts import MountSlot, allowed_weapon_sizes
from core.mech.weapon import MechWeaponDefinition, WeaponSize
from core.mech.system import MechSystemDefinition
from core.pilot.skill import SkillSet
from core.pilot.license import License

if TYPE_CHECKING:
    from typing import Literal

MechBuildIssue = ValidationIssue


class MechBuildValidation(ValidationResult):
    """Validation result for a mech build."""

    limited_uses: LimitedUseSummary = Field(default_factory=LimitedUseSummary)
    overcharge_cost_caps: list[OverchargeCostCapEffect] = Field(default_factory=list)
    ai_system_limit: int | None = None
    mode_effects: list[ModeEffect] = Field(default_factory=list)


def _validate_mount_allocation(
    frame: MechFrameDefinition,
    build: MechBuild,
    mounts: list[MountSlot] | None = None,
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []
    mount_slots = mounts or frame.mounts
    mount_count = len(mount_slots)
    weapons_by_mount: dict[int, list[MountedWeapon]] = {}

    for mounted in build.weapons:
        if mounted.mount_index < 0 or mounted.mount_index >= mount_count:
            issues.append(
                MechBuildIssue(
                    code="mount_index_out_of_range",
                    message=f"Mount index {mounted.mount_index} is out of range.",
                )
            )
            continue
        weapons_by_mount.setdefault(mounted.mount_index, []).append(mounted)

    for index, mounted_weapons in weapons_by_mount.items():
        slot = mount_slots[index]
        if slot.slot_type == "integrated":
            if not slot.integrated_weapon_id:
                issues.append(
                    MechBuildIssue(
                        code="integrated_mount_missing_weapon",
                        message=f"Mount {index} is integrated but has no weapon assigned.",
                    )
                )
                continue
            if len(mounted_weapons) > 1:
                issues.append(
                    MechBuildIssue(
                        code="integrated_mount_capacity",
                        message=f"Mount {index} allows only its integrated weapon.",
                    )
                )
            for mounted in mounted_weapons:
                if mounted.weapon_id != slot.integrated_weapon_id:
                    issues.append(
                        MechBuildIssue(
                            code="integrated_weapon_mismatch",
                            message=(
                                f"Mount {index} only allows integrated weapon "
                                f"'{slot.integrated_weapon_id}'."
                            ),
                        )
                    )
            continue

        sizes = [mounted.weapon_size for mounted in mounted_weapons]
        allowed = allowed_weapon_sizes(slot.slot_type)

        for size in sizes:
            if size not in allowed:
                issues.append(
                    MechBuildIssue(
                        code="weapon_size_not_allowed",
                        message=(f"Mount {index} does not allow weapon size '{size}'."),
                    )
                )

        if slot.slot_type in {"main", "heavy", "integrated"} and len(sizes) > 1:
            issues.append(
                MechBuildIssue(
                    code="too_many_weapons_on_mount",
                    message=f"Mount {index} allows only one weapon.",
                )
            )
            continue

        if slot.slot_type == "aux_aux":
            if any(size != "aux" for size in sizes):
                issues.append(
                    MechBuildIssue(
                        code="aux_aux_only",
                        message=f"Mount {index} only allows auxiliary weapons.",
                    )
                )
            if len(sizes) > 2:
                issues.append(
                    MechBuildIssue(
                        code="aux_aux_capacity",
                        message=f"Mount {index} allows at most 2 aux weapons.",
                    )
                )

        if slot.slot_type == "main_aux":
            main_count = sum(1 for size in sizes if size == "main")
            if main_count > 1:
                issues.append(
                    MechBuildIssue(
                        code="main_aux_main_limit",
                        message=f"Mount {index} allows at most 1 main weapon.",
                    )
                )
            if any(size not in {"main", "aux"} for size in sizes):
                issues.append(
                    MechBuildIssue(
                        code="main_aux_invalid_size",
                        message=f"Mount {index} only allows main or aux weapons.",
                    )
                )
            if len(sizes) > 2:
                issues.append(
                    MechBuildIssue(
                        code="main_aux_capacity",
                        message=f"Mount {index} allows at most 2 weapons.",
                    )
                )

        if slot.slot_type == "flexible":
            main_present = any(size == "main" for size in sizes)
            if main_present and len(sizes) > 1:
                issues.append(
                    MechBuildIssue(
                        code="flexible_main_capacity",
                        message=f"Mount {index} allows only 1 main weapon.",
                    )
                )
            if not main_present and len(sizes) > 2:
                issues.append(
                    MechBuildIssue(
                        code="flexible_aux_capacity",
                        message=f"Mount {index} allows at most 2 aux weapons.",
                    )
                )
            if any(size not in {"main", "aux"} for size in sizes):
                issues.append(
                    MechBuildIssue(
                        code="flexible_invalid_size",
                        message=f"Mount {index} only allows main or aux weapons.",
                    )
                )

    return issues


def _validate_integrated_weapon_restrictions(
    frame: MechFrameDefinition,
    build: MechBuild,
    weapon_definitions: dict[str, MechWeaponDefinition] | None = None,
    mounts: list[MountSlot] | None = None,
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []
    if not weapon_definitions:
        return issues
    mount_slots = mounts or frame.mounts
    mount_count = len(mount_slots)

    for mounted in build.weapons:
        definition = weapon_definitions.get(mounted.weapon_id)
        if not definition or not definition.integrated_only:
            continue
        if (
            definition.integrated_frame_id
            and frame.id != definition.integrated_frame_id
        ):
            issues.append(
                MechBuildIssue(
                    code="integrated_weapon_frame_mismatch",
                    message=(
                        f"Weapon '{definition.name}' is integrated to frame "
                        f"'{definition.integrated_frame_id}', not '{frame.id}'."
                    ),
                )
            )
        if mounted.mount_index < 0 or mounted.mount_index >= mount_count:
            continue
        slot = mount_slots[mounted.mount_index]
        if slot.slot_type != "integrated":
            issues.append(
                MechBuildIssue(
                    code="integrated_weapon_requires_integrated_mount",
                    message=(
                        f"Weapon '{definition.name}' must be mounted in an integrated slot."
                    ),
                )
            )
            continue
        if slot.integrated_weapon_id and slot.integrated_weapon_id != mounted.weapon_id:
            issues.append(
                MechBuildIssue(
                    code="integrated_weapon_mismatch",
                    message=(
                        f"Mount {mounted.mount_index} only allows integrated weapon "
                        f"'{slot.integrated_weapon_id}'."
                    ),
                )
            )

    return issues


def _validate_ai_system_limits(
    build: MechBuild,
    system_definitions: dict[str, MechSystemDefinition] | None,
    ai_system_limit: int | None,
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []
    if ai_system_limit is None or system_definitions is None:
        return issues

    ai_count = 0
    for system in build.systems:
        definition = system_definitions.get(system.system_id)
        if not definition:
            continue
        if any(tag.tag == "ai" for tag in definition.tags):
            ai_count += 1

    if ai_count > ai_system_limit:
        issues.append(
            MechBuildIssue(
                code="ai_system_limit_exceeded",
                message=(
                    f"AI systems installed ({ai_count}) exceed allowed limit "
                    f"({ai_system_limit})."
                ),
            )
        )
    return issues


def _collect_mode_effects(
    bonus_effects: list[MechanicalEffect] | None,
) -> list[ModeEffect]:
    if not bonus_effects:
        return []
    modes: list[ModeEffect] = []
    for effect in bonus_effects:
        modes.extend(effect.mode_effects)
    return modes


def _validate_superheavy(
    frame: MechFrameDefinition,
    build: MechBuild,
    mounts: list[MountSlot] | None = None,
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []
    mount_slots = mounts or frame.mounts
    available_mounts = [
        index
        for index, slot in enumerate(mount_slots)
        if slot.slot_type != "integrated"
    ]
    superheavy_mounts = [
        mounted.mount_index
        for mounted in build.weapons
        if mounted.weapon_size == "superheavy"
    ]
    if not superheavy_mounts:
        return issues

    for mount_index in superheavy_mounts:
        slot = mount_slots[mount_index]
        if slot.slot_type != "heavy":
            issues.append(
                MechBuildIssue(
                    code="superheavy_requires_heavy_mount",
                    message=f"Superheavy weapon must be mounted on a heavy mount (index {mount_index}).",
                )
            )

    used_mounts = {mounted.mount_index for mounted in build.weapons}
    used_non_integrated = {index for index in used_mounts if index in available_mounts}
    free_mounts = len(available_mounts) - len(used_non_integrated)
    if free_mounts < len(superheavy_mounts):
        issues.append(
            MechBuildIssue(
                code="superheavy_requires_extra_mount",
                message="Not enough free mounts to pay superheavy mount cost.",
            )
        )

    return issues


def _validate_system_points(
    frame: MechFrameDefinition,
    build: MechBuild,
    skills: SkillSet,
    grit: int,
    system_definitions: dict[str, MechSystemDefinition] | None = None,
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []
    stats = compute_mech_stats(frame, skills, grit)
    total_sp = build.total_sp(system_definitions)
    if total_sp > stats.system_points:
        issues.append(
            MechBuildIssue(
                code="system_points_exceeded",
                message=(
                    f"System points spent {total_sp} exceed budget "
                    f"{stats.system_points}."
                ),
            )
        )
    return issues


def _build_license_lookup(
    licenses: list[License] | None,
    license_ranks: dict[str, int] | None,
) -> dict[str, int] | None:
    if license_ranks is not None:
        return license_ranks
    if licenses is None:
        return None
    return {lic.license_id: lic.rank for lic in licenses}


def _validate_license_access(
    label: str,
    license_id: str | None,
    license_rank: int | None,
    license_lookup: dict[str, int] | None,
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []
    if not license_lookup or not license_id:
        return issues
    required_rank = license_rank or 1
    current_rank = license_lookup.get(license_id, 0)
    if current_rank < required_rank:
        issues.append(
            MechBuildIssue(
                code="license_requirement_not_met",
                message=(
                    f"{label} requires license '{license_id}' rank {required_rank}, "
                    f"but current rank is {current_rank}."
                ),
            )
        )
    return issues


def _validate_unique_tags(
    build: MechBuild,
    weapon_definitions: dict[str, MechWeaponDefinition] | None = None,
    system_definitions: dict[str, MechSystemDefinition] | None = None,
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []

    if weapon_definitions:
        weapon_counts = Counter(mounted.weapon_id for mounted in build.weapons)
        for weapon_id, count in weapon_counts.items():
            if count <= 1:
                continue
            weapon_def = weapon_definitions.get(weapon_id)
            if weapon_def and weapon_def.unique:
                issues.append(
                    MechBuildIssue(
                        code="unique_weapon_duplicate",
                        message=f"Weapon '{weapon_id}' is unique and cannot be duplicated.",
                    )
                )

    if system_definitions:
        system_counts = Counter(installed.system_id for installed in build.systems)
        for system_id, count in system_counts.items():
            if count <= 1:
                continue
            system_def = system_definitions.get(system_id)
            if system_def and system_def.unique:
                issues.append(
                    MechBuildIssue(
                        code="unique_system_duplicate",
                        message=f"System '{system_id}' is unique and cannot be duplicated.",
                    )
                )

    return issues


def validate_mech_build(
    frame: MechFrameDefinition,
    build: MechBuild,
    skills: SkillSet,
    grit: int,
    weapon_definitions: dict[str, MechWeaponDefinition] | None = None,
    system_definitions: dict[str, MechSystemDefinition] | None = None,
    licenses: list[License] | None = None,
    license_ranks: dict[str, int] | None = None,
    bonus_effects: list[MechanicalEffect] | None = None,
) -> MechBuildValidation:
    """Validate a mech build against frame, skills, and grit."""
    issues: list[MechBuildIssue] = []

    if weapon_definitions is None or system_definitions is None:
        from core.mech.compendium import (
            WEAPON_DEFINITIONS_BY_ID,
            SYSTEM_DEFINITIONS_BY_ID,
        )

        weapon_definitions = weapon_definitions or WEAPON_DEFINITIONS_BY_ID
        system_definitions = system_definitions or SYSTEM_DEFINITIONS_BY_ID

    grant_definitions: dict[str, MechWeaponDefinition] = {}
    grant_mounts: list[MountSlot] = []
    try:
        grant_definitions, grant_mounts = resolve_weapon_grants(
            bonus_effects,
            weapon_definitions,
        )
    except ValueError as exc:
        issues.append(
            MechBuildIssue(
                code="weapon_grant_conflict",
                message=str(exc),
            )
        )
    if grant_definitions:
        weapon_definitions = {**weapon_definitions, **grant_definitions}

    effective_mounts = [*frame.mounts, *grant_mounts]

    issues.extend(_validate_mount_allocation(frame, build, mounts=effective_mounts))
    issues.extend(_validate_superheavy(frame, build, mounts=effective_mounts))
    issues.extend(
        _validate_system_points(frame, build, skills, grit, system_definitions)
    )
    issues.extend(_validate_unique_tags(build, weapon_definitions, system_definitions))
    issues.extend(
        _validate_integrated_weapon_restrictions(
            frame,
            build,
            weapon_definitions,
            mounts=effective_mounts,
        )
    )

    license_lookup = _build_license_lookup(licenses, license_ranks)
    issues.extend(
        _validate_license_access(
            label=f"Frame '{frame.name}'",
            license_id=frame.license_id,
            license_rank=frame.license_rank,
            license_lookup=license_lookup,
        )
    )

    for mounted in build.weapons:
        definition = weapon_definitions.get(mounted.weapon_id)
        if not definition:
            continue
        issues.extend(
            _validate_license_access(
                label=f"Weapon '{definition.name}'",
                license_id=definition.license_id,
                license_rank=definition.license_rank,
                license_lookup=license_lookup,
            )
        )

    for system in build.systems:
        definition = system_definitions.get(system.system_id)
        if not definition:
            continue
        issues.extend(
            _validate_license_access(
                label=f"System '{definition.name}'",
                license_id=definition.license_id,
                license_rank=definition.license_rank,
                license_lookup=license_lookup,
            )
        )

    effect_sources: list[MechanicalEffect] = []
    if frame.core_system:
        effect_sources.append(frame.core_system.effects)
    effect_sources.extend(trait.effects for trait in frame.traits)
    effect_sources.extend(bonus_effects or [])
    for mounted in build.weapons:
        definition = weapon_definitions.get(mounted.weapon_id)
        if definition:
            effect_sources.append(definition.effects)
    for system in build.systems:
        definition = system_definitions.get(system.system_id)
        if definition:
            effect_sources.append(definition.effects)
    for issue in validate_mechanical_effects(effect_sources):
        issues.append(
            MechBuildIssue(
                code=f"effect_validation_{issue.severity}",
                message=f"Effect validation: {issue.message}",
                severity=issue.severity,
            )
        )

    stats = compute_mech_stats(frame, skills, grit, bonus_effects=bonus_effects)
    ai_system_limit = stats.ai_system_limit
    issues.extend(
        _validate_ai_system_limits(build, system_definitions, ai_system_limit)
    )
    limited_uses = compute_limited_uses(
        build,
        stats,
        weapon_definitions,
        system_definitions,
        bonus_effects=bonus_effects,
    )
    mode_effects = _collect_mode_effects(bonus_effects)

    valid = not any(issue.severity == "error" for issue in issues)
    return MechBuildValidation(
        valid=valid,
        issues=issues,
        limited_uses=limited_uses,
        overcharge_cost_caps=stats.overcharge_cost_caps,
        ai_system_limit=ai_system_limit,
        mode_effects=mode_effects,
    )
