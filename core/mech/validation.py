"""Validation helpers for mech builds."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

from core.mech.build import MechBuild, compute_mech_stats
from core.mech.frame import MechFrameDefinition
from core.mech.mounts import allowed_weapon_sizes
from core.mech.weapon import WeaponSize
from core.pilot.skill import SkillSet


class MechBuildIssue(BaseModel):
    """A mech build validation issue."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"

    model_config = {"frozen": True}


class MechBuildValidation(BaseModel):
    """Validation result for a mech build."""

    valid: bool
    issues: list[MechBuildIssue] = Field(default_factory=list)

    model_config = {"frozen": True}


def _validate_mount_allocation(
    frame: MechFrameDefinition,
    build: MechBuild,
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []
    mount_count = len(frame.mounts)
    weapons_by_mount: dict[int, list[WeaponSize]] = {}

    for mounted in build.weapons:
        if mounted.mount_index < 0 or mounted.mount_index >= mount_count:
            issues.append(
                MechBuildIssue(
                    code="mount_index_out_of_range",
                    message=f"Mount index {mounted.mount_index} is out of range.",
                )
            )
            continue
        weapons_by_mount.setdefault(mounted.mount_index, []).append(mounted.weapon_size)

    for index, sizes in weapons_by_mount.items():
        slot = frame.mounts[index]
        allowed = allowed_weapon_sizes(slot.slot_type)
        if slot.slot_type == "integrated" and slot.integrated_weapon_id:
            issues.append(
                MechBuildIssue(
                    code="integrated_mount_locked",
                    message=f"Mount {index} is integrated and cannot be reassigned.",
                )
            )
            continue

        for size in sizes:
            if size not in allowed:
                issues.append(
                    MechBuildIssue(
                        code="weapon_size_not_allowed",
                        message=(
                            f"Mount {index} does not allow weapon size '{size}'."
                        ),
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


def _validate_superheavy(
    frame: MechFrameDefinition,
    build: MechBuild,
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []
    superheavy_mounts = [
        mounted.mount_index for mounted in build.weapons if mounted.weapon_size == "superheavy"
    ]
    if not superheavy_mounts:
        return issues

    for mount_index in superheavy_mounts:
        slot = frame.mounts[mount_index]
        if slot.slot_type != "heavy":
            issues.append(
                MechBuildIssue(
                    code="superheavy_requires_heavy_mount",
                    message=f"Superheavy weapon must be mounted on a heavy mount (index {mount_index}).",
                )
            )

    used_mounts = {mounted.mount_index for mounted in build.weapons}
    free_mounts = len(frame.mounts) - len(used_mounts)
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
) -> list[MechBuildIssue]:
    issues: list[MechBuildIssue] = []
    stats = compute_mech_stats(frame, skills, grit)
    if build.total_sp() > stats.system_points:
        issues.append(
            MechBuildIssue(
                code="system_points_exceeded",
                message=(
                    f"System points spent {build.total_sp()} exceed budget "
                    f"{stats.system_points}."
                ),
            )
        )
    return issues


def validate_mech_build(
    frame: MechFrameDefinition,
    build: MechBuild,
    skills: SkillSet,
    grit: int,
) -> MechBuildValidation:
    """Validate a mech build against frame, skills, and grit."""
    issues: list[MechBuildIssue] = []

    issues.extend(_validate_mount_allocation(frame, build))
    issues.extend(_validate_superheavy(frame, build))
    issues.extend(_validate_system_points(frame, build, skills, grit))

    valid = not any(issue.severity == "error" for issue in issues)
    return MechBuildValidation(valid=valid, issues=issues)
