"""Character validation for Lancer TTRPG.

Provides holistic validation of a Character including:
- Pilot progression validation
- Mech build validation for each mech
- License gating checks
- LL0-specific rules (GMS only, exact point allocation)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pydantic import Field

from core.shared.validation import ValidationIssue, ValidationResult
from core.pilot.validation import validate_pilot_progression
from core.mech.build_validation import validate_mech_build, MechBuildValidation
from core.mech.compendium import (
    get_frame_definition,
    WEAPON_DEFINITIONS_BY_ID,
    SYSTEM_DEFINITIONS_BY_ID,
)

if TYPE_CHECKING:
    from core.character.character import Character, MechConfiguration

CharacterIssue = ValidationIssue


class MechValidationEntry(ValidationResult):
    """Validation result for a single mech."""

    mech_id: str
    mech_name: str
    build_validation: MechBuildValidation | None = None


class CharacterValidation(ValidationResult):
    """Validation result for a complete character."""

    pilot_valid: bool = True
    pilot_issues: list[CharacterIssue] = Field(default_factory=list)
    mech_validations: list[MechValidationEntry] = Field(default_factory=list)


def _validate_gms_only_frame(
    mech: "MechConfiguration",
) -> list[CharacterIssue]:
    """Validate that a mech uses only GMS frame (for LL0)."""
    issues: list[CharacterIssue] = []
    frame = get_frame_definition(mech.frame_id)

    if frame is None:
        issues.append(
            CharacterIssue(
                code="unknown_frame",
                message=f"Mech '{mech.name}': Unknown frame ID '{mech.frame_id}'.",
            )
        )
        return issues

    # GMS frames have license_id = None
    if frame.license_id is not None:
        issues.append(
            CharacterIssue(
                code="ll0_non_gms_frame",
                message=(
                    f"Mech '{mech.name}': LL0 characters can only use GMS frames. "
                    f"Frame '{frame.name}' requires license '{frame.license_id}'."
                ),
            )
        )

    return issues


def _validate_gms_only_weapons(
    mech: "MechConfiguration",
) -> list[CharacterIssue]:
    """Validate that a mech uses only GMS weapons (for LL0)."""
    issues: list[CharacterIssue] = []

    for mounted in mech.build.weapons:
        weapon_def = WEAPON_DEFINITIONS_BY_ID.get(mounted.weapon_id)
        if weapon_def is None:
            issues.append(
                CharacterIssue(
                    code="unknown_weapon",
                    message=(
                        f"Mech '{mech.name}': Unknown weapon ID '{mounted.weapon_id}'."
                    ),
                )
            )
            continue

        if weapon_def.license_id is not None:
            issues.append(
                CharacterIssue(
                    code="ll0_non_gms_weapon",
                    message=(
                        f"Mech '{mech.name}': LL0 characters can only use GMS weapons. "
                        f"Weapon '{weapon_def.name}' requires license '{weapon_def.license_id}'."
                    ),
                )
            )

    return issues


def _validate_gms_only_systems(
    mech: "MechConfiguration",
) -> list[CharacterIssue]:
    """Validate that a mech uses only GMS systems (for LL0)."""
    issues: list[CharacterIssue] = []

    for installed in mech.build.systems:
        system_def = SYSTEM_DEFINITIONS_BY_ID.get(installed.system_id)
        if system_def is None:
            issues.append(
                CharacterIssue(
                    code="unknown_system",
                    message=(
                        f"Mech '{mech.name}': Unknown system ID '{installed.system_id}'."
                    ),
                )
            )
            continue

        if system_def.license_id is not None:
            issues.append(
                CharacterIssue(
                    code="ll0_non_gms_system",
                    message=(
                        f"Mech '{mech.name}': LL0 characters can only use GMS systems. "
                        f"System '{system_def.name}' requires license '{system_def.license_id}'."
                    ),
                )
            )

    return issues


def _validate_ll0_character(character: "Character") -> list[CharacterIssue]:
    """Validate LL0-specific rules for a character.

    At LL0:
    - Must use GMS frame only (Everest)
    - Must use GMS weapons only
    - Must use GMS systems only
    - Exact point allocation (2 skill points, 4 triggers at +2, 3 rank I talents)
    """
    issues: list[CharacterIssue] = []

    # Validate each mech is GMS-only
    for mech in character.mechs:
        issues.extend(_validate_gms_only_frame(mech))
        issues.extend(_validate_gms_only_weapons(mech))
        issues.extend(_validate_gms_only_systems(mech))

    return issues


def _validate_mech_configuration(
    character: "Character",
    mech: "MechConfiguration",
) -> MechValidationEntry:
    """Validate a single mech configuration within a character."""
    issues: list[CharacterIssue] = []

    frame = get_frame_definition(mech.frame_id)
    if frame is None:
        issues.append(
            CharacterIssue(
                code="unknown_frame",
                message=f"Mech '{mech.name}': Unknown frame ID '{mech.frame_id}'.",
            )
        )
        return MechValidationEntry(
            valid=False,
            issues=issues,
            mech_id=mech.id,
            mech_name=mech.name,
            build_validation=None,
        )

    # Run full mech build validation
    build_validation = validate_mech_build(
        frame=frame,
        build=mech.build,
        skills=character.pilot.skills,
        grit=character.pilot.grit,
        licenses=character.pilot.licenses,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        bonus_effects=character.core_bonus_effects if character.core_bonus_effects else None,
    )

    # Convert build validation issues to character issues
    for issue in build_validation.issues:
        issues.append(
            CharacterIssue(
                code=f"mech_{issue.code}",
                message=f"Mech '{mech.name}': {issue.message}",
                severity=issue.severity,
            )
        )

    return MechValidationEntry(
        valid=build_validation.valid,
        issues=issues,
        mech_id=mech.id,
        mech_name=mech.name,
        build_validation=build_validation,
    )


def validate_character(character: "Character") -> CharacterValidation:
    """Validate a complete character against all game rules.

    Performs:
    1. Pilot progression validation (skill points, triggers, talents, licenses)
    2. Mech build validation for each mech (mounts, SP, license gating)
    3. LL0-specific rules if character is at license level 0

    Returns:
        CharacterValidation with all issues found
    """
    all_issues: list[CharacterIssue] = []

    # 1. Validate pilot progression
    pilot_validation = validate_pilot_progression(character.pilot)
    pilot_issues = [
        CharacterIssue(
            code=f"pilot_{issue.code}",
            message=issue.message,
            severity=issue.severity,
        )
        for issue in pilot_validation.issues
    ]
    all_issues.extend(pilot_issues)

    # 2. Validate each mech
    mech_validations: list[MechValidationEntry] = []
    for mech in character.mechs:
        mech_result = _validate_mech_configuration(character, mech)
        mech_validations.append(mech_result)
        all_issues.extend(mech_result.issues)

    # 3. LL0-specific validation
    if character.pilot.level == 0:
        ll0_issues = _validate_ll0_character(character)
        all_issues.extend(ll0_issues)

    # 4. Validate that character has at least one mech (warning)
    if not character.mechs:
        all_issues.append(
            CharacterIssue(
                code="no_mechs",
                message="Character has no mech configurations.",
                severity="warning",
            )
        )

    # 5. Validate active mech is set if mechs exist (warning)
    if character.mechs and character.active_mech_id is None:
        all_issues.append(
            CharacterIssue(
                code="no_active_mech",
                message="Character has mechs but no active mech selected.",
                severity="warning",
            )
        )

    valid = not any(issue.severity == "error" for issue in all_issues)

    return CharacterValidation(
        valid=valid,
        issues=all_issues,
        pilot_valid=pilot_validation.valid,
        pilot_issues=pilot_issues,
        mech_validations=mech_validations,
    )
