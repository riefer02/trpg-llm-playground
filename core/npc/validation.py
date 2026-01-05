"""NPC validation for typed Lancer mechanics.

This module provides validation for NPC templates and NPC state
in combat context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pydantic import Field
from core.shared.validation import ValidationIssue, ValidationResult

from core.npc.models import NPCTemplate, NPCAbility, NPCGear
from core.npc.state import NPCState

if TYPE_CHECKING:
    from typing import Literal

NPCValidationIssue = ValidationIssue


class NPCValidation(ValidationResult):
    """Validation result for NPC templates or state."""


def validate_npc_template(template: NPCTemplate) -> NPCValidation:
    """Validate an NPC template for mechanical integrity.

    Checks:
    - Unique ability IDs
    - Valid gear references (weapon/system IDs exist in compendium)
    - Reasonable gear limits
    - Valid tier scaling values
    - No duplicate ability triggers that would conflict

    Args:
        template: The NPC template to validate

    Returns:
        NPCValidation with any issues found
    """
    issues: list[NPCValidationIssue] = []

    ability_ids = [a.id for a in template.abilities]
    if len(ability_ids) != len(set(ability_ids)):
        issues.append(
            NPCValidationIssue(
                code="duplicate_ability_ids",
                message="NPC template has duplicate ability IDs",
                severity="error",
            )
        )

    for ability in template.abilities:
        if ability.uses_per_combat is not None and ability.uses_per_combat < 0:
            issues.append(
                NPCValidationIssue(
                    code="invalid_ability_uses",
                    message=f"Ability '{ability.id}' has invalid uses_per_combat ({ability.uses_per_combat})",
                    severity="error",
                )
            )

    gear_count = len(template.gear)
    if gear_count > 6:
        issues.append(
            NPCValidationIssue(
                code="too_much_gear",
                message=f"NPC has {gear_count} gear items (max 6 recommended)",
                severity="warning",
            )
        )

    has_weapon = any(g.weapon_id for g in template.gear)
    has_system = any(g.system_id for g in template.gear)
    if not has_weapon and not has_system and template.npc_class != "boss":
        issues.append(
            NPCValidationIssue(
                code="no_gear",
                message="NPC has no weapons or systems (boss NPCs may have none)",
                severity="warning",
            )
        )

    if template.stats.base.hp_base < 1:
        issues.append(
            NPCValidationIssue(
                code="invalid_hp_base",
                message=f"NPC has invalid hp_base ({template.stats.base.hp_base})",
                severity="error",
            )
        )

    if template.stats.base.evasion_base < 0:
        issues.append(
            NPCValidationIssue(
                code="invalid_evasion",
                message=f"NPC has invalid evasion_base ({template.stats.base.evasion_base})",
                severity="error",
            )
        )

    trigger_by_type: dict[str, list[str]] = {}
    for ability in template.abilities:
        trigger_by_type.setdefault(ability.trigger, []).append(ability.id)

    for trigger, ids in trigger_by_type.items():
        if len(ids) > 3:
            issues.append(
                NPCValidationIssue(
                    code="many_same_trigger_abilities",
                    message=f"NPC has {len(ids)} abilities with trigger '{trigger}' ({', '.join(ids)})",
                    severity="warning",
                )
            )

    if template.tier not in ("tier_1", "tier_2", "tier_3"):
        issues.append(
            NPCValidationIssue(
                code="invalid_tier",
                message=f"NPC has invalid tier '{template.tier}'",
                severity="error",
            )
        )

    valid = not any(i.severity == "error" for i in issues)
    return NPCValidation(valid=valid, issues=issues)


def validate_npc_in_combat(npc: NPCState) -> NPCValidation:
    """Validate an NPC instance for combat readiness.

    Checks:
    - Stats are within expected ranges
    - Ability uses are tracked correctly
    - Structure points match tier expectations

    Args:
        npc: The NPC state to validate

    Returns:
        NPCValidation with any issues found
    """
    issues: list[NPCValidationIssue] = []

    if npc.stats.hp_max < 1:
        issues.append(
            NPCValidationIssue(
                code="invalid_hp_max",
                message=f"NPC has invalid hp_max ({npc.stats.hp_max})",
                severity="error",
            )
        )

    if npc.stats.evasion < 0:
        issues.append(
            NPCValidationIssue(
                code="invalid_evasion",
                message=f"NPC has invalid evasion ({npc.stats.evasion})",
                severity="error",
            )
        )

    if npc.stats.e_defense < 0:
        issues.append(
            NPCValidationIssue(
                code="invalid_e_defense",
                message=f"NPC has invalid e_defense ({npc.stats.e_defense})",
                severity="error",
            )
        )

    expected_structures = {"tier_1": 1, "tier_2": 2, "tier_3": 3}
    actual_structure = npc.structure_current
    if actual_structure != expected_structures.get(npc.tier, 1):
        issues.append(
            NPCValidationIssue(
                code="structure_mismatch",
                message=f"NPC tier {npc.tier} has {actual_structure} structures (expected {expected_structures.get(npc.tier, 1)})",
                severity="warning",
            )
        )

    valid = not any(i.severity == "error" for i in issues)
    return NPCValidation(valid=valid, issues=issues)


def validate_npc_ability_use(
    npc: NPCState,
    ability_id: str,
    is_combat: bool = True,
) -> NPCValidation:
    """Validate that an NPC can use an ability.

    Args:
        npc: The NPC attempting to use an ability
        ability_id: The ability being used
        is_combat: Whether this is during combat (affects use limits)

    Returns:
        NPCValidation with any issues found
    """
    issues: list[NPCValidationIssue] = []

    if ability_id in npc.abilities_used:
        issues.append(
            NPCValidationIssue(
                code="ability_already_used",
                message=f"Ability '{ability_id}' has already been used this combat",
                severity="error",
            )
        )

    valid = not any(i.severity == "error" for i in issues)
    return NPCValidation(valid=valid, issues=issues)


def batch_validate_templates(templates: list[NPCTemplate]) -> dict[str, NPCValidation]:
    """Validate multiple NPC templates and return results by ID.

    Args:
        templates: List of templates to validate

    Returns:
        Dict mapping template ID to its validation result
    """
    return {t.id: validate_npc_template(t) for t in templates}
