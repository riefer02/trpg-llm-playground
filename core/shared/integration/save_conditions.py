"""Save → Condition integration for Lancer combat.

This module provides automatic condition application on failed saves,
matching Lancer's "must pass X or become Y" pattern from the PR2 book.

Examples from the book:
- Stun Mine: "A character that fails this save is stunned" (line 18949)
- Sealant Mine: "A character that fails this save is immobilized" (line 18952)
- Omni-harpoon: "On a failed save, they are knocked prone" (line 7521)
- Various weapons: "must pass a HULL save or become stunned" (line 8306)

All condition applications are deterministic - no GM choice required.
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import SaveType, StatusType
from core.shared.saves import SaveRequest, SaveResult, resolve_save
from core.shared.conditions import apply_condition


class SaveConditionMapping(FrozenModel):
    """Maps save type + failure → automatic condition application.

    This captures the Lancer pattern of "must pass X or become Y" where
    conditions apply automatically on failed save, not as GM discretion.

    Args:
        save_type: Type of save this mapping applies to (HULL, AGILITY, etc.)
        condition: Condition to apply on failure
        applies_on: When to apply the condition
            - "failure": Apply on any failed save (regular or critical)
            - "critical_failure": Apply only on natural 1 (automatic fail)
    """

    save_type: SaveType
    condition: StatusType
    applies_on: Literal["failure", "critical_failure"] = "failure"


class SaveConditionResult(FrozenModel):
    """Outcome of save resolution with condition application.

    Attributes:
        save_result: The underlying save resolution result
        conditions_applied: List of conditions applied (may be empty)
        condition_results: Details of each condition application
    """

    save_result: SaveResult
    conditions_applied: list[StatusType] = Field(default_factory=list)
    condition_results: list[dict] = Field(default_factory=list)


HULL_SAVE_MAPPINGS: list[SaveConditionMapping] = [
    SaveConditionMapping(
        save_type="hull", condition="immobilized", applies_on="failure"
    ),
    SaveConditionMapping(
        save_type="hull", condition="stunned", applies_on="critical_failure"
    ),
]

AGILITY_SAVE_MAPPINGS: list[SaveConditionMapping] = [
    SaveConditionMapping(save_type="agility", condition="prone", applies_on="failure"),
    SaveConditionMapping(
        save_type="agility", condition="stunned", applies_on="critical_failure"
    ),
]

SYSTEMS_SAVE_MAPPINGS: list[SaveConditionMapping] = [
    SaveConditionMapping(save_type="systems", condition="jammed", applies_on="failure"),
    SaveConditionMapping(
        save_type="systems", condition="stunned", applies_on="critical_failure"
    ),
]

ENGINEERING_SAVE_MAPPINGS: list[SaveConditionMapping] = [
    SaveConditionMapping(
        save_type="engineering", condition="impaired", applies_on="failure"
    ),
    SaveConditionMapping(
        save_type="engineering", condition="stunned", applies_on="critical_failure"
    ),
]

COMMON_SAVE_CONDITION_MAPPINGS: list[SaveConditionMapping] = (
    HULL_SAVE_MAPPINGS
    + AGILITY_SAVE_MAPPINGS
    + SYSTEMS_SAVE_MAPPINGS
    + ENGINEERING_SAVE_MAPPINGS
)


def resolve_save_with_conditions(
    save_request: SaveRequest,
    mappings: list[SaveConditionMapping],
    target_conditions: list[StatusType],
) -> SaveConditionResult:
    """Execute save and auto-apply conditions per Lancer's deterministic pattern.

    This function resolves a save and automatically applies conditions based
    on the result, following the "must pass X or become Y" pattern from the
    Lancer rules. Condition application is deterministic - no GM choice.

    Args:
        save_request: The save to resolve
        mappings: Condition mappings to apply on failure
        target_conditions: Current conditions on the save target (for mutation)

    Returns:
        SaveConditionResult with save outcome and any applied conditions
    """
    save_result = resolve_save(save_request)

    conditions_applied: list[StatusType] = []
    condition_results: list[dict] = []

    for mapping in mappings:
        if mapping.save_type != save_request.save_type:
            continue

        should_apply = False

        if mapping.applies_on == "critical_failure":
            should_apply = save_result.degree == "critical_failure"
        else:
            should_apply = not save_result.success

        if should_apply:
            result = apply_condition(target_conditions, mapping.condition)
            conditions_applied.append(mapping.condition)
            condition_results.append(
                {
                    "condition": mapping.condition,
                    "applied": result.applied,
                    "reason": mapping.applies_on,
                }
            )

    return SaveConditionResult(
        save_result=save_result,
        conditions_applied=conditions_applied,
        condition_results=condition_results,
    )


def get_default_mappings_for_save_type(
    save_type: SaveType,
) -> list[SaveConditionMapping]:
    """Get default condition mappings for a save type.

    Args:
        save_type: The save type to get mappings for

    Returns:
        List of default mappings for this save type
    """
    mapping_map: dict[SaveType, list[SaveConditionMapping]] = {
        "hull": HULL_SAVE_MAPPINGS,
        "agility": AGILITY_SAVE_MAPPINGS,
        "systems": SYSTEMS_SAVE_MAPPINGS,
        "engineering": ENGINEERING_SAVE_MAPPINGS,
    }
    return mapping_map.get(save_type, [])
