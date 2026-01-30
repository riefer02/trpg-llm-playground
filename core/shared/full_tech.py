"""Full Tech action resolution primitives for Lancer TTRPG.

Implements resolution logic for the Full Tech full action per PR2 4260-4263:
- Choose and perform two options from the Quick Tech list
- Can repeat the same option twice
- Uses a system or tech option that takes a Full Tech action to activate

Resolution Pattern:
1. resolve_full_tech() - Pure resolution logic, returns what SHOULD happen
2. apply_full_tech_result() - Applies result to combatant state
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType
from core.shared.id_helpers import CombatantIdField

from core.mech.tech_actions import (
    ScanInfoType,
    ScanResult,
    BolsterResult,
    LockOnResult,
    InvadeResult,
    resolve_scan,
    resolve_bolster,
    resolve_lock_on,
    resolve_invade,
)
from core.mech.combat_state import CombatantState
from core.mech.combat_resolution import ResolutionSettings


FullTechOption = Literal["scan", "bolster", "lock_on", "invade"]


class FullTechOptionSelection(FrozenModel):
    """Minimal selection for a Full Tech option (option + target)."""

    option: FullTechOption
    target_id: CombatantIdField
    scan_options: list[ScanInfoType] | None = None


class ScanTechParams(FrozenModel):
    """Parameters for Scan action in Full Tech."""

    target_id: CombatantIdField
    scan_options: list[Literal["stats", "hidden_info", "public_info"]] = Field(
        default_factory=lambda: ["stats", "hidden_info", "public_info"]
    )


class BolsterTechParams(FrozenModel):
    """Parameters for Bolster action in Full Tech."""

    target_id: CombatantIdField
    attacker_systems: int = Field(..., ge=0)
    accuracy_bonus: int = Field(default=2, ge=0)


class LockOnTechParams(FrozenModel):
    """Parameters for Lock On action in Full Tech."""

    target_id: CombatantIdField
    accuracy_bonus: int = Field(default=1, ge=0)


class InvadeTechParams(FrozenModel):
    """Parameters for Invade action in Full Tech."""

    target_id: CombatantIdField
    tech_attack_bonus: int = Field(..., description="The actor's tech attack bonus")
    target_e_defense: int = Field(..., ge=0)
    heat_on_hit: int = Field(default=2, ge=0)


class FullTechFirstOption(FrozenModel):
    """First tech option for Full Tech action."""

    option: FullTechOption
    scan_params: ScanTechParams | None = None
    bolster_params: BolsterTechParams | None = None
    lock_on_params: LockOnTechParams | None = None
    invade_params: InvadeTechParams | None = None


class FullTechSecondOption(FrozenModel):
    """Second tech option for Full Tech action."""

    option: FullTechOption
    scan_params: ScanTechParams | None = None
    bolster_params: BolsterTechParams | None = None
    lock_on_params: LockOnTechParams | None = None
    invade_params: InvadeTechParams | None = None


class FullTechInput(FrozenModel):
    """Input for Full Tech action resolution.

    Represents the two tech options chosen when performing a Full Tech action.
    """

    actor_id: str = Field(..., description="ID of the actor performing Full Tech")
    first_option: FullTechFirstOption = Field(
        ..., description="First tech option (Quick Tech action)"
    )
    second_option: FullTechSecondOption = Field(
        ..., description="Second tech option (Quick Tech action, can repeat)"
    )
    settings: ResolutionSettings | None = Field(
        default=None, description="Optional resolution settings for forced rolls"
    )


class FullTechResolutionResult(FrozenModel):
    """Complete result of Full Tech resolution (pure logic).

    Provides detailed breakdown of what should happen during Full Tech action.
    """

    actor_id: str = Field(..., description="ID of the actor")
    first_option: FullTechOption = Field(..., description="First tech option taken")
    first_result: ScanResult | BolsterResult | LockOnResult | InvadeResult | None = (
        Field(default=None, description="Result of first tech action")
    )
    second_option: FullTechOption = Field(..., description="Second tech option taken")
    second_result: ScanResult | BolsterResult | LockOnResult | InvadeResult | None = (
        Field(default=None, description="Result of second tech action")
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )
    is_valid: bool = Field(
        default=True, description="Whether the action is valid and can be performed"
    )


class FullTechApplicationResult(FrozenModel):
    """Result of applying Full Tech result to combatant state.

    Returns the updated combatant with all tech effects applied.
    """

    updated_combatant: CombatantState = Field(
        ..., description="Combatant with Full Tech result applied"
    )
    first_tech_applied: bool = Field(
        default=False, description="Whether first tech action was successfully applied"
    )
    second_tech_applied: bool = Field(
        default=False, description="Whether second tech action was successfully applied"
    )
    targets_affected: list[str] = Field(
        default_factory=list, description="Target IDs affected by this Full Tech"
    )
    conditions_applied: list[StatusType] = Field(
        default_factory=list, description="Conditions applied to targets"
    )
    heat_dealt: int = Field(default=0, ge=0, description="Total heat dealt to targets")


def _resolve_single_tech(
    actor_id: str,
    option: FullTechOption,
    settings: ResolutionSettings | None,
) -> ScanResult | BolsterResult | LockOnResult | InvadeResult | None:
    """Resolve a single Quick Tech action based on option type.

    Args:
        actor_id: ID of the actor performing the tech action
        option: The tech action type
        settings: Optional resolution settings for forced rolls

    Returns:
        The resolution result for the tech action
    """
    return None


def _resolve_first_tech(
    actor_id: str,
    option: FullTechFirstOption,
    settings: ResolutionSettings | None,
) -> ScanResult | BolsterResult | LockOnResult | InvadeResult | None:
    """Resolve the first tech option of Full Tech action.

    Args:
        actor_id: ID of the actor performing the tech action
        option: The first tech option with its parameters
        settings: Optional resolution settings for forced rolls

    Returns:
        The resolution result for the first tech action
    """
    if option.option == "scan":
        if not option.scan_params:
            return None
        return resolve_scan(
            actor_id=actor_id,
            target_id=option.scan_params.target_id,
            scan_options=option.scan_params.scan_options,
        )
    elif option.option == "bolster":
        if not option.bolster_params:
            return None
        return resolve_bolster(
            actor_id=actor_id,
            target_id=option.bolster_params.target_id,
            attacker_systems=option.bolster_params.attacker_systems,
            accuracy_bonus=option.bolster_params.accuracy_bonus,
            settings=settings,
        )
    elif option.option == "lock_on":
        if not option.lock_on_params:
            return None
        return resolve_lock_on(
            actor_id=actor_id,
            target_id=option.lock_on_params.target_id,
            accuracy_bonus=option.lock_on_params.accuracy_bonus,
        )
    elif option.option == "invade":
        if not option.invade_params:
            return None
        return resolve_invade(
            actor_id=actor_id,
            target_id=option.invade_params.target_id,
            tech_attack_bonus=option.invade_params.tech_attack_bonus,
            target_e_defense=option.invade_params.target_e_defense,
            heat_on_hit=option.invade_params.heat_on_hit,
            settings=settings,
        )
    return None


def _resolve_second_tech(
    actor_id: str,
    option: FullTechSecondOption,
    settings: ResolutionSettings | None,
) -> ScanResult | BolsterResult | LockOnResult | InvadeResult | None:
    """Resolve the second tech option of Full Tech action.

    Args:
        actor_id: ID of the actor performing the tech action
        option: The second tech option with its parameters
        settings: Optional resolution settings for forced rolls

    Returns:
        The resolution result for the second tech action
    """
    if option.option == "scan":
        if not option.scan_params:
            return None
        return resolve_scan(
            actor_id=actor_id,
            target_id=option.scan_params.target_id,
            scan_options=option.scan_params.scan_options,
        )
    elif option.option == "bolster":
        if not option.bolster_params:
            return None
        return resolve_bolster(
            actor_id=actor_id,
            target_id=option.bolster_params.target_id,
            attacker_systems=option.bolster_params.attacker_systems,
            accuracy_bonus=option.bolster_params.accuracy_bonus,
            settings=settings,
        )
    elif option.option == "lock_on":
        if not option.lock_on_params:
            return None
        return resolve_lock_on(
            actor_id=actor_id,
            target_id=option.lock_on_params.target_id,
            accuracy_bonus=option.lock_on_params.accuracy_bonus,
        )
    elif option.option == "invade":
        if not option.invade_params:
            return None
        return resolve_invade(
            actor_id=actor_id,
            target_id=option.invade_params.target_id,
            tech_attack_bonus=option.invade_params.tech_attack_bonus,
            target_e_defense=option.invade_params.target_e_defense,
            heat_on_hit=option.invade_params.heat_on_hit,
            settings=settings,
        )
    return None


def _validate_tech_params(
    option: FullTechFirstOption | FullTechSecondOption,
) -> str | None:
    """Validate that required parameters are present for the tech option.

    Args:
        option: The tech option with its parameters

    Returns:
        Error message if invalid, None if valid
    """
    if option.option == "scan":
        if not option.scan_params:
            return "scan_params required for scan option"
    elif option.option == "bolster":
        if not option.bolster_params:
            return "bolster_params required for bolster option"
    elif option.option == "lock_on":
        if not option.lock_on_params:
            return "lock_on_params required for lock_on option"
    elif option.option == "invade":
        if not option.invade_params:
            return "invade_params required for invade option"
    return None


def resolve_full_tech(input: FullTechInput) -> FullTechResolutionResult:
    """Resolve a Full Tech action per PR2 4260-4263.

    Full Tech is a Full Action that allows the pilot to:
    - Perform two Quick Tech actions in sequence
    - Options can be repeated (e.g., Lock On + Lock On)

    Args:
        input: Full Tech input with two tech options

    Returns:
        Detailed breakdown of what should happen during Full Tech
    """
    errors: list[str] = []

    first_error = _validate_tech_params(input.first_option)
    if first_error:
        errors.append(f"First option: {first_error}")

    second_error = _validate_tech_params(input.second_option)
    if second_error:
        errors.append(f"Second option: {second_error}")

    is_valid = len(errors) == 0

    first_result = None
    second_result = None

    if is_valid:
        first_result = _resolve_first_tech(
            input.actor_id, input.first_option, input.settings
        )
        second_result = _resolve_second_tech(
            input.actor_id, input.second_option, input.settings
        )

    return FullTechResolutionResult(
        actor_id=input.actor_id,
        first_option=input.first_option.option,
        first_result=first_result,
        second_option=input.second_option.option,
        second_result=second_result,
        validation_errors=errors,
        is_valid=is_valid,
    )


def _extract_target_from_result(
    result: ScanResult | BolsterResult | LockOnResult | InvadeResult | None,
) -> str | None:
    """Extract target ID from a tech action result.

    Args:
        result: The tech action result

    Returns:
        Target ID if available, None otherwise
    """
    if result is None:
        return None
    return getattr(result, "target_id", None)


def _extract_conditions_applied(
    result: ScanResult | BolsterResult | LockOnResult | InvadeResult | None,
) -> list[StatusType]:
    """Extract conditions applied from a tech action result.

    Args:
        result: The tech action result

    Returns:
        List of conditions applied
    """
    if result is None:
        return []
    if isinstance(result, InvadeResult) and result.conditions_applied:
        return result.conditions_applied
    return []


def _calculate_heat_dealt(
    result: ScanResult | BolsterResult | LockOnResult | InvadeResult | None,
) -> int:
    """Calculate heat dealt from a tech action result.

    Args:
        result: The tech action result

    Returns:
        Heat dealt to targets
    """
    if result is None:
        return 0
    if isinstance(result, InvadeResult) and result.heat_applied is not None:
        return result.heat_applied
    return 0


def apply_full_tech_result(
    combatant: CombatantState,
    result: FullTechResolutionResult,
) -> FullTechApplicationResult:
    """Apply Full Tech result to combatant state.

    Updates combatant with all tech effects applied to targets.

    Args:
        combatant: Current combatant state
        result: Resolution result to apply

    Returns:
        Updated combatant with Full Tech effects applied
    """
    if (
        not result.is_valid
        or result.first_result is None
        or result.second_result is None
    ):
        return FullTechApplicationResult(
            updated_combatant=combatant,
            first_tech_applied=False,
            second_tech_applied=False,
        )

    targets_affected: list[str] = []
    conditions_applied: list[StatusType] = []
    heat_dealt = 0

    first_target = _extract_target_from_result(result.first_result)
    if first_target:
        targets_affected.append(first_target)

    second_target = _extract_target_from_result(result.second_result)
    if second_target:
        targets_affected.append(second_target)

    conditions_applied.extend(_extract_conditions_applied(result.first_result))
    conditions_applied.extend(_extract_conditions_applied(result.second_result))

    heat_dealt = _calculate_heat_dealt(result.first_result)
    heat_dealt += _calculate_heat_dealt(result.second_result)

    return FullTechApplicationResult(
        updated_combatant=combatant,
        first_tech_applied=result.first_result is not None,
        second_tech_applied=result.second_result is not None,
        targets_affected=list(set(targets_affected)),
        conditions_applied=conditions_applied,
        heat_dealt=heat_dealt,
    )


try:
    from core.shared.heat import MeltdownState
    from core.mech.combat_state import CombatantState

    CombatantState.model_rebuild(
        _types_namespace={"MeltdownState": MeltdownState, "StabilizeState": type(None)}
    )
except ImportError:
    pass  # CombatantState or MeltdownState not yet available during initial import
