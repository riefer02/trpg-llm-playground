"""Pending decision models and resolution for Lancer combat.

This module provides models and functions for surfacing player decisions
during combat resolution, following the pattern established for overwatch
and reaction opportunities.

Decision Types:
- hull_save: Hull save required (structure cascade at 2 structure)
- engineering_save: Engineering save required (meltdown cascade)
- engineering_check: Engineering check required (dangerous terrain)
- system_trauma: Player chooses mount OR system to destroy

PR2 Rules References:
- Save Target = 10 + Attacker's Grit + bonuses
- Players can voluntarily fail any save (PR2 line 1370)
- System Trauma: Player chooses mount OR system to destroy (PR2 4618-4622)
- Limited systems with 0 charges are not valid trauma targets
"""

from __future__ import annotations

import uuid
from typing import Literal, TYPE_CHECKING
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import SaveType, StatusType
from core.shared.saves import SaveRequest, SaveResult, resolve_save
from core.shared.id_helpers import CombatantIdField

if TYPE_CHECKING:
    from core.mech.combat_state import (
        CombatantState,
        MechInventory,
        MechCombatScenario,
    )
    from core.shared.structure import StructureResolutionResult, SystemTraumaSelection
    from core.shared.heat import OverheatResolutionResult


# Decision type literals
DecisionType = Literal[
    "hull_save",  # Hull save for structure cascade at 2 structure
    "engineering_save",  # Engineering save for meltdown cascade
    "engineering_check",  # Engineering check for dangerous terrain
    "system_trauma",  # Player chooses mount or system to destroy
]

DecisionChoice = Literal[
    "roll",  # Make the save/check roll
    "voluntary_fail",  # Voluntarily fail the save (PR2 1370)
    "use_reroll",  # Use available reroll from talent, then roll
]


class PendingDecision(FrozenModel):
    """A pending decision that requires player input.

    This model captures all context needed to present a decision prompt
    to the player and resolve their choice.
    """

    decision_id: str = Field(
        ..., description="Unique identifier for this decision"
    )
    decision_type: DecisionType = Field(
        ..., description="Type of decision required"
    )
    combatant_id: CombatantIdField = Field(
        ..., description="ID of combatant who must make the decision"
    )
    trigger_source: str = Field(
        ..., description="What triggered this decision (e.g., 'structure_cascade', 'meltdown', 'dangerous_terrain')"
    )
    trigger_round: int = Field(
        ..., ge=1, description="Round number when this decision was created"
    )

    # Save-specific fields
    save_type: SaveType | None = Field(
        default=None, description="Type of save required (hull, engineering, etc.)"
    )
    save_target: int | None = Field(
        default=None, ge=0, description="Save target DC"
    )
    save_bonus: int = Field(
        default=0, description="Bonus to save roll from skills"
    )

    # Trauma-specific fields
    trauma_target: Literal["mount", "system"] | None = Field(
        default=None, description="Resolved trauma target category (mount/system)"
    )
    eligible_mounts: list[int] = Field(
        default_factory=list, description="Mount indices eligible for destruction"
    )
    eligible_systems: list[str] = Field(
        default_factory=list, description="System IDs eligible for destruction"
    )

    # Reroll availability (from talents like Exemplar)
    reroll_available: bool = Field(
        default=False, description="Whether a reroll is available from talents"
    )
    reroll_source: str | None = Field(
        default=None, description="Source of reroll if available"
    )


class DecisionResolution(FrozenModel):
    """Player's resolution of a pending decision.

    Contains the player's choice and any additional parameters
    needed to resolve the decision.
    """

    choice: DecisionChoice = Field(
        ..., description="Player's chosen action"
    )
    selected_mount_index: int | None = Field(
        default=None, ge=0, description="Mount index selected for system trauma"
    )
    selected_system_id: str | None = Field(
        default=None, description="System ID selected for system trauma"
    )
    used_reroll: bool = Field(
        default=False, description="Whether the player used their reroll"
    )


class SaveDecisionResult(FrozenModel):
    """Result of resolving a save decision.

    Contains the save result and whether the save was voluntarily failed.
    """

    decision_id: str = Field(..., description="ID of the resolved decision")
    save_result: SaveResult | None = Field(
        default=None, description="Save roll result if roll was made"
    )
    voluntarily_failed: bool = Field(
        default=False, description="Whether the save was voluntarily failed"
    )
    reroll_used: bool = Field(
        default=False, description="Whether a reroll was used"
    )
    success: bool = Field(
        ..., description="Whether the save/check succeeded"
    )


class TraumaDecisionResult(FrozenModel):
    """Result of resolving a system trauma decision.

    Contains the selected target and validation status.
    """

    decision_id: str = Field(..., description="ID of the resolved decision")
    selected_target: Literal["mount", "system"] = Field(
        ..., description="What type of target was selected"
    )
    mount_index: int | None = Field(
        default=None, description="Mount index if mount was selected"
    )
    system_id: str | None = Field(
        default=None, description="System ID if system was selected"
    )
    valid_selection: bool = Field(
        ..., description="Whether the selection was valid"
    )
    error_message: str | None = Field(
        default=None, description="Error message if selection was invalid"
    )


def generate_decision_id() -> str:
    """Generate a unique decision ID."""
    return f"decision_{uuid.uuid4().hex[:12]}"


def create_hull_save_decision(
    combatant_id: str,
    trigger_round: int,
    save_target: int = 10,
    save_bonus: int = 0,
    reroll_available: bool = False,
    reroll_source: str | None = None,
) -> PendingDecision:
    """Create a pending hull save decision for structure cascade at 2 structure.

    Per PR2: Direct hit at 2 structure requires a hull check DC 10 to avoid destruction.
    Players can voluntarily fail this save (PR2 1370).

    Args:
        combatant_id: ID of combatant making the save
        trigger_round: Current round number
        save_target: Save DC (default 10)
        save_bonus: Bonus from hull skill
        reroll_available: Whether reroll is available from talents
        reroll_source: Source of reroll if available

    Returns:
        PendingDecision for hull save
    """
    return PendingDecision(
        decision_id=generate_decision_id(),
        decision_type="hull_save",
        combatant_id=combatant_id,
        trigger_source="structure_cascade",
        trigger_round=trigger_round,
        save_type="hull",
        save_target=save_target,
        save_bonus=save_bonus,
        reroll_available=reroll_available,
        reroll_source=reroll_source,
    )


def create_engineering_save_decision(
    combatant_id: str,
    trigger_round: int,
    save_target: int = 10,
    save_bonus: int = 0,
    trigger_source: str = "meltdown",
    reroll_available: bool = False,
    reroll_source: str | None = None,
) -> PendingDecision:
    """Create a pending engineering save decision for meltdown cascade.

    Per PR2: Meltdown at 2 stress requires an engineering check DC 10.
    Players can voluntarily fail this save (PR2 1370).

    Args:
        combatant_id: ID of combatant making the save
        trigger_round: Current round number
        save_target: Save DC (default 10)
        save_bonus: Bonus from engineering skill
        trigger_source: What triggered the save
        reroll_available: Whether reroll is available from talents
        reroll_source: Source of reroll if available

    Returns:
        PendingDecision for engineering save
    """
    return PendingDecision(
        decision_id=generate_decision_id(),
        decision_type="engineering_save",
        combatant_id=combatant_id,
        trigger_source=trigger_source,
        trigger_round=trigger_round,
        save_type="engineering",
        save_target=save_target,
        save_bonus=save_bonus,
        reroll_available=reroll_available,
        reroll_source=reroll_source,
    )


def create_engineering_check_decision(
    combatant_id: str,
    trigger_round: int,
    save_target: int = 10,
    save_bonus: int = 0,
    trigger_source: str = "dangerous_terrain",
    reroll_available: bool = False,
    reroll_source: str | None = None,
) -> PendingDecision:
    """Create a pending engineering check decision for dangerous terrain.

    Per PR2: Moving through dangerous terrain requires an engineering check
    (DC varies by terrain type). Players can voluntarily fail this check.

    Args:
        combatant_id: ID of combatant making the check
        trigger_round: Current round number
        save_target: Check DC (default 10)
        save_bonus: Bonus from engineering skill
        trigger_source: What triggered the check
        reroll_available: Whether reroll is available from talents
        reroll_source: Source of reroll if available

    Returns:
        PendingDecision for engineering check
    """
    return PendingDecision(
        decision_id=generate_decision_id(),
        decision_type="engineering_check",
        combatant_id=combatant_id,
        trigger_source=trigger_source,
        trigger_round=trigger_round,
        save_type="engineering",
        save_target=save_target,
        save_bonus=save_bonus,
        reroll_available=reroll_available,
        reroll_source=reroll_source,
    )


def create_system_trauma_decision(
    combatant_id: str,
    trigger_round: int,
    trauma_target: Literal["mount", "system"],
    eligible_mounts: list[int],
    eligible_systems: list[str],
) -> PendingDecision:
    """Create a pending system trauma decision.

    Per PR2 4618-4622: On system trauma result, player chooses what to destroy.
    Systems/weapons with limited tag and 0 charges are not valid targets.

    Args:
        combatant_id: ID of combatant making the decision
        trigger_round: Current round number
        eligible_mounts: Mount indices eligible for destruction
        eligible_systems: System IDs eligible for destruction

    Returns:
        PendingDecision for system trauma selection
    """
    return PendingDecision(
        decision_id=generate_decision_id(),
        decision_type="system_trauma",
        combatant_id=combatant_id,
        trigger_source="system_trauma",
        trigger_round=trigger_round,
        trauma_target=trauma_target,
        eligible_mounts=eligible_mounts,
        eligible_systems=eligible_systems,
    )


def check_structure_decisions(
    combatant: "CombatantState",
    structure_result: "StructureResolutionResult",
    current_round: int,
) -> list[PendingDecision]:
    """Check if structure damage result requires player decisions.

    Detects:
    - Hull save required (direct hit at 2 structure)
    - System trauma selection required

    Args:
        combatant: Combatant who took structure damage
        structure_result: Result of structure damage resolution
        current_round: Current round number

    Returns:
        List of PendingDecision objects (may be empty)
    """
    decisions: list[PendingDecision] = []

    # Check for hull save requirement (direct hit at 2 structure)
    if structure_result.hull_check_request is not None:
        hull_skill = combatant.stats.grit  # Hull bonus from grit
        decisions.append(
            create_hull_save_decision(
                combatant_id=combatant.id,
                trigger_round=current_round,
                save_target=structure_result.hull_check_request.save_target,
                save_bonus=hull_skill,
            )
        )

    # Check for system trauma selection
    if (
        structure_result.system_trauma is not None
        and structure_result.system_trauma.resolved_target != "direct_hit"
    ):
        decisions.append(
            create_system_trauma_decision(
                combatant_id=combatant.id,
                trigger_round=current_round,
                trauma_target=structure_result.system_trauma.resolved_target,
                eligible_mounts=structure_result.system_trauma.eligible_mounts,
                eligible_systems=structure_result.system_trauma.eligible_systems,
            )
        )

    return decisions


def check_overheat_decisions(
    combatant: "CombatantState",
    overheat_result: "OverheatResolutionResult",
    current_round: int,
) -> list[PendingDecision]:
    """Check if overheat result requires player decisions.

    Detects:
    - Engineering save required (meltdown at 2 stress)

    Args:
        combatant: Combatant who overheated
        overheat_result: Result of overheat resolution
        current_round: Current round number

    Returns:
        List of PendingDecision objects (may be empty)
    """
    decisions: list[PendingDecision] = []

    # Check for engineering save requirement (meltdown at 2 stress)
    if overheat_result.engineering_check_request is not None:
        eng_skill = combatant.stats.engineering_skill
        decisions.append(
            create_engineering_save_decision(
                combatant_id=combatant.id,
                trigger_round=current_round,
                save_target=overheat_result.engineering_check_request.save_target,
                save_bonus=eng_skill,
            )
        )

    return decisions


def check_dangerous_terrain_decision(
    combatant: "CombatantState",
    terrain_name: str,
    check_target: int,
    current_round: int,
) -> PendingDecision:
    """Create a decision for dangerous terrain entry.

    Per PR2: Moving through dangerous terrain requires an engineering check.
    Players can voluntarily fail this check.

    Args:
        combatant: Combatant entering dangerous terrain
        terrain_name: Name of the terrain type
        check_target: DC for the engineering check
        current_round: Current round number

    Returns:
        PendingDecision for engineering check
    """
    eng_skill = combatant.stats.engineering_skill
    return create_engineering_check_decision(
        combatant_id=combatant.id,
        trigger_round=current_round,
        save_target=check_target,
        save_bonus=eng_skill,
        trigger_source=f"dangerous_terrain:{terrain_name}",
    )


def resolve_save_decision(
    decision: PendingDecision,
    resolution: DecisionResolution,
    target_conditions: list[StatusType] | None = None,
    force_roll: int | None = None,
) -> SaveDecisionResult:
    """Resolve a save decision based on player's choice.

    Handles:
    - Rolling the save normally
    - Voluntarily failing the save
    - Using a reroll before rolling

    Args:
        decision: The pending decision to resolve
        resolution: Player's resolution choice
        target_conditions: Current conditions on the combatant
        force_roll: Optional forced roll for testing

    Returns:
        SaveDecisionResult with resolution details
    """
    if decision.decision_type not in ("hull_save", "engineering_save", "engineering_check"):
        raise ValueError(f"Cannot resolve save for decision type: {decision.decision_type}")

    # Handle voluntary failure
    if resolution.choice == "voluntary_fail":
        return SaveDecisionResult(
            decision_id=decision.decision_id,
            save_result=None,
            voluntarily_failed=True,
            reroll_used=False,
            success=False,
        )

    # Handle roll (with or without reroll)
    if decision.save_type is None or decision.save_target is None:
        raise ValueError("Save decision missing save_type or save_target")

    save_request = SaveRequest(
        save_type=decision.save_type,
        save_target=decision.save_target,
        save_bonus=decision.save_bonus,
        target_conditions=target_conditions or [],
        force_roll=force_roll,
    )

    save_result = resolve_save(save_request)

    # Handle reroll if requested and available
    reroll_used = False
    if resolution.choice == "use_reroll" and decision.reroll_available:
        reroll_used = True
        # Roll again and keep better result
        second_request = SaveRequest(
            save_type=decision.save_type,
            save_target=decision.save_target,
            save_bonus=decision.save_bonus,
            target_conditions=target_conditions or [],
            # Don't force roll on reroll - use random
        )
        second_result = resolve_save(second_request)
        # Keep the better result (success wins, or higher total)
        if second_result.success and not save_result.success:
            save_result = second_result
        elif second_result.total > save_result.total:
            save_result = second_result

    return SaveDecisionResult(
        decision_id=decision.decision_id,
        save_result=save_result,
        voluntarily_failed=False,
        reroll_used=reroll_used,
        success=save_result.success,
    )


def resolve_trauma_decision(
    decision: PendingDecision,
    resolution: DecisionResolution,
) -> TraumaDecisionResult:
    """Resolve a system trauma decision based on player's choice.

    Validates that the selected target is eligible for destruction.

    Args:
        decision: The pending decision to resolve
        resolution: Player's resolution choice

    Returns:
        TraumaDecisionResult with selection details
    """
    if decision.decision_type != "system_trauma":
        raise ValueError(f"Cannot resolve trauma for decision type: {decision.decision_type}")

    # Check if mount was selected
    if resolution.selected_mount_index is not None:
        if resolution.selected_mount_index not in decision.eligible_mounts:
            return TraumaDecisionResult(
                decision_id=decision.decision_id,
                selected_target="mount",
                mount_index=resolution.selected_mount_index,
                system_id=None,
                valid_selection=False,
                error_message=f"Mount {resolution.selected_mount_index} is not eligible for destruction",
            )
        return TraumaDecisionResult(
            decision_id=decision.decision_id,
            selected_target="mount",
            mount_index=resolution.selected_mount_index,
            system_id=None,
            valid_selection=True,
            error_message=None,
        )

    # Check if system was selected
    if resolution.selected_system_id is not None:
        if resolution.selected_system_id not in decision.eligible_systems:
            return TraumaDecisionResult(
                decision_id=decision.decision_id,
                selected_target="system",
                mount_index=None,
                system_id=resolution.selected_system_id,
                valid_selection=False,
                error_message=f"System {resolution.selected_system_id} is not eligible for destruction",
            )
        return TraumaDecisionResult(
            decision_id=decision.decision_id,
            selected_target="system",
            mount_index=None,
            system_id=resolution.selected_system_id,
            valid_selection=True,
            error_message=None,
        )

    # No valid selection made
    return TraumaDecisionResult(
        decision_id=decision.decision_id,
        selected_target="mount",  # Default
        mount_index=None,
        system_id=None,
        valid_selection=False,
        error_message="No mount or system selected for trauma",
    )


def apply_system_trauma_selection(
    combatant: "CombatantState",
    trauma_result: TraumaDecisionResult,
) -> "CombatantState":
    """Apply a validated system trauma selection to a combatant's inventory."""
    if not trauma_result.valid_selection:
        return combatant
    if combatant.inventory is None:
        return combatant

    from core.shared.state_helpers import destroy_mount, destroy_system

    if trauma_result.selected_target == "mount":
        if trauma_result.mount_index is None:
            return combatant
        updated_inventory = destroy_mount(
            combatant.inventory, trauma_result.mount_index
        )
    else:
        if trauma_result.system_id is None:
            return combatant
        updated_inventory = destroy_system(
            combatant.inventory, trauma_result.system_id
        )

    return combatant.model_copy(update={"inventory": updated_inventory})


def apply_failed_hull_save(combatant: "CombatantState") -> "CombatantState":
    """Apply a failed hull save from a direct hit by destroying the mech."""
    new_resources = combatant.resources.model_copy(
        update={"hp_current": 0, "structure_current": 0}
    )
    return combatant.model_copy(update={"resources": new_resources})


def get_pending_decisions_for_combatant(
    scenario: "MechCombatScenario",
    combatant_id: str,
) -> list[PendingDecision]:
    """Get all pending decisions for a specific combatant.

    Args:
        scenario: Current combat scenario
        combatant_id: ID of combatant to get decisions for

    Returns:
        List of pending decisions for the combatant
    """
    return [
        d for d in scenario.pending_decisions
        if d.combatant_id == combatant_id
    ]


def remove_decision_from_scenario(
    scenario: "MechCombatScenario",
    decision_id: str,
) -> "MechCombatScenario":
    """Remove a resolved decision from the scenario.

    Args:
        scenario: Current combat scenario
        decision_id: ID of decision to remove

    Returns:
        Updated scenario with decision removed
    """
    new_decisions = [
        d for d in scenario.pending_decisions
        if d.decision_id != decision_id
    ]
    return scenario.model_copy(update={"pending_decisions": new_decisions})


def add_decision_to_scenario(
    scenario: "MechCombatScenario",
    decision: PendingDecision,
) -> "MechCombatScenario":
    """Add a pending decision to the scenario.

    Args:
        scenario: Current combat scenario
        decision: Decision to add

    Returns:
        Updated scenario with decision added
    """
    new_decisions = list(scenario.pending_decisions) + [decision]
    return scenario.model_copy(update={"pending_decisions": new_decisions})


__all__ = [
    "DecisionType",
    "DecisionChoice",
    "PendingDecision",
    "DecisionResolution",
    "SaveDecisionResult",
    "TraumaDecisionResult",
    "generate_decision_id",
    "create_hull_save_decision",
    "create_engineering_save_decision",
    "create_engineering_check_decision",
    "create_system_trauma_decision",
    "check_structure_decisions",
    "check_overheat_decisions",
    "check_dangerous_terrain_decision",
    "resolve_save_decision",
    "resolve_trauma_decision",
    "apply_system_trauma_selection",
    "apply_failed_hull_save",
    "get_pending_decisions_for_combatant",
    "remove_decision_from_scenario",
    "add_decision_to_scenario",
]
