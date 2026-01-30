"""AI/NHP control resolution primitives for Lancer TTRPG.

Implements AI control mechanics per PR2 5056-5161:

AI Property:
- Mech with AI system gains AI property
- Can install only 1 AI system (unless modified by effects)
- At start of turn: free action to cede control to AI

Unshackling (NHP only):
- Each structure/overheat check: roll d20
- On 1: NHP becomes Unshackled
- Unshackled NHP gains immediate control, controlled by GM
- Behavior: ignore, overrule, illogical, or remove_pilot

Re-shackling:
- Shut Down action re-shackles unshackled NHP

Control States:
- "pilot": Default, pilot in control
- "cede": Pilot ceded to AI, pilot in mech, auto-resume next turn
- "cede_remote": Pilot ceded and exited mech, must remount
- "unshackled": NHP took control, GM controls, Shut Down required

Resolution Pattern:
1. resolve_cede_control() / resolve_cede_control_remote() - Control cede
2. resolve_unshackle_check() - Unshackle trigger on structure/overheat
3. resolve_unshackled_behavior() - Select NHP behavior
4. apply_ai_control_result() - Apply control state changes
5. resolve_remount() - Remount from remote cede
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.dice import DiceExpression
from core.mech.combat_state import CombatantState


AIType = Literal["compcon", "nhp"]

AIControlState = Literal["pilot", "cede", "cede_remote", "unshackled"]

NHPBehaviorType = Literal["ignore_pilot", "overrule_pilot", "illogical", "remove_pilot"]

RemovePilotMode = Literal["eject_adjacent", "remove_completely"]


class NHPBehaviorConfig(FrozenModel):
    """Configuration for NHP behavior when unshackled."""

    behavior: NHPBehaviorType
    remove_pilot_mode: RemovePilotMode = "eject_adjacent"
    eject_target_space: int = 1


DEFAULT_NHP_BEHAVIORS: list[NHPBehaviorConfig] = [
    NHPBehaviorConfig(behavior="ignore_pilot"),
    NHPBehaviorConfig(behavior="overrule_pilot"),
    NHPBehaviorConfig(behavior="illogical"),
    NHPBehaviorConfig(behavior="remove_pilot", remove_pilot_mode="eject_adjacent"),
]


class AIRule(FrozenModel):
    """Rule configuration for AI control mechanics."""

    unshackle_threshold: int = 1
    unshackle_on_structure_check: bool = True
    unshackle_on_overheat_check: bool = True
    behavior_weights: dict[NHPBehaviorType, float] = Field(
        default_factory=lambda: {
            "ignore_pilot": 0.25,
            "overrule_pilot": 0.25,
            "illogical": 0.25,
            "remove_pilot": 0.25,
        }
    )
    cede_turns_duration: int = 1


DEFAULT_AI_RULES = AIRule()


class CedeControlInput(FrozenModel):
    """Input for ceding control to AI (pilot remains in mech)."""

    actor_id: str = Field(..., description="ID of the pilot ceding control")
    rules: AIRule | None = Field(default=None, description="Override rules")


class CedeControlRemoteInput(FrozenModel):
    """Input for ceding control to AI with remote exit (pilot exits mech)."""

    actor_id: str = Field(..., description="ID of the pilot ceding control remotely")
    rules: AIRule | None = Field(default=None, description="Override rules")


class UnshackleCheckInput(FrozenModel):
    """Input for unshackle check on structure or overheat check."""

    actor_id: str = Field(..., description="ID of the mech with NHP")
    check_type: Literal["structure", "overheat"] = Field(
        ..., description="Type of check that triggered unshackle opportunity"
    )
    rules: AIRule | None = Field(default=None, description="Override rules")
    force_roll: int | None = Field(
        default=None, description="Forced d20 roll value for testing"
    )


class UnshackledBehaviorInput(FrozenModel):
    """Input for selecting NHP behavior when unshackled."""

    actor_id: str = Field(..., description="ID of the unshackled mech")
    rules: AIRule | None = Field(default=None, description="Override rules")


class RemountInput(FrozenModel):
    """Input for remounting mech from remote cede state."""

    pilot_id: str = Field(..., description="ID of the pilot remounting")
    mech_id: str = Field(..., description="ID of the mech to remount")
    is_adjacent: bool = Field(
        default=True, description="Whether pilot is adjacent to mech"
    )


class CedeControlResult(FrozenModel):
    """Result of ceding control to AI (pilot remains in mech)."""

    actor_id: str = Field(..., description="ID of the pilot")
    control_ceded: bool = Field(default=True, description="Whether control was ceded")
    cede_turns: int = Field(default=1, description="Duration of cede in turns")
    validation_errors: list[str] = Field(default_factory=list)


class CedeControlRemoteResult(FrozenModel):
    """Result of ceding control to AI with remote exit."""

    actor_id: str = Field(..., description="ID of the pilot")
    control_ceded: bool = Field(default=True, description="Whether control was ceded")
    pilot_exited: bool = Field(default=True, description="Whether pilot exited mech")
    exit_position: str = Field(
        default="adjacent", description="Position of pilot after exit"
    )
    validation_errors: list[str] = Field(default_factory=list)


class UnshackleCheckResult(FrozenModel):
    """Result of unshackle check on structure/overheat."""

    actor_id: str = Field(..., description="ID of the mech")
    check_type: str = Field(..., description="Type of check performed")
    d20_roll: int = Field(..., description="d20 roll result")
    unshackle_occurred: bool = Field(
        default=False, description="Whether unshackling occurred"
    )
    nhps_affected: int = Field(default=0, description="Number of NHPs that unshackled")
    validation_errors: list[str] = Field(default_factory=list)


class UnshackledBehaviorResult(FrozenModel):
    """Result of selecting NHP behavior when unshackled."""

    actor_id: str = Field(..., description="ID of the unshackled mech")
    behavior: NHPBehaviorType = Field(..., description="Selected behavior")
    behavior_config: NHPBehaviorConfig = Field(
        ..., description="Full behavior configuration"
    )
    remove_pilot_mode: RemovePilotMode | None = Field(
        default=None, description="Mode for remove_pilot behavior"
    )


class RemountResult(FrozenModel):
    """Result of remounting from remote cede state."""

    pilot_id: str = Field(..., description="ID of the pilot")
    mech_id: str = Field(..., description="ID of the mech")
    remount_success: bool = Field(
        default=False, description="Whether remount succeeded"
    )
    control_restored: bool = Field(
        default=False, description="Whether control was restored"
    )
    validation_errors: list[str] = Field(default_factory=list)


class CedeControlApplicationResult(FrozenModel):
    """Result of applying cede control to combatant state."""

    updated_combatant: CombatantState = Field(..., description="Updated combatant")
    control_state_changed: bool = Field(
        default=False, description="Whether control state changed"
    )
    cede_turns_remaining: int = Field(default=0, description="Cede duration remaining")


class CedeControlRemoteApplicationResult(FrozenModel):
    """Result of applying remote cede control to combatant state."""

    updated_combatant: CombatantState = Field(..., description="Updated mech combatant")
    pilot_state: dict = Field(
        default_factory=dict, description="Pilot state after exit"
    )
    control_state_changed: bool = Field(default=False)
    pilot_exited: bool = Field(default=False)


class UnshackleApplicationResult(FrozenModel):
    """Result of applying unshackle to combatant state."""

    updated_combatant: CombatantState = Field(..., description="Updated combatant")
    unshackled: bool = Field(default=False, description="Whether unshackled")
    nhp_behavior: NHPBehaviorType | None = Field(
        default=None, description="Selected NHP behavior"
    )
    pilot_ejected: bool = Field(default=False, description="Whether pilot was ejected")
    control_state: AIControlState = Field(default="pilot")


class RemountApplicationResult(FrozenModel):
    """Result of applying remount to combatant state."""

    updated_pilot: CombatantState | None = Field(
        default=None, description="Updated pilot state"
    )
    updated_mech: CombatantState | None = Field(
        default=None, description="Updated mech state"
    )
    remount_success: bool = Field(default=False)
    control_restored: bool = Field(default=False)


def resolve_cede_control(
    input: CedeControlInput,
    has_ai_property: bool = True,
) -> CedeControlResult:
    """Resolve ceding control to AI (pilot remains in mech) per PR2.

    Free action at start of turn. Pilot remains in mech and can resume
    control at the start of their next turn.

    Args:
        input: Cede control input with actor information
        has_ai_property: Whether mech has AI property

    Returns:
        Detailed breakdown of cede control result
    """
    rules = input.rules or DEFAULT_AI_RULES
    errors: list[str] = []

    if not has_ai_property:
        errors.append("Mech does not have AI property")

    return CedeControlResult(
        actor_id=input.actor_id,
        control_ceded=len(errors) == 0,
        cede_turns=rules.cede_turns_duration,
        validation_errors=errors,
    )


def resolve_cede_control_remote(
    input: CedeControlRemoteInput,
    has_ai_property: bool = True,
) -> CedeControlRemoteResult:
    """Resolve ceding control to AI with remote exit per PR2.

    Free action at start of turn. Pilot exits mech to adjacent space
    and must remount to resume control.

    Args:
        input: Remote cede control input with actor information
        has_ai_property: Whether mech has AI property

    Returns:
        Detailed breakdown of remote cede result
    """
    rules = input.rules or DEFAULT_AI_RULES
    errors: list[str] = []

    if not has_ai_property:
        errors.append("Mech does not have AI property")

    return CedeControlRemoteResult(
        actor_id=input.actor_id,
        control_ceded=len(errors) == 0,
        pilot_exited=len(errors) == 0,
        exit_position="adjacent",
        validation_errors=errors,
    )


def resolve_unshackle_check(
    input: UnshackleCheckInput,
    has_nhp: bool = True,
) -> UnshackleCheckResult:
    """Resolve unshackle check on structure or overheat per PR2 5081-5082.

    Each time you roll a structure check or overheating check, roll a d20.
    On a roll of 1, your NHP's casket has suffered a traumatic impact or
    code incursion and your NHP becomes Unshackled.

    Only applies to NHPs, not Comp/Con units.

    Args:
        input: Unshackle check input with actor and check type
        has_nhp: Whether mech has NHP installed

    Returns:
        Detailed breakdown of unshackle check result
    """
    rules = input.rules or DEFAULT_AI_RULES
    errors: list[str] = []

    if not has_nhp:
        return UnshackleCheckResult(
            actor_id=input.actor_id,
            check_type=input.check_type,
            d20_roll=0,
            unshackle_occurred=False,
            nhps_affected=0,
            validation_errors=errors,
        )

    if input.force_roll is not None:
        d20_roll = input.force_roll
    else:
        d20_roll = DiceExpression.parse("1d20").roll()[0]

    unshackle_occurred = d20_roll == rules.unshackle_threshold

    return UnshackleCheckResult(
        actor_id=input.actor_id,
        check_type=input.check_type,
        d20_roll=d20_roll,
        unshackle_occurred=unshackle_occurred,
        nhps_affected=1 if unshackle_occurred else 0,
        validation_errors=errors,
    )


def resolve_unshackled_behavior(
    input: UnshackledBehaviorInput,
) -> UnshackledBehaviorResult:
    """Resolve NHP behavior when unshackled per PR2 5134-5135.

    Unshackled NHP generally plans its own agenda and will always act
    in one of the following ways: ignore you, overrule you, act outside
    the constraints of human logic patterns or desires, or try to get
    you out of the way.

    Args:
        input: Behavior selection input

    Returns:
        Selected NHP behavior with full configuration
    """
    rules = input.rules or DEFAULT_AI_RULES

    import random

    behavior = random.choices(
        list(rules.behavior_weights.keys()),
        weights=list(rules.behavior_weights.values()),
        k=1,
    )[0]

    behavior_config = None
    for config in DEFAULT_NHP_BEHAVIORS:
        if config.behavior == behavior:
            behavior_config = config
            break

    if behavior_config is None:
        behavior_config = NHPBehaviorConfig(behavior=behavior)

    return UnshackledBehaviorResult(
        actor_id=input.actor_id,
        behavior=behavior,
        behavior_config=behavior_config,
        remove_pilot_mode=behavior_config.remove_pilot_mode
        if behavior == "remove_pilot"
        else None,
    )


def resolve_remount(
    input: RemountInput,
    pilot_in_cede_remote: bool = True,
) -> RemountResult:
    """Resolve remounting mech from remote cede state per PR2.

    Pilot must be adjacent to their mech to remount. Remounting restores
    pilot control and ends the remote cede state.

    Args:
        input: Remount input with pilot and mech IDs
        pilot_in_cede_remote: Whether pilot is actually in cede_remote state

    Returns:
        Detailed breakdown of remount result
    """
    errors: list[str] = []

    if not pilot_in_cede_remote:
        errors.append("Pilot is not in cede_remote state")

    if not input.is_adjacent:
        errors.append("Pilot must be adjacent to mech to remount")

    return RemountResult(
        pilot_id=input.pilot_id,
        mech_id=input.mech_id,
        remount_success=len(errors) == 0,
        control_restored=len(errors) == 0,
        validation_errors=errors,
    )


def apply_cede_control(
    combatant: CombatantState,
    result: CedeControlResult,
) -> CedeControlApplicationResult:
    """Apply cede control result to combatant state.

    Updates control state to "cede" and sets cede duration.

    Args:
        combatant: Current combatant state
        result: Resolution result to apply

    Returns:
        Updated combatant with cede control applied
    """
    if not result.control_ceded or result.validation_errors:
        return CedeControlApplicationResult(
            updated_combatant=combatant,
            control_state_changed=False,
            cede_turns_remaining=0,
        )

    updated_combatant = combatant.model_copy(
        update={
            "ai_control_state": "cede",
        }
    )

    return CedeControlApplicationResult(
        updated_combatant=updated_combatant,
        control_state_changed=True,
        cede_turns_remaining=result.cede_turns,
    )


def apply_cede_control_remote(
    combatant: CombatantState,
    pilot_combatant: CombatantState,
    result: CedeControlRemoteResult,
) -> CedeControlRemoteApplicationResult:
    """Apply remote cede control result to combatant states.

    Updates mech to "cede_remote" and creates pilot state for exited pilot.

    Args:
        combatant: Mech combatant state
        pilot_combatant: Pilot combatant state
        result: Resolution result to apply

    Returns:
        Updated combatants with remote cede applied
    """
    if not result.control_ceded or result.validation_errors:
        return CedeControlRemoteApplicationResult(
            updated_combatant=combatant,
            pilot_state={},
            control_state_changed=False,
            pilot_exited=False,
        )

    updated_mech = combatant.model_copy(
        update={
            "ai_control_state": "cede_remote",
        }
    )

    pilot_state = {
        "position": result.exit_position,
        "statuses": list(pilot_combatant.statuses),
        "conditions": list(pilot_combatant.conditions),
    }

    return CedeControlRemoteApplicationResult(
        updated_combatant=updated_mech,
        pilot_state=pilot_state,
        control_state_changed=True,
        pilot_exited=True,
    )


def apply_unshackle(
    combatant: CombatantState,
    unshackle_result: UnshackleCheckResult,
    behavior_result: UnshackledBehaviorResult,
) -> UnshackleApplicationResult:
    """Apply unshackle result to combatant state.

    Updates control state to "unshackled" and sets NHP behavior.

    Args:
        combatant: Current combatant state
        unshackle_result: Unshackle check result
        behavior_result: Behavior selection result

    Returns:
        Updated combatant with unshackle applied
    """
    if not unshackle_result.unshackle_occurred:
        return UnshackleApplicationResult(
            updated_combatant=combatant,
            unshackled=False,
            nhp_behavior=None,
            pilot_ejected=False,
            control_state=combatant.ai_control_state,
        )

    pilot_ejected = behavior_result.behavior == "remove_pilot"

    updated_statuses = list(combatant.statuses)
    if "unshackled" not in updated_statuses:
        updated_statuses.append("unshackled")

    updated_combatant = combatant.model_copy(
        update={
            "ai_control_state": "unshackled",
            "nhp_behavior": behavior_result.behavior,
            "statuses": updated_statuses,
        }
    )

    return UnshackleApplicationResult(
        updated_combatant=updated_combatant,
        unshackled=True,
        nhp_behavior=behavior_result.behavior,
        pilot_ejected=pilot_ejected,
        control_state="unshackled",
    )


def apply_remount(
    pilot: CombatantState,
    mech: CombatantState,
    result: RemountResult,
) -> RemountApplicationResult:
    """Apply remount result to combatant states.

    Restores pilot to mech and resets control state to "pilot".

    Args:
        pilot: Pilot combatant state
        mech: Mech combatant state
        result: Remount resolution result

    Returns:
        Updated combatants with remount applied
    """
    if not result.remount_success:
        return RemountApplicationResult(
            updated_pilot=pilot,
            updated_mech=mech,
            remount_success=False,
            control_restored=False,
        )

    updated_pilot = pilot.model_copy(
        update={
            "position": mech.position,
        }
    )

    updated_mech = mech.model_copy(
        update={
            "ai_control_state": "pilot",
        }
    )

    return RemountApplicationResult(
        updated_pilot=updated_pilot,
        updated_mech=updated_mech,
        remount_success=True,
        control_restored=True,
    )


def check_has_ai_property(combatant: CombatantState) -> bool:
    """Check if a combatant has AI property.

    AI property is determined by having an AI-type system installed.

    Args:
        combatant: Combatant to check

    Returns:
        True if combatant has AI property
    """
    if not combatant.inventory:
        return False


    for system in combatant.inventory.systems:
        if system.system_id:
            from core.mech.compendium import get_system_by_id

            try:
                sys = get_system_by_id(system.system_id)
                if sys and hasattr(sys, "system_type"):
                    if sys.system_type == "ai":
                        return True
            except (ImportError, AttributeError):
                pass

    return False


def check_has_nhp(combatant: CombatantState) -> bool:
    """Check if a combatant has an NHP installed.

    NHPs are a specific type of AI that can unshackle.

    Args:
        combatant: Combatant to check

    Returns:
        True if combatant has NHP installed
    """
    if combatant.ai_type == "nhp":
        return True

    if not combatant.inventory:
        return False


    for system in combatant.inventory.systems:
        if system.system_id:
            from core.mech.compendium import get_system_by_id

            try:
                sys = get_system_by_id(system.system_id)
                if sys and hasattr(sys, "system_type"):
                    if sys.system_type == "ai":
                        if hasattr(sys, "is_nhp") and sys.is_nhp:
                            return True
            except (ImportError, AttributeError):
                pass

    return False


try:
    from core.shared.heat import MeltdownState
    from core.mech.combat_state import CombatantState

    CombatantState.model_rebuild(
        _types_namespace={"MeltdownState": MeltdownState, "StabilizeState": type(None)}
    )
except ImportError:
    pass
