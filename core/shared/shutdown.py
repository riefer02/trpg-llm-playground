"""Shut Down and Boot Up action resolution primitives for Lancer TTRPG.

Implements resolution logic for:
- Shut Down (Quick action): PR2 4299-4306
- Boot Up (Full action): PR2 4307-4316

Shut Down Effects:
- Mech enters Shut Down status (stunned, immune to tech)
- Heat resets to 0, ends exposed condition
- All tech effects/conditions end immediately
- Any unshackled AI are re-shackled

Boot Up Effects:
- Ends Shut Down status
- Must be piloting the mech to boot up

Resolution Pattern:
1. resolve_shutdown() / resolve_boot_up() - Pure resolution logic
2. apply_shutdown_result() / apply_boot_up_result() - Apply to combatant state
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType

from core.mech.combat_state import CombatantState


class ShutDownInput(FrozenModel):
    """Input for Shut Down action resolution."""

    actor_id: str = Field(..., description="ID of the actor performing Shut Down")


class BootUpInput(FrozenModel):
    """Input for Boot Up action resolution."""

    actor_id: str = Field(..., description="ID of the actor performing Boot Up")
    is_piloting: bool = Field(
        default=True, description="Whether the pilot is currently piloting the mech"
    )


class TechEffectType(FrozenModel):
    """Represents a tech effect or condition that can be ended by Shut Down."""

    effect_type: Literal["condition", "status", "lock_on"]
    effect_id: StatusType | Literal["lock_on"]
    effect_name: str


SHUT_DOWN_ENDED_EFFECTS: list[TechEffectType] = [
    TechEffectType(effect_type="status", effect_id="lock_on", effect_name="Lock On"),
    TechEffectType(
        effect_type="condition", effect_id="impaired", effect_name="Impaired"
    ),
    TechEffectType(effect_type="condition", effect_id="slowed", effect_name="Slowed"),
    TechEffectType(effect_type="condition", effect_id="jammed", effect_name="Jammed"),
    TechEffectType(effect_type="condition", effect_id="stunned", effect_name="Stunned"),
]


class ShutDownResolutionResult(FrozenModel):
    """Complete result of Shut Down resolution (pure logic).

    Provides detailed breakdown of what should happen during Shut Down action.
    """

    actor_id: str = Field(..., description="ID of the actor performing Shut Down")
    heat_cleared: bool = Field(
        default=True, description="Whether heat was cleared to 0"
    )
    exposed_cleared: bool = Field(
        default=True, description="Whether exposed condition was ended"
    )
    shutdown_status_applied: bool = Field(
        default=True, description="Whether shutdown status was applied"
    )
    tech_effects_ended: list[str] = Field(
        default_factory=list, description="Tech effects/conditions that were ended"
    )
    ai_reshackled: bool = Field(
        default=False, description="Whether unshackled AI were re-shackled"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class BootUpResolutionResult(FrozenModel):
    """Complete result of Boot Up resolution (pure logic).

    Provides detailed breakdown of what should happen during Boot Up action.
    """

    actor_id: str = Field(..., description="ID of the actor performing Boot Up")
    shutdown_status_ended: bool = Field(
        default=True, description="Whether shutdown status was ended"
    )
    pilot_required: bool = Field(
        default=True, description="Whether pilot must be present to boot up"
    )
    was_piloting: bool = Field(
        default=True, description="Whether the actor was piloting the mech"
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Any validation errors encountered"
    )


class ShutDownApplicationResult(FrozenModel):
    """Result of applying Shut Down result to combatant state."""

    updated_combatant: CombatantState = Field(
        ..., description="Combatant with Shut Down result applied"
    )
    heat_cleared: bool = Field(default=False, description="Whether heat was cleared")
    exposed_cleared: bool = Field(
        default=False, description="Whether exposed was ended"
    )
    shutdown_status_added: bool = Field(
        default=False, description="Whether shutdown status was added"
    )
    statuses_removed: list[StatusType] = Field(
        default_factory=list, description="Statuses that were removed"
    )
    conditions_removed: list[StatusType] = Field(
        default_factory=list, description="Conditions that were removed"
    )


class BootUpApplicationResult(FrozenModel):
    """Result of applying Boot Up result to combatant state."""

    updated_combatant: CombatantState = Field(
        ..., description="Combatant with Boot Up result applied"
    )
    shutdown_status_removed: bool = Field(
        default=False, description="Whether shutdown status was removed"
    )
    statuses_removed: list[StatusType] = Field(
        default_factory=list, description="Statuses that were removed"
    )


class ShutDownRule(FrozenModel):
    """Rule configuration for Shut Down action."""

    clears_heat: bool = Field(default=True)
    clears_exposed: bool = Field(default=True)
    ends_tech_effects: bool = Field(default=True)
    reshackles_ai: bool = Field(default=False)
    ends_conditions: list[StatusType] = Field(
        default_factory=lambda: [
            "impaired",
            "slowed",
            "jammed",
            "stunned",
            "lock_on",
        ]
    )


class BootUpRule(FrozenModel):
    """Rule configuration for Boot Up action."""

    requires_pilot: bool = Field(default=True)
    clears_shutdown_status: bool = Field(default=True)


DEFAULT_SHUTDOWN_RULES = ShutDownRule()
DEFAULT_BOOTUP_RULES = BootUpRule()


def resolve_shutdown(
    input: ShutDownInput, rules: ShutDownRule | None = None
) -> ShutDownResolutionResult:
    """Resolve a Shut Down action per PR2 4299-4306.

    Shut Down is a Quick Action that causes the mech to:
    - Enter Shut Down status (stunned, immune to tech)
    - Cool to 0 heat and end exposed condition
    - End all tech effects and conditions
    - Re-shackle any unshackled AI

    Args:
        input: Shut Down input with actor information
        rules: Optional rule configuration

    Returns:
        Detailed breakdown of what should happen during Shut Down
    """
    if rules is None:
        rules = DEFAULT_SHUTDOWN_RULES

    tech_effects_ended: list[str] = []

    if rules.ends_tech_effects:
        for effect in SHUT_DOWN_ENDED_EFFECTS:
            tech_effects_ended.append(effect.effect_name)

    return ShutDownResolutionResult(
        actor_id=input.actor_id,
        heat_cleared=rules.clears_heat,
        exposed_cleared=rules.clears_exposed,
        shutdown_status_applied=True,
        tech_effects_ended=tech_effects_ended,
        ai_reshackled=rules.reshackles_ai,
    )


def apply_shutdown_result(
    combatant: CombatantState, result: ShutDownResolutionResult
) -> ShutDownApplicationResult:
    """Apply Shut Down result to combatant state.

    Updates combatant with heat cleared, statuses/conditions removed, and shutdown status applied.

    Args:
        combatant: Current combatant state
        result: Resolution result to apply

    Returns:
        Updated combatant with Shut Down effects applied
    """
    statuses_removed: list[StatusType] = []
    conditions_removed: list[StatusType] = []

    updated_statuses = list(combatant.statuses)
    updated_conditions = list(combatant.conditions)
    updated_resources = combatant.resources

    if result.heat_cleared:
        updated_resources = updated_resources.model_copy(update={"heat_current": 0})

    if result.exposed_cleared and "exposed" in updated_statuses:
        updated_statuses.remove("exposed")
        statuses_removed.append("exposed")

    for effect in SHUT_DOWN_ENDED_EFFECTS:
        if effect.effect_type == "status" or effect.effect_id == "lock_on":
            if effect.effect_id in updated_statuses:
                updated_statuses.remove(effect.effect_id)
                statuses_removed.append(effect.effect_id)
        elif effect.effect_type == "condition":
            if effect.effect_id in updated_conditions:
                updated_conditions.remove(effect.effect_id)
                conditions_removed.append(effect.effect_id)

    if "shutdown" not in updated_statuses:
        updated_statuses.append("shutdown")

    ai_state_reset = False
    if result.ai_reshackled and combatant.ai_control_state in [
        "ceded",
        "cede_remote",
        "unshackled",
    ]:
        ai_state_reset = True
        updated_statuses = [s for s in updated_statuses if s != "unshackled"]

    update_dict: dict = {
        "statuses": updated_statuses,
        "conditions": updated_conditions,
        "resources": updated_resources,
    }

    if ai_state_reset:
        update_dict["ai_control_state"] = "pilot"
        update_dict["nhp_behavior"] = None

    updated_combatant = combatant.model_copy(update=update_dict)

    return ShutDownApplicationResult(
        updated_combatant=updated_combatant,
        heat_cleared=result.heat_cleared,
        exposed_cleared=result.exposed_cleared,
        shutdown_status_added="shutdown" in updated_statuses,
        statuses_removed=statuses_removed,
        conditions_removed=conditions_removed,
    )


def resolve_boot_up(
    input: BootUpInput, rules: BootUpRule | None = None
) -> BootUpResolutionResult:
    """Resolve a Boot Up action per PR2 4307-4316.

    Boot Up is a Full Action that ends the Shut Down status on a mech.
    The pilot must be piloting the mech to boot it up.

    Args:
        input: Boot Up input with actor and piloting information
        rules: Optional rule configuration

    Returns:
        Detailed breakdown of what should happen during Boot Up
    """
    if rules is None:
        rules = DEFAULT_BOOTUP_RULES

    errors: list[str] = []

    if rules.requires_pilot and not input.is_piloting:
        errors.append("Must be piloting the mech to boot up")

    return BootUpResolutionResult(
        actor_id=input.actor_id,
        shutdown_status_ended=rules.clears_shutdown_status,
        pilot_required=rules.requires_pilot,
        was_piloting=input.is_piloting,
        validation_errors=errors,
    )


def apply_boot_up_result(
    combatant: CombatantState, result: BootUpResolutionResult
) -> BootUpApplicationResult:
    """Apply Boot Up result to combatant state.

    Updates combatant by removing the shutdown status.

    Args:
        combatant: Current combatant state
        result: Resolution result to apply

    Returns:
        Updated combatant with Boot Up effects applied
    """
    if not result.shutdown_status_ended or "shutdown" not in combatant.statuses:
        return BootUpApplicationResult(
            updated_combatant=combatant,
            shutdown_status_removed=False,
        )

    updated_statuses = list(combatant.statuses)
    statuses_removed: list[StatusType] = []

    if "shutdown" in updated_statuses:
        updated_statuses.remove("shutdown")
        statuses_removed.append("shutdown")

    updated_combatant = combatant.model_copy(update={"statuses": updated_statuses})

    return BootUpApplicationResult(
        updated_combatant=updated_combatant,
        shutdown_status_removed=True,
        statuses_removed=statuses_removed,
    )


try:
    from core.shared.heat import MeltdownState
    from core.mech.combat_state import CombatantState

    CombatantState.model_rebuild(
        _types_namespace={"MeltdownState": MeltdownState, "StabilizeState": type(None)}
    )
except ImportError:
    pass  # CombatantState or MeltdownState not yet available during initial import
