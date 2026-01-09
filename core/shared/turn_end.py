"""Turn end resolution system for Lancer TTRPG.

Implements turn end mechanics per PR2 5099-5102:

Turn End Processing:
- End-of-turn triggers activate (talents, gear, abilities)
- Multiple triggers: actor chooses order
- Effects with "end_of_turn" duration expire
- Effects with "end_of_next_turn" are tracked for next turn

Note: Per-round reactions reset at ROUND start (see combat_resolution.py:945).
No conditions auto-clear at turn end - conditions require actions to clear.

Resolution Pattern:
1. resolve_turn_end_triggers() - Process triggers in actor-specified order
2. expire_end_of_turn_effects() - Clear end_of_turn duration effects
3. advance_end_of_next_turn_effects() - Track effects for next turn
4. resolve_turn_end() - Orchestrate complete turn end processing
"""

from __future__ import annotations

from typing import Literal, Any
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.effects import EffectDuration
from core.shared.id_helpers import CombatantIdField


TurnEndTriggerSource = Literal[
    "talent",
    "system",
    "weapon",
    "condition",
    "custom",
]


class TurnEndEffectState(FrozenModel):
    """State tracking for effects that expire at turn boundaries.

    Tracks effects like:
    - Disengage (ignores engagement/reactions)
    - Temporary buffs from talents/systems
    - Cover grants from abilities
    - Invisibility effects

    Attributes:
        effect_id: Unique identifier for this effect instance
        effect_type: Type of effect (disengage, buff, cover_grant, etc.)
        target_id: Target combatant (None for self/aura effects)
        duration_type: When the effect expires
        applied_by: Actor who applied the effect
        effect_data: Additional effect details
    """

    effect_id: str = Field(..., description="Unique identifier for this effect")
    effect_type: str = Field(..., description="Type of effect")
    target_id: CombatantIdField | None = Field(
        default=None, description="Target of the effect"
    )
    duration_type: EffectDuration = Field(..., description="When the effect expires")
    applied_by: str = Field(..., description="Actor who applied this effect")
    effect_data: dict[str, Any] = Field(
        default_factory=dict, description="Additional effect details"
    )


class TurnEndTrigger(FrozenModel):
    """A trigger that activates at the end of turn.

    Many talents and gear activate at end of turn. Per PR2 5099-5102:
    "If you have multiple effects that trigger at end of turn, you can
    choose the order in which they trigger."

    Attributes:
        trigger_id: Unique identifier for this trigger
        trigger_name: Human-readable name of the trigger
        source_type: What type of source owns this trigger
        actor_id: Who owns this trigger
        trigger_condition: Description of when it triggers
        effect_data: What happens when triggered
        order_priority: Default ordering when actor doesn't specify
    """

    trigger_id: str = Field(..., description="Unique identifier for this trigger")
    trigger_name: str = Field(..., description="Human-readable name")
    source_type: TurnEndTriggerSource = Field(..., description="Source type")
    actor_id: str = Field(..., description="Who owns this trigger")
    trigger_condition: str = Field(
        default="always", description="When this trigger activates"
    )
    effect_data: dict[str, Any] = Field(
        default_factory=dict, description="Effect to apply when triggered"
    )
    order_priority: int = Field(default=0, description="Default sort order")


class TurnEndInput(FrozenModel):
    """Input for turn end processing.

    Attributes:
        actor_id: Combatant ending their turn
        round_number: Current round number
        turn_number: Current turn number within round
        triggers: List of triggers to resolve
        active_effects: Currently active turn-end effects
        specified_order: Actor-specified trigger order (None = default)
    """

    actor_id: str = Field(..., description="Combatant ending their turn")
    round_number: int = Field(default=1, ge=1, description="Current round")
    turn_number: int = Field(default=1, ge=1, description="Turn number in round")
    triggers: list[TurnEndTrigger] = Field(
        default_factory=list, description="Triggers to resolve"
    )
    active_effects: dict[str, TurnEndEffectState] = Field(
        default_factory=dict, description="Active turn-end effects"
    )
    specified_order: list[str] | None = Field(
        default=None, description="Actor-specified trigger order"
    )


class TurnEndTriggerResult(FrozenModel):
    """Result of resolving a single turn end trigger.

    Attributes:
        trigger_id: Trigger that was resolved
        trigger_name: Name of the trigger
        triggered: Whether the trigger conditions were met
        effect_summary: Description of effects applied
        effects_created: New effects created by this trigger
    """

    trigger_id: str = Field(..., description="Trigger that was resolved")
    trigger_name: str = Field(..., description="Name of the trigger")
    triggered: bool = Field(default=True, description="Whether trigger activated")
    effect_summary: str = Field(default="", description="Effects applied")
    effects_created: list[str] = Field(
        default_factory=list, description="New effect IDs created"
    )


class TurnEndResult(FrozenModel):
    """Complete result of turn end processing.

    Attributes:
        actor_id: Combatant whose turn ended
        round_number: Round when turn ended
        turn_number: Turn number when turn ended
        triggers_resolved: Results of each trigger resolved
        effects_expired: Effect IDs that expired this turn
        effects_for_next_turn: Effect IDs now expiring next turn
        new_effects: New effects added to tracking
        status_summary: Human-readable summary
    """

    actor_id: str = Field(..., description="Combatant whose turn ended")
    round_number: int = Field(..., description="Round when turn ended")
    turn_number: int = Field(..., description="Turn number when turn ended")
    triggers_resolved: list[TurnEndTriggerResult] = Field(
        default_factory=list, description="Trigger resolution results"
    )
    effects_expired: list[str] = Field(
        default_factory=list, description="Effect IDs that expired"
    )
    effects_for_next_turn: list[str] = Field(
        default_factory=list, description="Effect IDs for next turn tracking"
    )
    new_effects: dict[str, TurnEndEffectState] = Field(
        default_factory=dict, description="New effects added"
    )
    status_summary: str = Field(default="", description="Summary of changes")


def resolve_turn_end_triggers(
    input: TurnEndInput,
) -> tuple[list[TurnEndTriggerResult], dict[str, TurnEndEffectState]]:
    """Resolve all end-of-turn triggers.

    Per PR2 5099-5102: "If you have multiple effects that trigger at end of turn,
    you can choose the order in which they trigger."

    Processing:
    1. Sort triggers by specified order or default priority
    2. Resolve each trigger in order
    3. Collect any new effects created

    Args:
        input: Turn end input with triggers to resolve

    Returns:
        Tuple of (trigger_results, new_effects_created)
    """
    triggers = input.triggers
    specified_order = input.specified_order

    if specified_order:
        trigger_map = {t.trigger_id: t for t in triggers}
        sorted_triggers = [
            trigger_map[tid] for tid in specified_order if tid in trigger_map
        ]
        remaining = [t for t in triggers if t.trigger_id not in specified_order]
        sorted_triggers.extend(sorted(remaining, key=lambda t: t.order_priority))
    else:
        sorted_triggers = sorted(triggers, key=lambda t: t.order_priority)

    results: list[TurnEndTriggerResult] = []
    new_effects: dict[str, TurnEndEffectState] = {}

    for trigger in sorted_triggers:
        effect_data = trigger.effect_data

        effect_type = effect_data.get("type", "buff")
        effect_summary = ""

        match effect_type:
            case "buff":
                buff_type = effect_data.get("buff_type", "accuracy")
                value = effect_data.get("value", 1)
                effect_summary = f"+{value} {buff_type} until end of next turn"
                effect_id = f"{trigger.trigger_id}:buff"
                new_effects[effect_id] = TurnEndEffectState(
                    effect_id=effect_id,
                    effect_type="buff",
                    target_id=effect_data.get("target_id"),
                    duration_type="end_of_next_turn",
                    applied_by=trigger.actor_id,
                    effect_data=effect_data,
                )

            case "condition":
                condition = effect_data.get("condition", "impaired")
                effect_summary = f"Apply {condition} until end of next turn"
                effect_id = f"{trigger.trigger_id}:condition"
                new_effects[effect_id] = TurnEndEffectState(
                    effect_id=effect_id,
                    effect_type="condition",
                    target_id=effect_data.get("target_id"),
                    duration_type="end_of_next_turn",
                    applied_by=trigger.actor_id,
                    effect_data=effect_data,
                )

            case "cover_grant":
                cover_type = effect_data.get("cover_type", "soft")
                effect_summary = f"Grant {cover_type} cover until end of next turn"
                effect_id = f"{trigger.trigger_id}:cover"
                new_effects[effect_id] = TurnEndEffectState(
                    effect_id=effect_id,
                    effect_type="cover_grant",
                    target_id=effect_data.get("target_id"),
                    duration_type="end_of_next_turn",
                    applied_by=trigger.actor_id,
                    effect_data=effect_data,
                )

            case "disengage":
                effect_summary = "Disengage: ignore engagement and reactions"
                effect_id = f"{trigger.trigger_id}:disengage"
                new_effects[effect_id] = TurnEndEffectState(
                    effect_id=effect_id,
                    effect_type="disengage",
                    target_id=trigger.actor_id,
                    duration_type="end_of_turn",
                    applied_by=trigger.actor_id,
                    effect_data=effect_data,
                )

            case "custom":
                effect_summary = effect_data.get("summary", "Custom effect applied")
                if effect_data.get("has_duration"):
                    effect_id = f"{trigger.trigger_id}:custom"
                    new_effects[effect_id] = TurnEndEffectState(
                        effect_id=effect_id,
                        effect_type="custom",
                        target_id=effect_data.get("target_id"),
                        duration_type=effect_data.get(
                            "duration_type", "end_of_next_turn"
                        ),
                        applied_by=trigger.actor_id,
                        effect_data=effect_data,
                    )

            case _:
                effect_summary = f"{effect_type} effect applied"

        result = TurnEndTriggerResult(
            trigger_id=trigger.trigger_id,
            trigger_name=trigger.trigger_name,
            triggered=True,
            effect_summary=effect_summary,
            effects_created=list(new_effects.keys())[
                len(new_effects) - len(effect_data.get("effects_created", [])) :
            ]
            if effect_data.get("effects_created")
            else [],
        )
        results.append(result)

    return results, new_effects


def expire_end_of_turn_effects(
    active_effects: dict[str, TurnEndEffectState],
) -> tuple[dict[str, TurnEndEffectState], list[str]]:
    """Expire effects with duration_type="end_of_turn".

    Called at the very end of a turn to clear effects like:
    - Disengage (ignores engagement/reactions)
    - Temporary buffs
    - Cover grants from abilities

    Args:
        active_effects: Currently active turn-end effects

    Returns:
        Tuple of (remaining_effects, expired_effect_ids)
    """
    remaining: dict[str, TurnEndEffectState] = {}
    expired: list[str] = []

    for effect_id, effect in active_effects.items():
        if effect.duration_type == "end_of_turn":
            expired.append(effect_id)
        else:
            remaining[effect_id] = effect

    return remaining, expired


def advance_end_of_next_turn_effects(
    active_effects: dict[str, TurnEndEffectState],
) -> tuple[dict[str, TurnEndEffectState], list[str]]:
    """Update end_of_next_turn effects for next turn tracking.

    When a combatant's turn ends, effects with duration_type="end_of_next_turn"
    targeting them should now expire at the end of the NEXT actor's turn.

    This function re-labels these effects as "end_of_turn" for tracking.

    Args:
        active_effects: Currently active turn-end effects

    Returns:
        Tuple of (updated_effects, effect_ids_for_next_turn)
    """
    updated: dict[str, TurnEndEffectState] = {}
    for_next_turn: list[str] = []

    for effect_id, effect in active_effects.items():
        if effect.duration_type == "end_of_next_turn":
            updated[effect_id] = effect.model_copy(
                update={"duration_type": "end_of_turn"}
            )
            for_next_turn.append(effect_id)
        else:
            updated[effect_id] = effect

    return updated, for_next_turn


def apply_turn_end_effect_to_state(
    combatant_statuses: dict[str, Any],
    effects: dict[str, TurnEndEffectState],
) -> dict[str, Any]:
    """Apply active turn-end effects to combatant state.

    Helper to integrate turn-end effects with combatant state.

    Args:
        combatant_statuses: Current status dict to modify
        effects: Active turn-end effects to apply

    Returns:
        Updated status dict with effects applied
    """
    disengage_active = False
    cover_type = None
    accuracy_bonus = 0
    defense_bonus = 0

    for effect in effects.values():
        if effect.effect_type == "disengage":
            disengage_active = True
        elif effect.effect_type == "cover_grant":
            cover_type = effect.effect_data.get("cover_type", "soft")
        elif effect.effect_type == "buff":
            buff_type = effect.effect_data.get("buff_type")
            value = effect.effect_data.get("value", 0)
            if buff_type == "accuracy":
                accuracy_bonus += value
            elif buff_type in ("evasion", "e_defense"):
                defense_bonus += value

    result = dict(combatant_statuses)
    if disengage_active:
        result["ignores_engagement"] = True
        result["prevents_reactions"] = True
    if cover_type:
        result["cover_grant"] = cover_type
    if accuracy_bonus:
        result["accuracy_bonus"] = accuracy_bonus
    if defense_bonus:
        result["defense_bonus"] = defense_bonus

    return result


def resolve_turn_end(
    input: TurnEndInput,
) -> TurnEndResult:
    """Orchestrate complete end-of-turn processing.

    Per PR2 5099-5102:
    1. All triggers activate at end of turn
    2. Multiple triggers: actor chooses order
    3. Effects with "end_of_turn" duration expire
    4. Status/conditions that clear at turn end are cleared

    Processing order:
    a) Resolve all end-of-turn triggers (in specified order)
    b) Expire "end_of_turn" duration effects
    c) Update "end_of_next_turn" tracking for next turn

    Note: Per-round reactions reset at ROUND start (see combat_resolution.py:945).
    No conditions auto-clear at turn end - conditions require actions.

    Args:
        input: Complete turn end input

    Returns:
        TurnEndResult with all changes
    """
    all_new_effects: dict[str, TurnEndEffectState] = {}

    trigger_results, trigger_new_effects = resolve_turn_end_triggers(input)
    all_new_effects.update(trigger_new_effects)

    merged_effects = {**input.active_effects, **all_new_effects}

    remaining_effects, expired = expire_end_of_turn_effects(merged_effects)

    updated_effects, for_next_turn = advance_end_of_next_turn_effects(remaining_effects)

    trigger_summaries = [r.effect_summary for r in trigger_results if r.effect_summary]
    if trigger_summaries:
        status_summary = f"Triggers: {', '.join(trigger_summaries[:3])}"
        if len(trigger_summaries) > 3:
            status_summary += f" (+{len(trigger_summaries) - 3} more)"
    elif expired:
        status_summary = f"Expired {len(expired)} effect(s)"
    else:
        status_summary = "No turn-end effects"

    return TurnEndResult(
        actor_id=input.actor_id,
        round_number=input.round_number,
        turn_number=input.turn_number,
        triggers_resolved=trigger_results,
        effects_expired=expired,
        effects_for_next_turn=for_next_turn,
        new_effects=updated_effects,
        status_summary=status_summary,
    )


def create_end_of_turn_effect(
    effect_id: str,
    effect_type: str,
    duration_type: EffectDuration,
    applied_by: str,
    target_id: str | None = None,
    effect_data: dict[str, Any] | None = None,
) -> TurnEndEffectState:
    """Helper to create a TurnEndEffectState.

    Args:
        effect_id: Unique identifier for this effect
        effect_type: Type of effect (disengage, buff, cover_grant, etc.)
        duration_type: When the effect expires
        applied_by: Actor who applied the effect
        target_id: Target combatant (None for self/aura)
        effect_data: Additional effect details

    Returns:
        New TurnEndEffectState
    """
    return TurnEndEffectState(
        effect_id=effect_id,
        effect_type=effect_type,
        target_id=target_id,
        duration_type=duration_type,
        applied_by=applied_by,
        effect_data=effect_data or {},
    )


def create_end_of_turn_trigger(
    trigger_id: str,
    trigger_name: str,
    source_type: TurnEndTriggerSource,
    actor_id: str,
    effect_data: dict[str, Any] | None = None,
    order_priority: int = 0,
) -> TurnEndTrigger:
    """Helper to create a TurnEndTrigger.

    Args:
        trigger_id: Unique identifier for this trigger
        trigger_name: Human-readable name
        source_type: What type of source owns this trigger
        actor_id: Who owns this trigger
        effect_data: Effect to apply when triggered
        order_priority: Default ordering when actor doesn't specify

    Returns:
        New TurnEndTrigger
    """
    return TurnEndTrigger(
        trigger_id=trigger_id,
        trigger_name=trigger_name,
        source_type=source_type,
        actor_id=actor_id,
        effect_data=effect_data or {},
        order_priority=order_priority,
    )


def has_active_turn_end_effect(
    effects: dict[str, TurnEndEffectState],
    effect_type: str,
    target_id: str | None = None,
) -> bool:
    """Check if a specific type of turn-end effect is active.

    Args:
        effects: Active turn-end effects
        effect_type: Type to check for
        target_id: Optional target to check

    Returns:
        True if such an effect exists
    """
    for effect in effects.values():
        if effect.effect_type != effect_type:
            continue
        if target_id is not None and effect.target_id != target_id:
            continue
        return True
    return False


def get_active_effects_by_type(
    effects: dict[str, TurnEndEffectState],
    effect_type: str,
) -> list[TurnEndEffectState]:
    """Get all active effects of a specific type.

    Args:
        effects: Active turn-end effects
        effect_type: Type to filter by

    Returns:
        List of matching effects
    """
    return [e for e in effects.values() if e.effect_type == effect_type]
