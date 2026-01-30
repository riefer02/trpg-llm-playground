"""LLM Tactician integration with NPC turn execution.

Provides LLM-powered NPC turn automation that can be used as a drop-in
replacement for the rule-based NPC AI in core.
"""

import json
import logging
from typing import Optional

from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
)
from core.mech.combat_execution import (
    start_turn,
    end_turn,
    execute_action,
)
from core.mech.npc_turn_execution import (
    NPCTurnResult,
    CombatTurn,
)
from core.shared.integration.npc_ai import NPCActionDecision

from .tactician import Tactician, TacticianConfig, LLMBackend

logger = logging.getLogger(__name__)


def execute_llm_npc_turn(
    scenario: MechCombatScenario,
    actor_id: str,
    current_round: int,
    current_turn_index: int,
    tactician_config: Optional[TacticianConfig] = None,
) -> tuple[MechCombatScenario, NPCTurnResult]:
    """Execute a full NPC turn using LLM tactician.

    This function mimics the signature of core's execute_npc_turn but uses
    the LLM tactician for decision making.

    Args:
        scenario: Current combat scenario
        actor_id: ID of the NPC to execute turn for
        current_round: Current round number (1-indexed)
        current_turn_index: Current turn index (0-indexed)
        tactician_config: Optional configuration for the tactician

    Returns:
        Tuple of (updated scenario, NPCTurnResult)
    """
    # Find the actor
    actor: Optional[CombatantState] = None
    for combatant in scenario.combatants:
        if combatant.id == actor_id:
            actor = combatant
            break

    if actor is None:
        return scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name="Unknown",
            skipped=True,
            skip_reason="Actor not found",
        )

    # Validate actor is AI-controlled
    if not actor.ai_controlled:
        return scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name=actor.name,
            skipped=True,
            skip_reason="Actor is not AI-controlled",
        )

    # Check if actor is destroyed
    if actor.resources.hp_current <= 0:
        return scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name=actor.name,
            skipped=True,
            skip_reason="Actor is destroyed",
        )

    # Start the turn
    updated_scenario, turn_start_result = start_turn(scenario, actor_id)

    # Refresh actor reference after turn start
    for combatant in updated_scenario.combatants:
        if combatant.id == actor_id:
            actor = combatant
            break

    # Check if there are any enemies (simplified)
    has_targets = False
    for combatant in updated_scenario.combatants:
        if combatant.id == actor_id:
            continue
        if combatant.side == actor.side:
            continue
        if combatant.resources.hp_current <= 0:
            continue
        if combatant.position is None:
            continue
        # Rough distance check (no sensor range for simplicity)
        if actor.position is not None:
            distance = actor.position.coord.distance_to(combatant.position.coord)
            if distance <= 20:  # Arbitrary large distance
                has_targets = True
                break

    if not has_targets:
        # No targets - skip to end turn
        current_turn = CombatTurn(actor_id=actor_id)
        final_scenario, turn_end_result, _, _ = end_turn(
            updated_scenario,
            current_round,
            current_turn_index,
            current_turn,
        )
        return final_scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name=actor.name,
            turn_start=turn_start_result,
            turn_end=turn_end_result,
            actions_taken=0,
            skipped=True,
            skip_reason="No valid targets in range",
        )

    # Initialize tactician
    config = tactician_config or TacticianConfig(
        backend=LLMBackend.OLLAMA,
        model="lancer-expert",
        role=actor.npc_role,  # Use NPC role from combatant state
    )
    tactician = Tactician(config)

    # Get economy state from turn start result
    economy_state = turn_start_result.economy.model_dump(mode="json")

    try:
        # Let tactician decide action(s) - supports single action or sequence
        action_inputs, reasoning_fields = tactician.decide_action(
            updated_scenario,
            actor_id,
            economy_state=economy_state,
        )

        # Build detailed reasoning JSON
        detailed_reasoning_dict = {
            "situation_assessment": reasoning_fields.get("situation_assessment", ""),
            "considered_options": reasoning_fields.get("considered_options", ""),
            "rationale": reasoning_fields.get("rationale", ""),
            "confidence": reasoning_fields.get("confidence"),
        }
        detailed_reasoning_json = json.dumps(detailed_reasoning_dict, indent=2)

        # Create turn for tracking
        current_turn = CombatTurn(actor_id=actor_id)
        economy = turn_start_result.economy
        current_scenario = updated_scenario

        # Track execution results
        action_results = []
        successful_actions = 0

        # Execute each action in sequence
        for i, action_input in enumerate(action_inputs):
            logger.info(
                f"Executing action {i + 1}/{len(action_inputs)}: {action_input.action_id}"
            )

            # Execute the action
            action_scenario, updated_turn, updated_economy, action_result = (
                execute_action(current_scenario, current_turn, economy, action_input)
            )

            # Update state for next action
            current_scenario = action_scenario
            current_turn = updated_turn
            economy = updated_economy
            action_results.append(action_result)

            if action_result.success:
                successful_actions += 1
            else:
                logger.warning(f"Action {i + 1} failed: {action_result.error}")
                # Continue with remaining actions even if one fails

        # Convert first action to NPCActionDecision for logging (legacy format)
        decision = None
        first_action_result = None

        if action_inputs:
            first_action = action_inputs[0]
            reasoning_text = reasoning_fields.get(
                "reasoning", f"LLM tactician chose {first_action.action_id}"
            )
            if len(action_inputs) > 1:
                reasoning_text += (
                    f" (plus {len(action_inputs) - 1} more actions in sequence)"
                )

            decision = NPCActionDecision(
                action=first_action.action_type,
                target_id=first_action.target_ids[0]
                if first_action.target_ids
                else None,
                reasoning=reasoning_text,
                detailed_reasoning=detailed_reasoning_json,
                fallback_used=False,
            )
            first_action_result = action_results[0] if action_results else None
        else:
            # This should not happen (parser would raise), but handle gracefully
            logger.error("No actions returned by tactician")
            # Skip turn with error
            raise ValueError("No actions decided by LLM tactician")

        # End the turn
        final_scenario, turn_end_result, new_round, new_turn_idx = end_turn(
            current_scenario,
            current_round,
            current_turn_index,
            current_turn,
        )

        return final_scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name=actor.name,
            decision=decision,
            action_result=first_action_result,
            turn_start=turn_start_result,
            turn_end=turn_end_result,
            actions_taken=successful_actions,
            skipped=False,
        )

    except Exception as e:
        logger.error(f"LLM tactician failed: {e}")
        # Fallback: use random action from available actions
        # For now, skip turn with error
        current_turn = CombatTurn(actor_id=actor_id)
        final_scenario, turn_end_result, _, _ = end_turn(
            updated_scenario,
            current_round,
            current_turn_index,
            current_turn,
        )
        return final_scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name=actor.name,
            turn_start=turn_start_result,
            turn_end=turn_end_result,
            actions_taken=0,
            skipped=True,
            skip_reason=f"LLM tactician failed: {e}",
        )
