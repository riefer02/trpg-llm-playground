"""Action parser for LLM tactician.

Parse LLM output into validated combat actions that can be executed by the combat system.
"""

import json
import re
from typing import List, Optional
from difflib import get_close_matches

from core.mech.combat_models import (
    AvailableAction,
    ActionExecutionInput,
)


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON object from text that may contain surrounding content.

    Args:
        text: Text that may contain a JSON object

    Returns:
        Parsed JSON dict or None if no JSON found
    """
    # Try parsing the whole text as JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object pattern
    json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try to find JSON array pattern (unlikely but possible)
    array_pattern = r"\[(?:[^\[\]]|(?:\{[^{}]*\}))*\]"
    matches = re.findall(array_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


def validate_parsed_json(
    parsed: dict,
) -> tuple[str, Optional[str], Optional[float], str, str, str, str]:
    """Validate parsed JSON has required fields and extract values.

    Args:
        parsed: Parsed JSON dict

    Returns:
        Tuple of (action_id, target_id, confidence, reasoning, situation_assessment, considered_options, rationale)

    Raises:
        ValueError: If required fields missing or invalid
    """
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not a dictionary")

    # Extract action_id (required)
    if "action_id" not in parsed:
        raise ValueError("Missing required field 'action_id'")
    action_id = parsed["action_id"]
    if not isinstance(action_id, str):
        raise ValueError("'action_id' must be a string")

    # Extract target_id (optional, can be null)
    target_id = parsed.get("target_id")
    if target_id is not None and not isinstance(target_id, str):
        raise ValueError("'target_id' must be string or null")

    # Extract confidence (optional)
    confidence = parsed.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            raise ValueError("'confidence' must be a number")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("'confidence' must be between 0.0 and 1.0")

    # Extract reasoning fields (optional, default empty string)
    reasoning = parsed.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ValueError("'reasoning' must be a string")

    situation_assessment = parsed.get("situation_assessment", "")
    if not isinstance(situation_assessment, str):
        raise ValueError("'situation_assessment' must be a string")

    considered_options = parsed.get("considered_options", "")
    if not isinstance(considered_options, str):
        raise ValueError("'considered_options' must be a string")

    rationale = parsed.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("'rationale' must be a string")

    return (
        action_id,
        target_id,
        confidence,
        reasoning,
        situation_assessment,
        considered_options,
        rationale,
    )


def extract_sequence_from_parsed(
    parsed: dict,
) -> list[tuple[str, Optional[str]]]:
    """Extract sequence of actions from parsed JSON.

    Returns list of (action_id, target_id) tuples.
    If 'sequence' field is present, returns its actions.
    Otherwise returns single action from 'action_id' and 'target_id'.
    """
    if "sequence" in parsed:
        sequence = parsed["sequence"]
        if not isinstance(sequence, list):
            raise ValueError("'sequence' must be a list")
        result = []
        for i, action_obj in enumerate(sequence):
            if not isinstance(action_obj, dict):
                raise ValueError(f"Action at index {i} must be a dict")
            action_id = action_obj.get("action_id")
            if action_id is None:
                raise ValueError(f"Action at index {i} missing 'action_id'")
            if not isinstance(action_id, str):
                raise ValueError(f"Action at index {i} 'action_id' must be string")
            target_id = action_obj.get("target_id")
            if target_id is not None and not isinstance(target_id, str):
                raise ValueError(
                    f"Action at index {i} 'target_id' must be string or null"
                )
            result.append((action_id, target_id))
        return result
    else:
        action_id = parsed.get("action_id")
        if action_id is None:
            raise ValueError("Missing required field 'action_id'")
        if not isinstance(action_id, str):
            raise ValueError("'action_id' must be a string")
        target_id = parsed.get("target_id")
        if target_id is not None and not isinstance(target_id, str):
            raise ValueError("'target_id' must be string or null")
        return [(action_id, target_id)]


def find_matching_action(
    action_id: str,
    available_actions: List[AvailableAction],
) -> AvailableAction:
    """Find matching AvailableAction for given action_id.

    Args:
        action_id: Action ID from LLM output
        available_actions: List of available actions

    Returns:
        Matching AvailableAction

    Raises:
        ValueError: If no matching action found
    """
    # Exact match
    for action in available_actions:
        if action.action_id == action_id:
            return action

    # Fuzzy match using action_id
    action_ids = [a.action_id for a in available_actions]
    matches = get_close_matches(action_id, action_ids, n=1, cutoff=0.6)
    if matches:
        matched_id = matches[0]
        for action in available_actions:
            if action.action_id == matched_id:
                return action

    # Fuzzy match using action_name (case-insensitive)
    action_names = [a.action_name.lower() for a in available_actions]
    matches = get_close_matches(action_id.lower(), action_names, n=1, cutoff=0.6)
    if matches:
        matched_name = matches[0]
        for action in available_actions:
            if action.action_name.lower() == matched_name:
                return action

    raise ValueError(
        f"No matching action found for '{action_id}'. "
        f"Available actions: {[a.action_id for a in available_actions]}"
    )


def parse_llm_action(
    llm_output: str,
    actor_id: str,
    available_actions: List[AvailableAction],
) -> tuple[ActionExecutionInput, dict]:
    """Parse LLM output into validated combat action and reasoning fields.

    Handles:
    - JSON parsing with fallback for malformed output
    - Validates parsed action against available_actions list
    - Returns closest valid action if exact match fails (fuzzy matching)
    - Extracts reasoning fields for AI reasoning display

    Args:
        llm_output: LLM output string (should contain JSON)
        actor_id: ID of the combatant taking action
        available_actions: List of available actions for this actor

    Returns:
        Tuple of (ActionExecutionInput, reasoning_fields dict)
        reasoning_fields includes: reasoning, confidence, situation_assessment,
        considered_options, rationale.

    Raises:
        ValueError: If parsing fails or no valid action can be determined
    """
    import warnings

    warnings.warn(
        "parse_llm_action is deprecated, use parse_llm_action_sequence instead",
        DeprecationWarning,
        stacklevel=2,
    )
    action_inputs, reasoning_fields = parse_llm_action_sequence(
        llm_output, actor_id, available_actions
    )
    if not action_inputs:
        raise ValueError("No actions parsed from LLM output")
    return action_inputs[0], reasoning_fields


def parse_llm_action_sequence(
    llm_output: str,
    actor_id: str,
    available_actions: List[AvailableAction],
) -> tuple[list[ActionExecutionInput], dict]:
    """Parse LLM output into sequence of validated combat actions and reasoning fields.

    Handles single action or sequence of actions.
    Returns list of ActionExecutionInput in order to execute.

    Args:
        llm_output: LLM output string (should contain JSON)
        actor_id: ID of the combatant taking action
        available_actions: List of available actions for this actor

    Returns:
        Tuple of (list[ActionExecutionInput], reasoning_fields dict)
        reasoning_fields includes: reasoning, confidence, situation_assessment,
        considered_options, rationale.

    Raises:
        ValueError: If parsing fails or no valid action can be determined
    """
    # Step 1: Extract JSON from LLM output
    parsed = extract_json_from_text(llm_output)
    if parsed is None:
        raise ValueError("No valid JSON found in LLM output")

    # Step 2: Extract reasoning fields
    (
        action_id,  # unused if sequence present
        target_id,  # unused if sequence present
        confidence,
        reasoning,
        situation_assessment,
        considered_options,
        rationale,
    ) = validate_parsed_json(parsed)

    # Step 3: Extract sequence of actions
    sequence = extract_sequence_from_parsed(parsed)

    # Step 4: Build ActionExecutionInput for each action in sequence
    action_inputs = []
    for seq_action_id, seq_target_id in sequence:
        # Find matching action (with fuzzy fallback)
        action = find_matching_action(seq_action_id, available_actions)

        # Validate target requirements
        if action.requires_target and seq_target_id is None:
            raise ValueError(
                f"Action '{action.action_id}' requires a target but none provided"
            )
        if not action.requires_target and seq_target_id is not None:
            # Some actions don't require target but target_id provided anyway
            # We'll ignore it for simplicity
            seq_target_id = None

        target_ids = [seq_target_id] if seq_target_id else []

        action_input = ActionExecutionInput(
            actor_id=actor_id,
            action_id=action.action_id,
            action_type=action.action_type,
            target_ids=target_ids,
            target_position=None,
            weapon_id=None,
            weapon_profile_id=None,
            system_id=None,
            full_tech_first=None,
            full_tech_second=None,
            movement_path=[],
            prompt_dangerous_terrain=False,
            is_overcharge=False,
            granted_by_overcharge=False,
            stabilize_primary=None,
            stabilize_secondary=None,
            apply_knockback=True,
            use_thrown=False,
            eject_direction=None,
            deploy_kind=None,
            deploy_name=None,
            mine_type=None,
            target_mount_id=None,
            target_deployable_id=None,
        )
        action_inputs.append(action_input)

    reasoning_fields = {
        "reasoning": reasoning,
        "confidence": confidence,
        "situation_assessment": situation_assessment,
        "considered_options": considered_options,
        "rationale": rationale,
    }

    return action_inputs, reasoning_fields
