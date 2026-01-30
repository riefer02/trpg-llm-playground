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


def validate_parsed_json(parsed: dict) -> tuple[str, Optional[str], Optional[float]]:
    """Validate parsed JSON has required fields and extract values.

    Args:
        parsed: Parsed JSON dict

    Returns:
        Tuple of (action_id, target_id, confidence)

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

    # reasoning field is ignored for parsing but could be logged

    return action_id, target_id, confidence


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
) -> ActionExecutionInput:
    """Parse LLM output into validated combat action.

    Handles:
    - JSON parsing with fallback for malformed output
    - Validates parsed action against available_actions list
    - Returns closest valid action if exact match fails (fuzzy matching)

    Args:
        llm_output: LLM output string (should contain JSON)
        actor_id: ID of the combatant taking action
        available_actions: List of available actions for this actor

    Returns:
        Validated ActionExecutionInput ready for combat system

    Raises:
        ValueError: If parsing fails or no valid action can be determined
    """
    # Step 1: Extract JSON from LLM output
    parsed = extract_json_from_text(llm_output)
    if parsed is None:
        raise ValueError("No valid JSON found in LLM output")

    # Step 2: Validate parsed JSON structure
    action_id, target_id, confidence = validate_parsed_json(parsed)

    # Step 3: Find matching action (with fuzzy fallback)
    action = find_matching_action(action_id, available_actions)

    # Step 4: Validate target requirements
    if action.requires_target and target_id is None:
        raise ValueError(
            f"Action '{action.action_id}' requires a target but none provided"
        )
    if not action.requires_target and target_id is not None:
        # Some actions don't require target but target_id provided anyway
        # We'll ignore it for simplicity
        target_id = None

    # Step 5: Build ActionExecutionInput
    # Note: weapon_id, system_id, etc. are not provided by LLM output
    # The combat system will need to infer them based on action_id or actor state
    target_ids = [target_id] if target_id else []

    return ActionExecutionInput(
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
