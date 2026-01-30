"""Unit tests for action parser."""

import pytest
from core.mech.combat_models import AvailableAction
from llm.src.tactician.action_parser import (
    parse_llm_action,
    extract_json_from_text,
    validate_parsed_json,
    find_matching_action,
)


# =============================================================================
# Test data
# =============================================================================


def create_mock_actions() -> list[AvailableAction]:
    """Create a list of mock AvailableAction objects for testing."""
    return [
        AvailableAction(
            action_id="skirmish",
            action_name="Skirmish",
            action_type="quick",
            is_available=True,
            requires_target=True,
            requires_weapon=True,
        ),
        AvailableAction(
            action_id="barrage",
            action_name="Barrage",
            action_type="full",
            is_available=True,
            requires_target=True,
            requires_weapon=True,
            max_targets=2,
        ),
        AvailableAction(
            action_id="move",
            action_name="Move",
            action_type="quick",
            is_available=True,
            requires_target=False,
            requires_path=True,
        ),
        AvailableAction(
            action_id="guard",
            action_name="Guard",
            action_type="quick",
            is_available=True,
            requires_target=True,
        ),
        AvailableAction(
            action_id="tech_attack_lock_on",
            action_name="Lock On",
            action_type="quick",
            is_available=True,
            requires_target=True,
            requires_system=True,
        ),
    ]


# =============================================================================
# extract_json_from_text tests
# =============================================================================


def test_extract_json_from_text_valid():
    """Test extracting valid JSON."""
    text = '{"action_id": "skirmish", "target_id": "enemy1", "confidence": 0.8}'
    result = extract_json_from_text(text)
    assert result == {"action_id": "skirmish", "target_id": "enemy1", "confidence": 0.8}


def test_extract_json_from_text_with_surrounding_text():
    """Test extracting JSON from text with surrounding content."""
    text = """
    Here is my decision:
    ```json
    {
        "action_id": "move",
        "target_id": null,
        "reasoning": "I need to reposition",
        "confidence": 0.7
    }
    ```
    That's my final answer.
    """
    result = extract_json_from_text(text)
    assert result is not None
    assert result["action_id"] == "move"
    assert result["target_id"] is None
    assert result["confidence"] == 0.7


def test_extract_json_from_text_multiple_objects():
    """Test extracting first valid JSON object when multiple present."""
    text = """
    First: {"action_id": "wrong", "target_id": "x"}
    Second: {"action_id": "skirmish", "target_id": "enemy1"}
    """
    result = extract_json_from_text(text)
    # Should extract first valid JSON
    assert result == {"action_id": "wrong", "target_id": "x"}


def test_extract_json_from_text_no_json():
    """Test extracting JSON when none exists."""
    text = "Just plain text without JSON"
    result = extract_json_from_text(text)
    assert result is None


# =============================================================================
# validate_parsed_json tests
# =============================================================================


def test_validate_parsed_json_valid():
    """Test validation of valid parsed JSON."""
    parsed = {
        "action_id": "skirmish",
        "target_id": "enemy1",
        "confidence": 0.9,
        "reasoning": "Some reason",
    }
    (
        action_id,
        target_id,
        confidence,
        reasoning,
        situation_assessment,
        considered_options,
        rationale,
    ) = validate_parsed_json(parsed)
    assert action_id == "skirmish"
    assert target_id == "enemy1"
    assert confidence == 0.9
    assert reasoning == "Some reason"
    assert situation_assessment == ""
    assert considered_options == ""
    assert rationale == ""


def test_validate_parsed_json_missing_action_id():
    """Test validation fails when action_id missing."""
    parsed = {"target_id": "enemy1"}
    with pytest.raises(ValueError, match="Missing required field 'action_id'"):
        validate_parsed_json(parsed)


def test_validate_parsed_json_invalid_action_id_type():
    """Test validation fails when action_id is not string."""
    parsed = {"action_id": 123}
    with pytest.raises(ValueError, match="'action_id' must be a string"):
        validate_parsed_json(parsed)


def test_validate_parsed_json_invalid_target_id_type():
    """Test validation fails when target_id is not string or null."""
    parsed = {"action_id": "skirmish", "target_id": 123}
    with pytest.raises(ValueError, match="'target_id' must be string or null"):
        validate_parsed_json(parsed)


def test_validate_parsed_json_invalid_confidence():
    """Test validation fails when confidence is invalid."""
    parsed = {"action_id": "skirmish", "confidence": "high"}
    with pytest.raises(ValueError, match="'confidence' must be a number"):
        validate_parsed_json(parsed)

    parsed = {"action_id": "skirmish", "confidence": 1.5}
    with pytest.raises(ValueError, match="'confidence' must be between 0.0 and 1.0"):
        validate_parsed_json(parsed)


# =============================================================================
# find_matching_action tests
# =============================================================================


def test_find_matching_action_exact():
    """Test exact matching of action_id."""
    actions = create_mock_actions()
    action = find_matching_action("skirmish", actions)
    assert action.action_id == "skirmish"


def test_find_matching_action_fuzzy_id():
    """Test fuzzy matching of action_id."""
    actions = create_mock_actions()
    # "skirmish" misspelled
    action = find_matching_action("skirmis", actions)
    assert action.action_id == "skirmish"


def test_find_matching_action_fuzzy_name():
    """Test fuzzy matching using action_name."""
    actions = create_mock_actions()
    # Use action_name lowercase
    action = find_matching_action("lock on", actions)
    assert action.action_id == "tech_attack_lock_on"


def test_find_matching_action_no_match():
    """Test no matching action raises error."""
    actions = create_mock_actions()
    with pytest.raises(ValueError, match="No matching action found"):
        find_matching_action("nonexistent", actions)


# =============================================================================
# parse_llm_action integration tests
# =============================================================================


def test_parse_llm_action_valid():
    """Test parsing valid LLM output with exact match."""
    llm_output = """{
        "action_id": "skirmish",
        "target_id": "enemy_striker",
        "reasoning": "Target is low HP",
        "confidence": 0.85
    }"""
    actions = create_mock_actions()
    action_input, reasoning_fields = parse_llm_action(llm_output, "actor_1", actions)

    assert action_input.actor_id == "actor_1"
    assert action_input.action_id == "skirmish"
    assert action_input.action_type == "quick"
    assert action_input.target_ids == ["enemy_striker"]
    assert action_input.weapon_id is None  # Not provided by LLM
    assert action_input.system_id is None
    assert reasoning_fields["reasoning"] == "Target is low HP"
    assert reasoning_fields["confidence"] == 0.85


def test_parse_llm_action_no_target():
    """Test parsing action that doesn't require target."""
    llm_output = """{
        "action_id": "move",
        "target_id": null,
        "reasoning": "Reposition",
        "confidence": 0.7
    }"""
    actions = create_mock_actions()
    action_input, reasoning_fields = parse_llm_action(llm_output, "actor_1", actions)

    assert action_input.action_id == "move"
    assert action_input.target_ids == []
    assert action_input.action_type == "quick"
    assert reasoning_fields["reasoning"] == "Reposition"
    assert reasoning_fields["confidence"] == 0.7


def test_parse_llm_action_missing_required_target():
    """Test parsing fails when action requires target but none provided."""
    llm_output = """{
        "action_id": "skirmish",
        "target_id": null,
        "confidence": 0.9
    }"""
    actions = create_mock_actions()
    with pytest.raises(ValueError, match="requires a target but none provided"):
        parse_llm_action(llm_output, "actor_1", actions)


def test_parse_llm_action_extra_target_ignored():
    """Test extra target_id for non-target action is ignored."""
    llm_output = """{
        "action_id": "move",
        "target_id": "enemy1",
        "confidence": 0.5
    }"""
    actions = create_mock_actions()
    action_input, reasoning_fields = parse_llm_action(llm_output, "actor_1", actions)
    # target_id should be ignored, not in target_ids
    assert action_input.target_ids == []
    assert reasoning_fields["confidence"] == 0.5


def test_parse_llm_action_fuzzy_match():
    """Test parsing with fuzzy matching of action_id."""
    llm_output = """{"action_id": "skirmis", "target_id": "enemy1"}"""
    actions = create_mock_actions()
    action_input, _ = parse_llm_action(llm_output, "actor_1", actions)
    assert action_input.action_id == "skirmish"


def test_parse_llm_action_malformed_json():
    """Test parsing malformed JSON with recoverable content."""
    llm_output = """Here's my choice:
    ```json
    {
        "action_id": "barrage",
        "target_id": "enemy1"
    }
    ```"""
    actions = create_mock_actions()
    action_input, _ = parse_llm_action(llm_output, "actor_1", actions)
    assert action_input.action_id == "barrage"


def test_parse_llm_action_no_json():
    """Test parsing fails when no JSON found."""
    llm_output = "I choose to attack!"
    actions = create_mock_actions()
    with pytest.raises(ValueError, match="No valid JSON found"):
        parse_llm_action(llm_output, "actor_1", actions)


def test_parse_llm_action_confidence_passthrough():
    """Test confidence value is parsed but not used in output (just validation)."""
    llm_output = """{
        "action_id": "skirmish",
        "target_id": "enemy1",
        "confidence": 0.42
    }"""
    actions = create_mock_actions()
    # Should not raise
    action_input, reasoning_fields = parse_llm_action(llm_output, "actor_1", actions)
    assert action_input.action_id == "skirmish"
    assert reasoning_fields["confidence"] == 0.42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
