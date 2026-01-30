"""Unit tests for LLM tactician."""

import pytest
from unittest.mock import patch, MagicMock

from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
)
from core.mech.action_economy import ActionEconomyState

from llm.src.tactician import (
    LLMBackend,
    TacticianConfig,
    Tactician,
)


@pytest.fixture
def mock_scenario():
    """Create a minimal combat scenario with one NPC combatant."""
    npc = CombatantState(
        id="npc1",
        name="Test NPC",
        side="hostiles",
        kind="npc",
        stats=CombatStats(
            size="size_1",
            hp_max=20,
            evasion=10,
            e_defense=8,
            armor=0,
            speed=5,
            sensor_range=10,
            tech_attack=0,
            grit=0,
            engineering_skill=0,
        ),
        resources=CombatResources(
            hp_current=20,
            heat_current=0,
            heat_cap=6,
            structure_current=1,
            stress_current=0,
            repairs_remaining=1,
            burn_marked=0,
        ),
        ai_controlled=True,
        ai_type="llm",
        npc_role="striker",
    )
    return MechCombatScenario(combatants=[npc])


def test_tactician_initialization():
    """Test Tactician initialization with different backends."""
    # Mock backend
    config = TacticianConfig(backend=LLMBackend.MOCK)
    tactician = Tactician(config)
    assert tactician.config.backend == LLMBackend.MOCK

    # Ollama backend (requires ollama package)
    config = TacticianConfig(backend=LLMBackend.OLLAMA)
    with patch.dict("sys.modules", {"ollama": MagicMock()}):
        tactician = Tactician(config)
        assert tactician.config.backend == LLMBackend.OLLAMA

    # Claude backend (requires anthropic package)
    config = TacticianConfig(backend=LLMBackend.CLAUDE)
    with patch.dict("sys.modules", {"anthropic": MagicMock()}):
        tactician = Tactician(config)
        assert tactician.config.backend == LLMBackend.CLAUDE


def test_tactician_decide_action_mock(mock_scenario):
    """Test decide_action with mock backend."""
    config = TacticianConfig(backend=LLMBackend.MOCK, max_retries=1)
    tactician = Tactician(config)

    # Mock the LLM call to return a valid JSON matching available actions
    with patch.object(tactician, "_call_llm") as mock_call:
        mock_call.return_value = """{
            "action_id": "hide",
            "target_id": null,
            "confidence": 0.8,
            "reasoning": "Test reasoning"
        }"""

        action_input, reasoning_fields = tactician.decide_action(
            mock_scenario,
            "npc1",
            economy_state=ActionEconomyState().model_dump(mode="json"),
        )

        assert len(action_input) == 1
        assert action_input[0].actor_id == "npc1"
        assert action_input[0].action_id == "hide"
        assert action_input[0].target_ids == []
        assert reasoning_fields["reasoning"] == "Test reasoning"
        assert reasoning_fields["confidence"] == 0.8


def test_tactician_decide_action_retry_on_failure(mock_scenario):
    """Test retry logic when LLM calls fail."""
    config = TacticianConfig(backend=LLMBackend.MOCK, max_retries=3)
    tactician = Tactician(config)

    call_count = 0

    def failing_call(prompt):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Simulated LLM failure")
        return """{
            "action_id": "hide",
            "target_id": null,
            "confidence": 0.7
        }"""

    with patch.object(tactician, "_call_llm", side_effect=failing_call):
        action_input, reasoning_fields = tactician.decide_action(
            mock_scenario,
            "npc1",
            economy_state=ActionEconomyState().model_dump(mode="json"),
        )

        assert call_count == 3
        assert len(action_input) == 1
        assert action_input[0].action_id == "hide"
        assert reasoning_fields["confidence"] == 0.7


def test_tactician_fallback_to_random(mock_scenario):
    """Test fallback to random action when all retries fail."""
    config = TacticianConfig(
        backend=LLMBackend.MOCK,
        max_retries=2,
        fallback_to_random=True,
    )
    tactician = Tactician(config)

    with patch.object(tactician, "_call_llm", side_effect=Exception("LLM failed")):
        # Also mock get_available_actions to return something
        with patch("llm.src.tactician.tactician.get_available_actions") as mock_avail:
            mock_avail.return_value.full_actions = []
            mock_avail.return_value.quick_actions = [
                MagicMock(
                    action_id="move",
                    action_name="Move",
                    action_type="quick",
                    is_available=True,
                    requires_target=False,
                )
            ]

            action_input, reasoning_fields = tactician.decide_action(
                mock_scenario,
                "npc1",
                economy_state=ActionEconomyState().model_dump(mode="json"),
            )

            # Should still return an action (random fallback)
            assert len(action_input) == 1
            assert action_input[0].actor_id == "npc1"
            # The random fallback will create a basic ActionExecutionInput
            # with action_id from the available actions
            assert action_input[0].action_id == "move"
            # Random fallback provides empty reasoning fields
            assert (
                reasoning_fields["reasoning"] == "Random fallback action (LLM failed)"
            )
            assert reasoning_fields["confidence"] == 0.0


def test_tactician_no_fallback_raises(mock_scenario):
    """Test that exception is raised when fallback disabled and LLM fails."""
    config = TacticianConfig(
        backend=LLMBackend.MOCK,
        max_retries=1,
        fallback_to_random=False,
    )
    tactician = Tactician(config)

    with patch.object(tactician, "_call_llm", side_effect=Exception("LLM failed")):
        with pytest.raises(ValueError, match="Failed to decide action"):
            tactician.decide_action(
                mock_scenario,
                "npc1",
                economy_state=ActionEconomyState().model_dump(mode="json"),
            )


def test_tactician_with_role():
    """Test tactician configuration with NPC role."""
    config = TacticianConfig(backend=LLMBackend.MOCK, role="striker")
    tactician = Tactician(config)
    assert tactician.config.role == "striker"
