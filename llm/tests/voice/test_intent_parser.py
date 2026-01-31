"""Unit tests for voice intent parser."""

import pytest
from unittest.mock import patch, MagicMock
from core.mech.combat_models import AvailableAction, ActionExecutionInput
from core.mech.grid import HexCoord, HexPosition
from core.mech.combat_state import MechCombatScenario
from llm.src.voice.intent_parser import (
    parse_voice_intent,
    VoiceIntentParser,
    VoiceIntentParserConfig,
    LLMBackend,
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
            requires_system=False,
            requires_path=False,
            max_targets=1,
            unavailable_reason=None,
        ),
        AvailableAction(
            action_id="move",
            action_name="Move",
            action_type="quick",
            is_available=True,
            requires_target=False,
            requires_weapon=False,
            requires_system=False,
            requires_path=True,
            max_targets=1,
            unavailable_reason=None,
        ),
        AvailableAction(
            action_id="tech_attack",
            action_name="Tech Attack",
            action_type="full",
            is_available=True,
            requires_target=True,
            requires_weapon=False,
            requires_system=False,
            requires_path=False,
            max_targets=1,
            unavailable_reason=None,
        ),
    ]


def create_mock_action_execution_input(
    action_id: str, target_id: str | None = None
) -> ActionExecutionInput:
    """Create a mock ActionExecutionInput for comparison."""
    return ActionExecutionInput(
        actor_id="player_1",
        action_id=action_id,
        action_type="quick",
        target_ids=[target_id] if target_id else [],
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


# =============================================================================
# Test cases
# =============================================================================


class TestVoiceIntentParser:
    """Test the VoiceIntentParser class."""

    def test_init_with_default_config(self):
        """Parser should initialize with default config."""
        parser = VoiceIntentParser()
        assert parser.config.backend == LLMBackend.OLLAMA
        assert parser.config.model == "lancer-expert"
        assert parser.config.max_retries == 3
        assert parser.config.fallback_to_random is True

    def test_init_with_custom_config(self):
        """Parser should accept custom configuration."""
        config = VoiceIntentParserConfig(
            backend=LLMBackend.MOCK,
            model="test-model",
            max_retries=5,
            fallback_to_random=False,
        )
        parser = VoiceIntentParser(config)
        assert parser.config.backend == LLMBackend.MOCK
        assert parser.config.model == "test-model"
        assert parser.config.max_retries == 5
        assert parser.config.fallback_to_random is False

    @patch("llm.src.voice.intent_parser.VoiceIntentParser._call_llm")
    def test_parse_successful_skirmish(self, mock_call_llm):
        """Parse a simple 'skirmish' command."""
        # Mock LLM response
        mock_call_llm.return_value = """{
            "action_id": "skirmish",
            "target_id": "enemy_1",
            "target_position": null,
            "weapon_id": "assault_rifle",
            "system_id": null,
            "confidence": 0.9,
            "reasoning": "Player wants to attack enemy_1 with rifle",
            "fallback_prompt": null
        }"""

        parser = VoiceIntentParser(VoiceIntentParserConfig(backend=LLMBackend.MOCK))
        available_actions = create_mock_actions()
        # We need a scenario and actor_id for parse method
        # Use mock scenario (empty)
        from core.mech.combat_state import MechCombatScenario

        scenario = MechCombatScenario(
            combatants=[],
            grapples=[],
            rounds=[],
            terrain=None,
            environment="standard",
            deployables={},
            sitrep_resolution=None,
            pending_decisions=[],
            objectives=[],
            mission_reserves=[],
        )
        actor_id = "player_1"

        action_input, confidence, fallback = parser.parse(
            "attack the enemy with my rifle",
            scenario,
            actor_id,
            available_actions,
        )

        assert action_input.action_id == "skirmish"
        assert action_input.target_ids == ["enemy_1"]
        assert confidence == 0.9
        assert fallback is None

    @patch("llm.src.voice.intent_parser.VoiceIntentParser._call_llm")
    def test_parse_with_position(self, mock_call_llm):
        """Parse a move command with target position."""
        mock_call_llm.return_value = """{
            "action_id": "move",
            "target_id": null,
            "target_position": {"q": 3, "r": 2},
            "weapon_id": null,
            "system_id": null,
            "confidence": 0.8,
            "reasoning": "Move to hex 3,2",
            "fallback_prompt": null
        }"""

        parser = VoiceIntentParser(VoiceIntentParserConfig(backend=LLMBackend.MOCK))
        available_actions = create_mock_actions()
        scenario = MechCombatScenario(
            combatants=[],
            grapples=[],
            rounds=[],
            terrain=None,
            environment="standard",
            deployables={},
            sitrep_resolution=None,
            pending_decisions=[],
            objectives=[],
            mission_reserves=[],
        )
        actor_id = "player_1"

        action_input, confidence, fallback = parser.parse(
            "move to position three two",
            scenario,
            actor_id,
            available_actions,
        )

        assert action_input.action_id == "move"
        assert action_input.target_position is not None
        assert isinstance(action_input.target_position, HexPosition)
        assert action_input.target_position.coord.q == 3
        assert action_input.target_position.coord.r == 2
        assert action_input.target_position.elevation == 0
        assert confidence == 0.8

    @patch("llm.src.voice.intent_parser.VoiceIntentParser._call_llm")
    def test_parse_low_confidence_fallback(self, mock_call_llm):
        """Low confidence should return fallback prompt."""
        mock_call_llm.return_value = """{
            "action_id": "skirmish",
            "target_id": "enemy_1",
            "target_position": null,
            "weapon_id": null,
            "system_id": null,
            "confidence": 0.4,
            "reasoning": "Unclear target",
            "fallback_prompt": "Which enemy do you want to attack?"
        }"""

        parser = VoiceIntentParser(VoiceIntentParserConfig(backend=LLMBackend.MOCK))
        available_actions = create_mock_actions()
        scenario = MechCombatScenario(
            combatants=[],
            grapples=[],
            rounds=[],
            terrain=None,
            environment="standard",
            deployables={},
            sitrep_resolution=None,
            pending_decisions=[],
            objectives=[],
            mission_reserves=[],
        )
        actor_id = "player_1"

        action_input, confidence, fallback = parser.parse(
            "attack",
            scenario,
            actor_id,
            available_actions,
        )

        assert action_input.action_id == "skirmish"
        assert confidence == 0.4
        assert fallback == "Which enemy do you want to attack?"

    def test_random_fallback_on_failure(self):
        """When all retries fail and fallback enabled, return random action."""
        parser = VoiceIntentParser(
            VoiceIntentParserConfig(
                backend=LLMBackend.MOCK,
                max_retries=1,
                fallback_to_random=True,
            )
        )
        # Mock _call_llm to raise exception
        with patch.object(parser, "_call_llm", side_effect=Exception("LLM error")):
            available_actions = create_mock_actions()
            scenario = MechCombatScenario(
                combatants=[],
                grapples=[],
                rounds=[],
                terrain=None,
                environment="standard",
                deployables={},
                sitrep_resolution=None,
                pending_decisions=[],
                objectives=[],
                mission_reserves=[],
            )
            actor_id = "player_1"

            action_input, confidence, fallback = parser.parse(
                "gibberish",
                scenario,
                actor_id,
                available_actions,
            )

            # Should return some action (randomly chosen)
            assert action_input.action_id in ["skirmish", "move", "tech_attack"]
            assert confidence == 0.0
            assert fallback is not None and "Random fallback" in fallback

    def test_error_when_no_fallback(self):
        """When fallback disabled, raise error after max retries."""
        parser = VoiceIntentParser(
            VoiceIntentParserConfig(
                backend=LLMBackend.MOCK,
                max_retries=1,
                fallback_to_random=False,
            )
        )
        with patch.object(parser, "_call_llm", side_effect=Exception("LLM error")):
            available_actions = create_mock_actions()
            scenario = MechCombatScenario(
                combatants=[],
                grapples=[],
                rounds=[],
                terrain=None,
                environment="standard",
                deployables={},
                sitrep_resolution=None,
                pending_decisions=[],
                objectives=[],
                mission_reserves=[],
            )
            actor_id = "player_1"

            with pytest.raises(ValueError, match="Failed to parse voice intent"):
                parser.parse(
                    "gibberish",
                    scenario,
                    actor_id,
                    available_actions,
                )


class TestParseVoiceIntentFunction:
    """Test the public parse_voice_intent function."""

    @patch("llm.src.voice.intent_parser.VoiceIntentParser")
    def test_function_without_context(self, mock_parser_class):
        """Function should work without scenario/actor_id."""
        mock_parser = MagicMock()
        mock_parser.parse.return_value = (
            create_mock_action_execution_input("skirmish", "enemy_1"),
            0.9,
            None,
        )
        mock_parser_class.return_value = mock_parser

        available_actions = create_mock_actions()
        result = parse_voice_intent(
            "attack enemy",
            available_actions,
            scenario=None,
            actor_id=None,
        )

        assert result.action_id == "skirmish"
        assert result.target_ids == ["enemy_1"]
        # Should have called parse with mock scenario
        assert mock_parser.parse.called

    @patch("llm.src.voice.intent_parser.VoiceIntentParser")
    def test_function_with_context(self, mock_parser_class):
        """Function should pass scenario/actor_id when provided."""
        mock_parser = MagicMock()
        mock_parser.parse.return_value = (
            create_mock_action_execution_input("move"),
            0.8,
            None,
        )
        mock_parser_class.return_value = mock_parser

        available_actions = create_mock_actions()
        from core.mech.combat_state import MechCombatScenario

        scenario = MechCombatScenario(
            combatants=[],
            grapples=[],
            rounds=[],
            terrain=None,
            environment="standard",
            deployables={},
            sitrep_resolution=None,
            pending_decisions=[],
            objectives=[],
            mission_reserves=[],
        )
        actor_id = "player_1"

        result = parse_voice_intent(
            "move forward",
            available_actions,
            scenario=scenario,
            actor_id=actor_id,
        )

        assert result.action_id == "move"
        # Verify parse called with correct arguments
        mock_parser.parse.assert_called_with(
            "move forward",
            scenario,
            actor_id,
            available_actions,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
