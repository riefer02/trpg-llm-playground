"""Unit tests for narrative briefing generation."""

from unittest.mock import patch, MagicMock
from core.gm_toolkit.mission import MissionConfig
from llm.src.mission.narrative import (
    generate_briefing,
    MissionNarrator,
    MissionNarratorConfig,
    LLMBackend,
)


# =============================================================================
# Test data
# =============================================================================


def create_test_mission() -> MissionConfig:
    """Create a test mission configuration."""
    return MissionConfig(
        id="test-mission-001",
        name="Operation Glass Hammer",
        difficulty=2,
        sitrep="control",
        terrain="urban",
        enemy_count=4,
        description="Secure control points in urban ruins",
        briefing="",  # Will be generated
        objectives=["Secure Alpha", "Secure Bravo", "Hold for 5 minutes"],
        enemy_intel="Two GMS Everests and two HA regulars",
        map_preview_url=None,
        enemy_force_preview=None,
    )


# =============================================================================
# MissionNarrator class tests
# =============================================================================


class TestMissionNarrator:
    """Test the MissionNarrator class."""

    def test_init_with_default_config(self):
        """Narrator should initialize with default config."""
        narrator = MissionNarrator()
        assert narrator.config.backend == LLMBackend.OLLAMA
        assert narrator.config.model == "lancer-expert"
        assert narrator.config.max_retries == 3
        assert narrator.config.cache_enabled is True

    def test_init_with_custom_config(self):
        """Narrator should accept custom configuration."""
        config = MissionNarratorConfig(
            backend=LLMBackend.MOCK,
            model="test-model",
            max_retries=5,
            cache_enabled=False,
        )
        narrator = MissionNarrator(config)
        assert narrator.config.backend == LLMBackend.MOCK
        assert narrator.config.model == "test-model"
        assert narrator.config.max_retries == 5
        assert narrator.config.cache_enabled is False

    def test_mock_backend_generates_briefing(self):
        """Mock backend should return a canned briefing."""
        config = MissionNarratorConfig(backend=LLMBackend.MOCK)
        narrator = MissionNarrator(config)
        mission = create_test_mission()

        briefing = narrator.generate_briefing(mission)

        assert isinstance(briefing, str)
        assert len(briefing) > 50
        # Should contain typical briefing elements
        assert "SITUATION" in briefing or "Harrison" in briefing

    @patch("llm.src.mission.narrative.MissionNarrator._call_llm")
    def test_generate_briefing_success(self, mock_call_llm):
        """Generate briefing with mocked LLM call."""
        mock_briefing = (
            "SITUATION: Harrison Armory has established an illegal forward operations base "
            "in the ruins of New Mumbai. Local resistance cells have requested assistance.\n\n"
            "MISSION: Infiltrate the urban zone, secure control points, and hold them "
            "until reinforcements arrive. Deny the enemy use of strategic positions.\n\n"
            "THREATS: Intel indicates a mixed force of HA regulars. Expect heavy resistance.\n\n"
            "EXTRACTION: Once all points are secured, a Union dropship will extract you.\n\n"
            "TERRAIN: The ruins provide ample cover but also conceal enemy positions."
        )
        mock_call_llm.return_value = mock_briefing

        config = MissionNarratorConfig(backend=LLMBackend.MOCK)
        narrator = MissionNarrator(config)
        mission = create_test_mission()

        briefing = narrator.generate_briefing(mission)

        assert briefing == mock_briefing
        mock_call_llm.assert_called_once()

    @patch("llm.src.mission.narrative.MissionNarrator._call_llm")
    def test_caching_works(self, mock_call_llm):
        """Briefings should be cached for identical missions."""
        mock_briefing = (
            "SITUATION: Harrison Armory has established an illegal forward operations base "
            "in the ruins of New Mumbai, violating Union demilitarization treaties. "
            "Local resistance cells have requested assistance."
        )
        mock_call_llm.return_value = mock_briefing

        config = MissionNarratorConfig(backend=LLMBackend.MOCK, cache_enabled=True)
        narrator = MissionNarrator(config)
        mission = create_test_mission()

        # First call
        briefing1 = narrator.generate_briefing(mission)
        assert briefing1 == mock_briefing
        assert mock_call_llm.call_count == 1

        # Second call with same mission - should use cache
        briefing2 = narrator.generate_briefing(mission)
        assert briefing2 == mock_briefing
        assert mock_call_llm.call_count == 1  # No additional call

        # Different mission - should call again
        mission2 = mission.model_copy(update={"id": "different-id"})
        briefing3 = narrator.generate_briefing(mission2)
        assert briefing3 == mock_briefing
        assert mock_call_llm.call_count == 2

    @patch("llm.src.mission.narrative.MissionNarrator._call_llm")
    def test_cache_disabled(self, mock_call_llm):
        """When cache disabled, each call should hit LLM."""
        mock_call_llm.return_value = (
            "SITUATION: Harrison Armory has established an illegal forward operations base "
            "in the ruins of New Mumbai, violating Union demilitarization treaties. "
            "Local resistance cells have requested assistance."
        )

        config = MissionNarratorConfig(backend=LLMBackend.MOCK, cache_enabled=False)
        narrator = MissionNarrator(config)
        mission = create_test_mission()

        narrator.generate_briefing(mission)
        narrator.generate_briefing(mission)

        assert mock_call_llm.call_count == 2  # Called twice

    @patch("llm.src.mission.narrative.MissionNarrator._call_llm")
    def test_retry_logic_on_failure(self, mock_call_llm):
        """Should retry on LLM failure."""
        mock_call_llm.side_effect = [
            Exception("First attempt failed"),
            "SITUATION: Harrison Armory has established an illegal forward operations base "
            "in the ruins of New Mumbai, violating Union demilitarization treaties. "
            "Local resistance cells have requested assistance.",
        ]

        config = MissionNarratorConfig(backend=LLMBackend.MOCK, max_retries=3)
        narrator = MissionNarrator(config)
        mission = create_test_mission()

        briefing = narrator.generate_briefing(mission)

        assert (
            briefing
            == "SITUATION: Harrison Armory has established an illegal forward operations base in the ruins of New Mumbai, violating Union demilitarization treaties. Local resistance cells have requested assistance."
        )
        assert mock_call_llm.call_count == 2

    @patch("llm.src.mission.narrative.MissionNarrator._call_llm")
    def test_all_retries_fail_uses_fallback(self, mock_call_llm):
        """When all retries fail, use fallback briefing."""
        mock_call_llm.side_effect = Exception("LLM unavailable")

        config = MissionNarratorConfig(backend=LLMBackend.MOCK, max_retries=2)
        narrator = MissionNarrator(config)
        mission = create_test_mission()

        briefing = narrator.generate_briefing(mission)

        assert isinstance(briefing, str)
        assert len(briefing) > 0
        assert mission.name in briefing
        assert mission.sitrep in briefing or "operation" in briefing

    @patch("llm.src.mission.narrative.MissionNarrator._call_llm")
    def test_empty_briefing_triggers_fallback(self, mock_call_llm):
        """Empty or very short LLM response should trigger fallback."""
        mock_call_llm.return_value = ""  # Empty response

        config = MissionNarratorConfig(backend=LLMBackend.MOCK)
        narrator = MissionNarrator(config)
        mission = create_test_mission()

        briefing = narrator.generate_briefing(mission)

        # Should use fallback
        assert mission.name in briefing
        assert len(briefing) > 0

    def test_build_briefing_prompt_includes_mission_data(self):
        """Prompt should contain mission parameters."""
        config = MissionNarratorConfig(backend=LLMBackend.MOCK)
        narrator = MissionNarrator(config)
        mission = create_test_mission()

        prompt = narrator._build_briefing_prompt(mission)

        # Check that mission details appear in prompt
        assert mission.name in prompt
        assert mission.sitrep in prompt
        assert mission.terrain in prompt
        assert str(mission.enemy_count) in prompt
        # Should contain JSON
        assert "```json" in prompt
        assert "Mission Parameters" in prompt

    def test_cache_key_generation(self):
        """Cache key should be deterministic and unique."""
        config = MissionNarratorConfig()
        narrator = MissionNarrator(config)
        mission1 = create_test_mission()
        mission2 = create_test_mission()

        key1 = narrator._generate_cache_key(mission1)
        key2 = narrator._generate_cache_key(mission2)

        # Same mission data -> same key
        assert key1 == key2

        # Different mission -> different key
        mission3 = mission1.model_copy(update={"id": "different"})
        key3 = narrator._generate_cache_key(mission3)
        assert key1 != key3


# =============================================================================
# Public function tests
# =============================================================================


@patch("llm.src.mission.narrative.MissionNarrator")
def test_generate_briefing_function(mock_narrator_class):
    """Test the public generate_briefing function."""
    mock_narrator = MagicMock()
    mock_narrator.generate_briefing.return_value = "Generated briefing"
    mock_narrator_class.return_value = mock_narrator

    mission = create_test_mission()
    briefing = generate_briefing(mission)

    assert briefing == "Generated briefing"
    mock_narrator_class.assert_called_once()
    mock_narrator.generate_briefing.assert_called_once_with(mission)
