"""Unit tests for narrative debrief generation."""

from unittest.mock import patch, MagicMock
from core.gm_toolkit.mission import MissionConfig
from llm.src.mission.debrief import (
    generate_debrief,
    MissionDebriefer,
    MissionDebrieferConfig,
    LLMBackend,
    CombatStats,
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


def create_test_stats() -> CombatStats:
    """Create test combat statistics."""
    return CombatStats(
        turns_taken=8,
        damage_dealt=2450,
        damage_received=1200,
        enemies_destroyed=3,
        near_deaths=1,
        objectives_completed=["Secure Alpha", "Secure Bravo"],
        kills_by_frame={"Everest": 2, "Barbarossa": 1},
        closest_call_hp=15,
        overkill_damage=300,
    )


# =============================================================================
# MissionDebriefer class tests
# =============================================================================


class TestMissionDebriefer:
    """Test the MissionDebriefer class."""

    def test_init_with_default_config(self):
        """Debriefer should initialize with default config."""
        debriefer = MissionDebriefer()
        assert debriefer.config.backend == LLMBackend.OLLAMA
        assert debriefer.config.model == "lancer-expert"
        assert debriefer.config.max_retries == 3
        assert debriefer.config.cache_enabled is True

    def test_init_with_custom_config(self):
        """Debriefer should accept custom configuration."""
        config = MissionDebrieferConfig(
            backend=LLMBackend.MOCK,
            model="test-model",
            max_retries=5,
            cache_enabled=False,
        )
        debriefer = MissionDebriefer(config)
        assert debriefer.config.backend == LLMBackend.MOCK
        assert debriefer.config.model == "test-model"
        assert debriefer.config.max_retries == 5
        assert debriefer.config.cache_enabled is False

    def test_mock_backend_generates_debrief(self):
        """Mock backend should return a canned debrief."""
        config = MissionDebrieferConfig(backend=LLMBackend.MOCK)
        debriefer = MissionDebriefer(config)
        mission = create_test_mission()
        outcome = "success"
        stats = create_test_stats()

        debrief = debriefer.generate_debrief(mission, outcome, stats)

        assert isinstance(debrief, str)
        assert len(debrief) > 50
        # Should contain typical debrief elements
        assert "OPERATION" in debrief or "AFTER‑ACTION" in debrief

    @patch("llm.src.mission.debrief.MissionDebriefer._call_llm")
    def test_generate_debrief_success(self, mock_call_llm):
        """Generate debrief with mocked LLM call."""
        mock_debrief = (
            "OPERATION GLASS HAMMER – AFTER‑ACTION REPORT\n\n"
            "The enemy forces have been neutralized and all primary objectives secured. "
            "Union command commends your performance in the urban theater. "
            "Your mech sustained moderate damage but remains operational for future deployments.\n\n"
            "Salvage teams recovered valuable components from the wreckage, adding to your reserves. "
            "The success of this operation strengthens Union's position in the sector. "
            "Expect follow‑up missions to capitalize on this victory."
        )
        mock_call_llm.return_value = mock_debrief

        config = MissionDebrieferConfig(backend=LLMBackend.MOCK)
        debriefer = MissionDebriefer(config)
        mission = create_test_mission()
        outcome = "success"
        stats = create_test_stats()

        debrief = debriefer.generate_debrief(mission, outcome, stats)

        assert debrief == mock_debrief
        mock_call_llm.assert_called_once()

    @patch("llm.src.mission.debrief.MissionDebriefer._call_llm")
    def test_caching_works(self, mock_call_llm):
        """Debriefs should be cached for identical inputs."""
        mock_debrief = (
            "OPERATION GLASS HAMMER – AFTER‑ACTION REPORT\n\n"
            "The enemy forces have been neutralized and all primary objectives secured."
        )
        mock_call_llm.return_value = mock_debrief

        config = MissionDebrieferConfig(backend=LLMBackend.MOCK, cache_enabled=True)
        debriefer = MissionDebriefer(config)
        mission = create_test_mission()
        outcome = "success"
        stats = create_test_stats()

        # First call
        debrief1 = debriefer.generate_debrief(mission, outcome, stats)
        assert debrief1 == mock_debrief
        assert mock_call_llm.call_count == 1

        # Second call with same inputs - should use cache
        debrief2 = debriefer.generate_debrief(mission, outcome, stats)
        assert debrief2 == mock_debrief
        assert mock_call_llm.call_count == 1  # No additional call

        # Different outcome - should call again
        debrief3 = debriefer.generate_debrief(mission, "failure", stats)
        assert debrief3 == mock_debrief
        assert mock_call_llm.call_count == 2

    @patch("llm.src.mission.debrief.MissionDebriefer._call_llm")
    def test_cache_disabled(self, mock_call_llm):
        """When cache disabled, each call should hit LLM."""
        mock_call_llm.return_value = (
            "OPERATION GLASS HAMMER – AFTER‑ACTION REPORT\n\n"
            "The enemy forces have been neutralized and all primary objectives secured."
        )

        config = MissionDebrieferConfig(backend=LLMBackend.MOCK, cache_enabled=False)
        debriefer = MissionDebriefer(config)
        mission = create_test_mission()
        outcome = "success"
        stats = create_test_stats()

        debriefer.generate_debrief(mission, outcome, stats)
        debriefer.generate_debrief(mission, outcome, stats)

        assert mock_call_llm.call_count == 2  # Called twice

    @patch("llm.src.mission.debrief.MissionDebriefer._call_llm")
    def test_retry_logic_on_failure(self, mock_call_llm):
        """Should retry on LLM failure."""
        mock_call_llm.side_effect = [
            Exception("First attempt failed"),
            "OPERATION GLASS HAMMER – AFTER‑ACTION REPORT\n\n"
            "The enemy forces have been neutralized and all primary objectives secured.",
        ]

        config = MissionDebrieferConfig(backend=LLMBackend.MOCK, max_retries=3)
        debriefer = MissionDebriefer(config)
        mission = create_test_mission()
        outcome = "success"
        stats = create_test_stats()

        debrief = debriefer.generate_debrief(mission, outcome, stats)

        assert "OPERATION GLASS HAMMER" in debrief
        assert mock_call_llm.call_count == 2

    @patch("llm.src.mission.debrief.MissionDebriefer._call_llm")
    def test_all_retries_fail_uses_fallback(self, mock_call_llm):
        """When all retries fail, use fallback debrief."""
        mock_call_llm.side_effect = Exception("LLM unavailable")

        config = MissionDebrieferConfig(backend=LLMBackend.MOCK, max_retries=2)
        debriefer = MissionDebriefer(config)
        mission = create_test_mission()
        outcome = "success"
        stats = create_test_stats()

        debrief = debriefer.generate_debrief(mission, outcome, stats)

        assert isinstance(debrief, str)
        assert len(debrief) > 0
        assert mission.name in debrief
        assert mission.sitrep in debrief or "operation" in debrief

    @patch("llm.src.mission.debrief.MissionDebriefer._call_llm")
    def test_empty_debrief_triggers_fallback(self, mock_call_llm):
        """Empty or very short LLM response should trigger fallback."""
        mock_call_llm.return_value = ""  # Empty response

        config = MissionDebrieferConfig(backend=LLMBackend.MOCK)
        debriefer = MissionDebriefer(config)
        mission = create_test_mission()
        outcome = "success"
        stats = create_test_stats()

        debrief = debriefer.generate_debrief(mission, outcome, stats)

        # Should use fallback
        assert mission.name in debrief
        assert len(debrief) > 0

    def test_build_debrief_prompt_includes_all_data(self):
        """Prompt should contain mission, outcome, and statistics."""
        config = MissionDebrieferConfig(backend=LLMBackend.MOCK)
        debriefer = MissionDebriefer(config)
        mission = create_test_mission()
        outcome = "success"
        stats = create_test_stats()

        prompt = debriefer._build_debrief_prompt(mission, outcome, stats)

        # Check that mission details appear in prompt
        assert mission.name in prompt
        assert mission.sitrep in prompt
        assert mission.terrain in prompt
        assert str(mission.enemy_count) in prompt
        # Outcome
        assert outcome.upper() in prompt
        # Stats
        assert str(stats.turns_taken) in prompt
        assert str(stats.damage_dealt) in prompt
        assert str(stats.enemies_destroyed) in prompt
        # Should contain JSON sections
        assert "```json" in prompt
        assert "Mission Parameters" in prompt
        assert "Combat Statistics" in prompt

    def test_cache_key_generation(self):
        """Cache key should be deterministic and unique."""
        config = MissionDebrieferConfig()
        debriefer = MissionDebriefer(config)
        mission1 = create_test_mission()
        mission2 = create_test_mission()
        outcome = "success"
        stats1 = create_test_stats()
        stats2 = create_test_stats()

        key1 = debriefer._generate_cache_key(mission1, outcome, stats1)
        key2 = debriefer._generate_cache_key(mission2, outcome, stats2)

        # Same inputs -> same key
        assert key1 == key2

        # Different mission -> different key
        mission3 = mission1.model_copy(update={"id": "different"})
        key3 = debriefer._generate_cache_key(mission3, outcome, stats1)
        assert key1 != key3

        # Different outcome -> different key
        key4 = debriefer._generate_cache_key(mission1, "failure", stats1)
        assert key1 != key4

        # Different stats -> different key
        stats3 = stats1.model_copy(update={"turns_taken": 999})
        key5 = debriefer._generate_cache_key(mission1, outcome, stats3)
        assert key1 != key5


# =============================================================================
# Public function tests
# =============================================================================


@patch("llm.src.mission.debrief.MissionDebriefer")
def test_generate_debrief_function(mock_debriefer_class):
    """Test the public generate_debrief function."""
    mock_debriefer = MagicMock()
    mock_debriefer.generate_debrief.return_value = "Generated debrief"
    mock_debriefer_class.return_value = mock_debriefer

    mission = create_test_mission()
    outcome = "success"
    stats = create_test_stats()
    debrief = generate_debrief(mission, outcome, stats)

    assert debrief == "Generated debrief"
    mock_debriefer_class.assert_called_once()
    mock_debriefer.generate_debrief.assert_called_once_with(mission, outcome, stats)
