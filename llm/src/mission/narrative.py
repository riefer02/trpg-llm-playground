"""Narrative briefing generation for Lancer missions.

Provides LLM-generated mission briefings based on mission configuration.
"""

import enum
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict
from core.gm_toolkit.mission import MissionConfig

logger = logging.getLogger(__name__)


class LLMBackend(enum.Enum):
    """Supported LLM backends for narrative generation."""

    OLLAMA = "ollama"
    CLAUDE = "claude"
    MOCK = "mock"  # For testing without real LLM


class MissionNarratorConfig:
    """Configuration for the mission narrator."""

    def __init__(
        self,
        backend: LLMBackend = LLMBackend.OLLAMA,
        model: str = "lancer-expert",
        max_retries: int = 3,
        enable_logging: bool = True,
        cache_enabled: bool = True,
        **backend_kwargs,
    ):
        self.backend = backend
        self.model = model
        self.max_retries = max_retries
        self.enable_logging = enable_logging
        self.cache_enabled = cache_enabled
        self.backend_kwargs = backend_kwargs


class MissionNarrator:
    """Generate narrative briefings for mission configurations."""

    def __init__(self, config: Optional[MissionNarratorConfig] = None):
        self.config = config or MissionNarratorConfig()
        self._setup_backend()
        self._cache: Dict[str, str] = {}  # Simple in-memory cache

    def _setup_backend(self):
        """Initialize the LLM backend based on config."""
        if self.config.backend == LLMBackend.OLLAMA:
            try:
                import ollama

                self._ollama_client = ollama
                logger.info("Ollama backend initialized for mission narration")
            except ImportError:
                raise ImportError(
                    "Ollama Python package not installed. "
                    "Install with: pip install ollama"
                )
        elif self.config.backend == LLMBackend.CLAUDE:
            try:
                import anthropic

                self._anthropic_client = anthropic
                logger.info("Claude backend initialized for mission narration")
            except ImportError:
                raise ImportError(
                    "Anthropic Python package not installed. "
                    "Install with: pip install anthropic"
                )
        elif self.config.backend == LLMBackend.MOCK:
            logger.info("Mock backend initialized for mission narration (for testing)")
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM backend with the prompt."""
        if self.config.backend == LLMBackend.OLLAMA:
            return self._call_ollama(prompt)
        elif self.config.backend == LLMBackend.CLAUDE:
            return self._call_claude(prompt)
        elif self.config.backend == LLMBackend.MOCK:
            return self._call_mock(prompt)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            response = self._ollama_client.chat(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a military intelligence officer generating mission briefings for Lancer pilots.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options=self.config.backend_kwargs.get("options", {}),
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        api_key = self.config.backend_kwargs.get("api_key")
        if not api_key:
            raise ValueError("Claude API key required in backend_kwargs")

        client = self._anthropic_client.Anthropic(api_key=api_key)
        try:
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.backend_kwargs.get("max_tokens", 1000),
                temperature=self.config.backend_kwargs.get("temperature", 0.3),
                system="You are a military intelligence officer generating mission briefings for Lancer pilots.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _call_mock(self, prompt: str) -> str:
        """Mock LLM response for testing."""
        # Return a canned briefing for testing
        return (
            "SITUATION: Harrison Armory has established an illegal forward operations base "
            "in the ruins of New Mumbai, violating Union demilitarization treaties. "
            "Local resistance cells have requested assistance.\n\n"
            "MISSION: Infiltrate the urban zone, secure three control points, and hold them "
            "until reinforcements arrive. Deny the enemy use of these strategic positions.\n\n"
            "THREATS: Intel indicates a mixed force of HA regulars supported by two GMS Everests "
            "and a HA Barbarossa acting as mobile artillery. Expect heavy resistance.\n\n"
            "EXTRACTION: Once all points are secured for 5 minutes, a Union dropship will extract you.\n\n"
            "TERRAIN: The ruins provide ample cover but also conceal enemy positions. "
            "Watch for collapsing structures and HA automated turrets."
        )

    def _load_briefing_prompt(self) -> str:
        """Load the briefing prompt template from markdown file."""
        prompt_path = Path(__file__).parent / "prompts" / "briefing.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Briefing prompt not found at {prompt_path}")
        return prompt_path.read_text()

    def _build_briefing_prompt(self, mission: MissionConfig) -> str:
        """Build a complete briefing prompt with mission details."""
        system_prompt = self._load_briefing_prompt()

        # Format mission data for inclusion
        mission_dict = mission.model_dump(mode="json", exclude={"enemy_force_preview"})
        if mission.enemy_force_preview:
            mission_dict["enemy_force_preview"] = (
                mission.enemy_force_preview.model_dump(mode="json")
            )

        mission_json = json.dumps(mission_dict, indent=2)

        prompt = f"""{system_prompt}

## Mission Parameters

Below is the mission configuration as JSON. Use these parameters to generate the briefing.

```json
{mission_json}
```

## Your Task

Generate a 2-3 paragraph mission briefing based on the parameters above.

Remember:
- Tone: Military sci-fi, Lancer universe authentic
- Structure: Situation, mission objective, known threats, extraction plan, terrain notes
- Length: 2-3 paragraphs total
- Format: Plain text only, no markdown or section headers

Generate ONLY the briefing text."""
        return prompt

    def _generate_cache_key(self, mission: MissionConfig) -> str:
        """Generate a cache key from mission parameters."""
        # Create a deterministic string representation of mission
        mission_dict = mission.model_dump(mode="json", exclude={"enemy_force_preview"})
        if mission.enemy_force_preview:
            mission_dict["enemy_force_preview"] = (
                mission.enemy_force_preview.model_dump(mode="json")
            )

        mission_str = json.dumps(mission_dict, sort_keys=True)
        return hashlib.md5(mission_str.encode()).hexdigest()

    def generate_briefing(self, mission: MissionConfig) -> str:
        """Generate a narrative briefing for the given mission.

        Args:
            mission: Mission configuration

        Returns:
            2-3 paragraph briefing text
        """
        # Check cache first
        if self.config.cache_enabled:
            cache_key = self._generate_cache_key(mission)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for mission {mission.id}")
                return self._cache[cache_key]

        # Build prompt
        prompt = self._build_briefing_prompt(mission)

        # Call LLM with retry logic
        for attempt in range(self.config.max_retries):
            try:
                briefing = self._call_llm(prompt).strip()
                # Basic validation - ensure it's not empty
                if briefing and len(briefing) > 50:
                    # Cache result
                    if self.config.cache_enabled:
                        cache_key = self._generate_cache_key(mission)
                        self._cache[cache_key] = briefing
                    return briefing
                else:
                    logger.warning(
                        f"LLM returned invalid briefing (too short) on attempt {attempt + 1}"
                    )
            except Exception as e:
                logger.warning(f"LLM call failed on attempt {attempt + 1}: {e}")
                # Continue to next attempt, or fallback after loop

        # Fallback to a generic briefing if all retries fail
        fallback = (
            f"Mission {mission.name}: {mission.sitrep} operation in {mission.terrain} terrain. "
            f"Engage and neutralize {mission.enemy_count} enemy units. Exercise extreme caution."
        )
        logger.error(f"Using fallback briefing for mission {mission.id}")
        return fallback


# Public function matching the user story signature
def generate_briefing(mission: MissionConfig) -> str:
    """Generate a narrative briefing for the given mission.

    This is the public function matching the user story acceptance criteria.

    Args:
        mission: Mission configuration

    Returns:
        2-3 paragraph briefing text
    """
    narrator = MissionNarrator()
    return narrator.generate_briefing(mission)
