"""Narrative debrief generation for Lancer missions.

Provides LLM-generated mission debriefs based on mission configuration, outcome, and combat statistics.
"""

import enum
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel, Field

from core.gm_toolkit.mission import MissionConfig
from core.shared.scenario import MissionOutcomeType

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Types for combat statistics
# -----------------------------------------------------------------------------


class CombatStats(BaseModel):
    """Combat statistics for debrief generation.

    This model captures key metrics from a mission that can be woven into
    narrative debriefs. All fields are optional to support partial data.
    """

    # Basic engagement metrics
    turns_taken: int = Field(default=0, ge=0, description="Number of turns taken")
    damage_dealt: int = Field(
        default=0, ge=0, description="Total damage dealt by pilot"
    )
    damage_received: int = Field(default=0, ge=0, description="Total damage received")
    enemies_destroyed: int = Field(
        default=0, ge=0, description="Number of enemy mechs destroyed"
    )

    # Notable events
    near_deaths: int = Field(
        default=0, ge=0, description="Times pilot's mech dropped below 25% HP"
    )
    objectives_completed: list[str] = Field(
        default_factory=list,
        description="List of objective IDs or descriptions that were completed",
    )

    # Optional detailed stats
    kills_by_frame: Dict[str, int] = Field(
        default_factory=dict, description="Map of frame type to number destroyed"
    )
    closest_call_hp: Optional[int] = Field(
        default=None, ge=0, description="Lowest HP reached during mission (if recorded)"
    )
    overkill_damage: Optional[int] = Field(
        default=None,
        ge=0,
        description="Excess damage dealt beyond what was needed to destroy targets",
    )


# -----------------------------------------------------------------------------
# LLM Backend Configuration
# -----------------------------------------------------------------------------


class LLMBackend(enum.Enum):
    """Supported LLM backends for debrief generation."""

    OLLAMA = "ollama"
    CLAUDE = "claude"
    MOCK = "mock"  # For testing without real LLM


class MissionDebrieferConfig:
    """Configuration for the mission debriefer."""

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


# -----------------------------------------------------------------------------
# Mission Debriefer
# -----------------------------------------------------------------------------


class MissionDebriefer:
    """Generate narrative debriefs for mission outcomes."""

    def __init__(self, config: Optional[MissionDebrieferConfig] = None):
        self.config = config or MissionDebrieferConfig()
        self._setup_backend()
        self._cache: Dict[str, str] = {}  # Simple in-memory cache

    def _setup_backend(self):
        """Initialize the LLM backend based on config."""
        if self.config.backend == LLMBackend.OLLAMA:
            try:
                import ollama

                self._ollama_client = ollama
                logger.info("Ollama backend initialized for mission debriefing")
            except ImportError:
                raise ImportError(
                    "Ollama Python package not installed. "
                    "Install with: pip install ollama"
                )
        elif self.config.backend == LLMBackend.CLAUDE:
            try:
                import anthropic

                self._anthropic_client = anthropic
                logger.info("Claude backend initialized for mission debriefing")
            except ImportError:
                raise ImportError(
                    "Anthropic Python package not installed. "
                    "Install with: pip install anthropic"
                )
        elif self.config.backend == LLMBackend.MOCK:
            logger.info("Mock backend initialized for mission debriefing (for testing)")
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
                        "content": "You are a military intelligence officer generating mission debriefs for Lancer pilots.",
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
                system="You are a military intelligence officer generating mission debriefs for Lancer pilots.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _call_mock(self, prompt: str) -> str:
        """Mock LLM response for testing."""
        # Return a canned debrief for testing
        return (
            "OPERATION GLASS HAMMER – AFTER‑ACTION REPORT\n\n"
            "The enemy forces have been neutralized and all primary objectives secured. "
            "Union command commends your performance in the urban theater. "
            "Your mech sustained moderate damage but remains operational for future deployments.\n\n"
            "Salvage teams recovered valuable components from the wreckage, adding to your reserves. "
            "The success of this operation strengthens Union's position in the sector. "
            "Expect follow‑up missions to capitalize on this victory."
        )

    def _load_debrief_prompt(self) -> str:
        """Load the debrief prompt template from markdown file."""
        prompt_path = Path(__file__).parent / "prompts" / "debrief.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Debrief prompt not found at {prompt_path}")
        return prompt_path.read_text()

    def _build_debrief_prompt(
        self,
        mission: MissionConfig,
        outcome: MissionOutcomeType,
        stats: CombatStats,
    ) -> str:
        """Build a complete debrief prompt with mission details."""
        system_prompt = self._load_debrief_prompt()

        # Format mission data
        mission_dict = mission.model_dump(mode="json", exclude={"enemy_force_preview"})
        if mission.enemy_force_preview:
            mission_dict["enemy_force_preview"] = (
                mission.enemy_force_preview.model_dump(mode="json")
            )

        mission_json = json.dumps(mission_dict, indent=2)
        stats_json = json.dumps(stats.model_dump(mode="json"), indent=2)

        prompt = f"""{system_prompt}

## Mission Parameters

Below is the mission configuration as JSON.

```json
{mission_json}
```

## Mission Outcome

The mission resulted in: **{outcome.upper()}**

## Combat Statistics

Below are the combat statistics recorded during the mission.

```json
{stats_json}
```

## Your Task

Generate a 2-3 paragraph mission debrief (epilogue) based on the parameters above.

Remember:
- Tone: Military sci‑fi, Lancer universe authentic
- Outcome: Reflect the mission result ({outcome}) in tone and content
- Incorporate statistics: Mention kills, near‑deaths, objectives, damage, etc. where relevant
- Length: 2-3 paragraphs total
- Format: Plain text only, no markdown or section headers

Generate ONLY the debrief text."""
        return prompt

    def _generate_cache_key(
        self,
        mission: MissionConfig,
        outcome: MissionOutcomeType,
        stats: CombatStats,
    ) -> str:
        """Generate a cache key from mission, outcome, and stats."""
        # Create deterministic string representations
        mission_dict = mission.model_dump(mode="json", exclude={"enemy_force_preview"})
        if mission.enemy_force_preview:
            mission_dict["enemy_force_preview"] = (
                mission.enemy_force_preview.model_dump(mode="json")
            )
        mission_str = json.dumps(mission_dict, sort_keys=True)
        stats_str = json.dumps(stats.model_dump(mode="json"), sort_keys=True)
        combined = f"{mission_str}|{outcome}|{stats_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def generate_debrief(
        self,
        mission: MissionConfig,
        outcome: MissionOutcomeType,
        stats: CombatStats,
    ) -> str:
        """Generate a narrative debrief for the given mission outcome.

        Args:
            mission: Mission configuration
            outcome: Mission outcome (success, partial, failure, catastrophic)
            stats: Combat statistics to incorporate into the narrative

        Returns:
            2-3 paragraph debrief text
        """
        # Check cache first
        if self.config.cache_enabled:
            cache_key = self._generate_cache_key(mission, outcome, stats)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for mission {mission.id}")
                return self._cache[cache_key]

        # Build prompt
        prompt = self._build_debrief_prompt(mission, outcome, stats)

        # Call LLM with retry logic
        for attempt in range(self.config.max_retries):
            try:
                debrief = self._call_llm(prompt).strip()
                # Basic validation - ensure it's not empty
                if debrief and len(debrief) > 50:
                    # Cache result
                    if self.config.cache_enabled:
                        cache_key = self._generate_cache_key(mission, outcome, stats)
                        self._cache[cache_key] = debrief
                    return debrief
                else:
                    logger.warning(
                        f"LLM returned invalid debrief (too short) on attempt {attempt + 1}"
                    )
            except Exception as e:
                logger.warning(f"LLM call failed on attempt {attempt + 1}: {e}")
                # Continue to next attempt, or fallback after loop

        # Fallback to a generic debrief if all retries fail
        fallback = (
            f"Mission {mission.name}: {mission.sitrep} operation in {mission.terrain} terrain. "
            f"Outcome: {outcome}. The mission has concluded."
        )
        logger.error(f"Using fallback debrief for mission {mission.id}")
        return fallback


# -----------------------------------------------------------------------------
# Public function matching user story acceptance criteria
# -----------------------------------------------------------------------------


def generate_debrief(
    mission: MissionConfig,
    outcome: MissionOutcomeType,
    stats: CombatStats,
) -> str:
    """Generate a narrative debrief for the given mission.

    This is the public function matching the user story acceptance criteria.

    Args:
        mission: Mission configuration
        outcome: Mission outcome (success, partial, failure, catastrophic)
        stats: Combat statistics to incorporate into the narrative

    Returns:
        2-3 paragraph debrief text
    """
    debriefer = MissionDebriefer()
    return debriefer.generate_debrief(mission, outcome, stats)
