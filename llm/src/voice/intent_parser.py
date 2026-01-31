"""Voice intent parser for Lancer combat.

Parse natural language voice commands into validated combat actions.
"""

import json
import re
import logging
import time
import random
from typing import Optional, Tuple, List, Dict, Any
from difflib import get_close_matches
from enum import Enum

from core.mech.combat_state import MechCombatScenario
from core.mech.combat_models import (
    AvailableAction,
    ActionExecutionInput,
)
from core.mech.grid import HexCoord, HexPosition

try:
    from llm.src.tactician.action_parser import (
        extract_json_from_text,
        find_matching_action,
    )
    from llm.src.tactician.state_serializer import serialize_combat_state

    TACTICIAN_AVAILABLE = True
except ImportError:
    TACTICIAN_AVAILABLE = False
    extract_json_from_text = None
    find_matching_action = None
    serialize_combat_state = None

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """Supported LLM backends for voice intent parsing."""

    OLLAMA = "ollama"
    CLAUDE = "claude"
    MOCK = "mock"  # For testing without real LLM


class VoiceIntentParserConfig:
    """Configuration for the voice intent parser."""

    def __init__(
        self,
        backend: LLMBackend = LLMBackend.OLLAMA,
        model: str = "lancer-expert",
        max_retries: int = 3,
        fallback_to_random: bool = True,
        enable_logging: bool = True,
        **backend_kwargs,
    ):
        self.backend = backend
        self.model = model
        self.max_retries = max_retries
        self.fallback_to_random = fallback_to_random
        self.enable_logging = enable_logging
        self.backend_kwargs = backend_kwargs


class VoiceIntentParser:
    """Parse natural language voice commands into combat actions."""

    def __init__(self, config: Optional[VoiceIntentParserConfig] = None):
        self.config = config or VoiceIntentParserConfig()
        self._setup_backend()

    def _setup_backend(self):
        """Initialize the LLM backend based on config."""
        if self.config.backend == LLMBackend.OLLAMA:
            try:
                import ollama

                self._ollama_client = ollama
                logger.info("Ollama backend initialized for voice intent parsing")
            except ImportError:
                raise ImportError(
                    "Ollama Python package not installed. "
                    "Install with: pip install ollama"
                )
        elif self.config.backend == LLMBackend.CLAUDE:
            try:
                import anthropic

                self._anthropic_client = anthropic
                logger.info("Claude backend initialized for voice intent parsing")
            except ImportError:
                raise ImportError(
                    "Anthropic Python package not installed. "
                    "Install with: pip install anthropic"
                )
        elif self.config.backend == LLMBackend.MOCK:
            logger.info(
                "Mock backend initialized for voice intent parsing (for testing)"
            )
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
                        "content": "You are a natural language understanding system for Lancer combat.",
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
                system="You are a natural language understanding system for Lancer combat.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _call_mock(self, prompt: str) -> str:
        """Mock LLM response for testing."""
        # Return a simple valid JSON response simulating a clear command
        return """{
            "action_id": "move",
            "target_id": null,
            "target_position": null,
            "weapon_id": null,
            "system_id": null,
            "confidence": 0.9,
            "reasoning": "Mock response for testing",
            "fallback_prompt": null
        }"""

    def _load_voice_intent_prompt(self) -> str:
        """Load the voice intent parsing system prompt."""
        import os

        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", "voice_intent.md"
        )
        with open(prompt_path, "r") as f:
            return f.read()

    def _build_prompt(
        self,
        transcript: str,
        scenario: MechCombatScenario,
        actor_id: str,
        available_actions: List[AvailableAction],
    ) -> str:
        """Build a complete prompt for voice intent parsing."""
        if not TACTICIAN_AVAILABLE:
            raise RuntimeError(
                "Tactician module not available. Cannot serialize combat state."
            )
        assert serialize_combat_state is not None  # For type checker
        combat_state = serialize_combat_state(scenario)

        # Prepare available actions context
        available_actions_context = []
        for action in available_actions:
            action_dict = {
                "action_id": action.action_id,
                "action_name": action.action_name,
                "action_type": action.action_type,
                "is_available": action.is_available,
                "requires_target": action.requires_target,
                "requires_weapon": action.requires_weapon,
                "requires_system": action.requires_system,
                "requires_path": action.requires_path,
                "max_targets": action.max_targets,
                "unavailable_reason": action.unavailable_reason,
            }
            available_actions_context.append(action_dict)

        system_prompt = self._load_voice_intent_prompt()

        prompt = f"""{system_prompt}

## Current Context

Transcript: "{transcript}"

Current Actor ID: {actor_id}

Combatants (IDs only): {[c.get("id", "unknown") for c in combat_state.get("combatants", [])]}

Available Actions: {[a["action_id"] for a in available_actions_context]}

## Task

Parse the transcript into a structured action using the JSON format specified above.

Respond ONLY with the JSON object.
"""
        return prompt

    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        """Extract JSON object from text, with fallback to regex."""
        # Use imported function if available
        if extract_json_from_text:
            return extract_json_from_text(text)

        # Fallback implementation
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def _validate_parsed_intent(
        self,
        parsed: Dict[str, Any],
    ) -> Tuple[
        str,
        Optional[str],
        Optional[HexPosition],
        Optional[str],
        Optional[str],
        float,
        str,
        Optional[str],
    ]:
        """Validate parsed JSON and extract fields."""
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not a dictionary")

        # Required fields
        if "action_id" not in parsed:
            raise ValueError("Missing required field 'action_id'")
        action_id = parsed["action_id"]
        if not isinstance(action_id, str):
            raise ValueError("'action_id' must be a string")

        # Optional fields
        target_id = parsed.get("target_id")
        if target_id is not None and not isinstance(target_id, str):
            raise ValueError("'target_id' must be string or null")

        target_position_raw = parsed.get("target_position")
        target_position: Optional[HexPosition] = None
        if target_position_raw is not None:
            if not isinstance(target_position_raw, dict):
                raise ValueError("'target_position' must be a dict or null")
            # Convert dict to HexPosition
            # Expect either {"q": int, "r": int} or {"coord": {"q": int, "r": int}, "elevation": int}
            if "coord" in target_position_raw:
                coord_dict = target_position_raw["coord"]
                elevation = target_position_raw.get("elevation", 0)
                if not isinstance(coord_dict, dict):
                    raise ValueError("'coord' must be a dict")
                q = coord_dict.get("q")
                r = coord_dict.get("r")
            else:
                q = target_position_raw.get("q")
                r = target_position_raw.get("r")
                elevation = target_position_raw.get("elevation", 0)
            if not isinstance(q, int) or not isinstance(r, int):
                raise ValueError(
                    "'target_position' must have integer q and r coordinates"
                )
            if not isinstance(elevation, int):
                raise ValueError("'target_position' elevation must be integer")
            target_position = HexPosition(
                coord=HexCoord(q=q, r=r),
                elevation=elevation,
            )

        weapon_id = parsed.get("weapon_id")
        if weapon_id is not None and not isinstance(weapon_id, str):
            raise ValueError("'weapon_id' must be string or null")

        system_id = parsed.get("system_id")
        if system_id is not None and not isinstance(system_id, str):
            raise ValueError("'system_id' must be string or null")

        confidence = parsed.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            raise ValueError("'confidence' must be a number")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("'confidence' must be between 0.0 and 1.0")

        reasoning = parsed.get("reasoning", "")
        if not isinstance(reasoning, str):
            raise ValueError("'reasoning' must be a string")

        fallback_prompt = parsed.get("fallback_prompt")
        if fallback_prompt is not None and not isinstance(fallback_prompt, str):
            raise ValueError("'fallback_prompt' must be string or null")

        return (
            action_id,
            target_id,
            target_position,
            weapon_id,
            system_id,
            confidence,
            reasoning,
            fallback_prompt,
        )

    def _choose_random_action(
        self,
        actor_id: str,
        available_actions: List[AvailableAction],
    ) -> Tuple[ActionExecutionInput, float, Optional[str]]:
        """Choose a random valid action as fallback."""
        valid_actions = [a for a in available_actions if a.is_available]
        if not valid_actions:
            valid_actions = available_actions

        action = random.choice(valid_actions)

        target_ids = []
        if action.requires_target:
            # No target selected - let combat system handle it
            pass

        logger.info(f"Random fallback action: {action.action_id}")

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

        return action_input, 0.0, "Random fallback action (voice intent parsing failed)"

    def parse(
        self,
        transcript: str,
        scenario: MechCombatScenario,
        actor_id: str,
        available_actions: List[AvailableAction],
    ) -> Tuple[ActionExecutionInput, float, Optional[str]]:
        """Parse natural language voice command into a combat action.

        Args:
            transcript: Player's spoken command
            scenario: Current combat scenario
            actor_id: ID of the player's combatant
            available_actions: List of available actions for this actor

        Returns:
            Tuple of (ActionExecutionInput, confidence_score, fallback_prompt)
            fallback_prompt is None if confidence is high enough

        Raises:
            ValueError: If parsing fails and fallback_to_random is False
        """
        logger.info(f"Parsing voice intent: '{transcript}' for actor {actor_id}")

        if not available_actions:
            raise ValueError(f"No available actions for actor {actor_id}")

        # Build prompt
        prompt = self._build_prompt(transcript, scenario, actor_id, available_actions)

        # Retry loop
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    f"LLM call attempt {attempt + 1}/{self.config.max_retries}"
                )
                llm_output = self._call_llm(prompt)

                # Extract JSON
                parsed = self._extract_json_from_text(llm_output)
                if parsed is None:
                    raise ValueError("No valid JSON found in LLM output")

                # Validate and extract fields
                (
                    action_id,
                    target_id,
                    target_position,
                    weapon_id,
                    system_id,
                    confidence,
                    reasoning,
                    fallback_prompt,
                ) = self._validate_parsed_intent(parsed)

                # Find matching action
                if not TACTICIAN_AVAILABLE or find_matching_action is None:
                    # Simple matching
                    matching_action = None
                    for action in available_actions:
                        if action.action_id == action_id:
                            matching_action = action
                            break
                    if matching_action is None:
                        # Fuzzy match
                        action_ids = [a.action_id for a in available_actions]
                        matches = get_close_matches(
                            action_id, action_ids, n=1, cutoff=0.6
                        )
                        if matches:
                            matched_id = matches[0]
                            for action in available_actions:
                                if action.action_id == matched_id:
                                    matching_action = action
                                    break
                    if matching_action is None:
                        raise ValueError(f"No matching action found for '{action_id}'")
                else:
                    matching_action = find_matching_action(action_id, available_actions)

                # Validate target requirements
                if matching_action.requires_target and target_id is None:
                    raise ValueError(
                        f"Action '{matching_action.action_id}' requires a target but none provided"
                    )
                if not matching_action.requires_target and target_id is not None:
                    # Ignore target_id for actions that don't require it
                    target_id = None

                target_ids = [target_id] if target_id else []

                # Build ActionExecutionInput
                action_input = ActionExecutionInput(
                    actor_id=actor_id,
                    action_id=matching_action.action_id,
                    action_type=matching_action.action_type,
                    target_ids=target_ids,
                    target_position=target_position,
                    weapon_id=weapon_id,
                    weapon_profile_id=None,  # TODO: Map from weapon_id
                    system_id=system_id,
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

                logger.info(
                    f"Voice intent parsed: {action_input.action_id} "
                    f"(confidence: {confidence:.2f}, target: {target_id})"
                )
                return action_input, confidence, fallback_prompt

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff before retry
                    time.sleep(0.5 * (2**attempt))
                continue

        # All retries failed
        if self.config.fallback_to_random:
            logger.warning(
                "All voice intent parsing attempts failed, falling back to random action"
            )
            return self._choose_random_action(actor_id, available_actions)
        else:
            error_msg = (
                f"Failed to parse voice intent after {self.config.max_retries} attempts"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from last_error


# Public function matching the user story signature
def parse_voice_intent(
    transcript: str,
    available_actions: List[AvailableAction],
    scenario: Optional[MechCombatScenario] = None,
    actor_id: Optional[str] = None,
) -> ActionExecutionInput:
    """Parse natural language voice command into a combat action.

    This is the public function matching the user story acceptance criteria.
    For context-aware parsing, provide scenario and actor_id.

    Args:
        transcript: Player's spoken command
        available_actions: List of available actions for the current actor
        scenario: Current combat scenario (optional, for context-aware parsing)
        actor_id: ID of the player's combatant (optional, required if scenario provided)

    Returns:
        ActionExecutionInput ready for combat system execution

    Raises:
        ValueError: If parsing fails
    """
    # Create parser with default config
    parser = VoiceIntentParser()

    # If scenario and actor_id provided, use context-aware parsing
    if scenario is not None and actor_id is not None:
        action_input, confidence, fallback_prompt = parser.parse(
            transcript, scenario, actor_id, available_actions
        )
    else:
        # Simplified parsing without context
        # For MVP, we'll create a mock scenario just to call parse
        # This is a workaround for backward compatibility
        from core.mech.combat_state import MechCombatScenario

        # Create minimal scenario
        mock_scenario = MechCombatScenario(
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
        mock_actor_id = "player_1"

        action_input, confidence, fallback_prompt = parser.parse(
            transcript, mock_scenario, mock_actor_id, available_actions
        )

    # If confidence is low and fallback_prompt is provided, log it
    if confidence < 0.7 and fallback_prompt:
        logger.info(f"Low confidence parsing voice intent: {fallback_prompt}")

    return action_input
