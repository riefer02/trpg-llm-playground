"""Tactical AI for Lancer combat.

Provides AI-driven tactical decision making for NPC mechs in combat.
"""

import enum
import logging
import random
import time
from typing import Optional, Dict, Any, List

from core.mech.combat_state import MechCombatScenario
from core.mech.combat_models import AvailableAction, ActionExecutionInput
from core.mech.combat_execution import get_available_actions

from .state_serializer import serialize_combat_state
from .prompts import build_tactical_prompt, build_tactical_prompt_with_role
from .action_parser import parse_llm_action_sequence

logger = logging.getLogger(__name__)


class LLMBackend(enum.Enum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    CLAUDE = "claude"
    MOCK = "mock"  # For testing without real LLM


class TacticianConfig:
    """Configuration for the Tactician."""

    def __init__(
        self,
        backend: LLMBackend = LLMBackend.OLLAMA,
        model: str = "lancer-expert",
        max_retries: int = 3,
        fallback_to_random: bool = True,
        enable_logging: bool = True,
        role: Optional[str] = None,
        difficulty: float = 0.5,
        **backend_kwargs,
    ):
        self.backend = backend
        self.model = model
        self.max_retries = max_retries
        self.fallback_to_random = fallback_to_random
        self.enable_logging = enable_logging
        self.role = role  # NPC role: striker, defender, artillery, controller
        self.difficulty = max(0.0, min(1.0, difficulty))  # clamp 0-1
        self.backend_kwargs = backend_kwargs


class Tactician:
    """AI tactician that decides actions for NPC combatants."""

    def __init__(self, config: Optional[TacticianConfig] = None):
        self.config = config or TacticianConfig()
        self._setup_backend()

    def _setup_backend(self):
        """Initialize the LLM backend based on config."""
        if self.config.backend == LLMBackend.OLLAMA:
            try:
                import ollama

                self._ollama_client = ollama
                logger.info("Ollama backend initialized")
            except ImportError:
                raise ImportError(
                    "Ollama Python package not installed. "
                    "Install with: pip install ollama"
                )
        elif self.config.backend == LLMBackend.CLAUDE:
            try:
                import anthropic

                self._anthropic_client = anthropic
                logger.info("Claude backend initialized")
            except ImportError:
                raise ImportError(
                    "Anthropic Python package not installed. "
                    "Install with: pip install anthropic"
                )
        elif self.config.backend == LLMBackend.MOCK:
            logger.info("Mock backend initialized (for testing)")
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
                        "content": "You are a tactical AI for Lancer combat.",
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
                system="You are a tactical AI for Lancer combat.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _call_mock(self, prompt: str) -> str:
        """Mock LLM response for testing."""
        # Return a simple valid JSON response for testing
        return """{
            "action_id": "move",
            "target_id": null,
            "confidence": 0.8,
            "reasoning": "Mock response for testing"
        }"""

    def decide_action(
        self,
        scenario: MechCombatScenario,
        actor_id: str,
        economy_state: Optional[Dict[str, Any]] = None,
    ) -> tuple[list[ActionExecutionInput], dict]:
        """Decide the best action(s) for the given actor in the current combat state.

        Args:
            scenario: Current combat scenario
            actor_id: ID of the NPC combatant to decide for
            economy_state: Optional action economy state (if None, will be derived)

        Returns:
            Tuple of (list of ActionExecutionInput ready for combat system execution,
                     reasoning_fields dict containing reasoning, confidence,
                     situation_assessment, considered_options, rationale)

        Raises:
            ValueError: If unable to decide action (and fallback disabled)
        """
        logger.info(f"Tactician deciding action for actor {actor_id}")

        # Get available actions for the actor
        if economy_state is None:
            # For now, we need to get the action economy from the scenario
            # This is a simplification - in real integration, economy should be passed

            # We'll assume the turn has already been started and economy is available
            # For MVP, we'll get available actions with a default economy
            from core.mech.action_economy import ActionEconomyState

            economy = ActionEconomyState()
        else:
            # Convert dict to ActionEconomyState if needed
            from core.mech.action_economy import ActionEconomyState

            economy = ActionEconomyState.model_validate(economy_state)

        available_result = get_available_actions(scenario, actor_id, economy)
        available_actions = (
            available_result.full_actions + available_result.quick_actions
        )

        if not available_actions:
            raise ValueError(f"No available actions for actor {actor_id}")

        # Serialize combat state
        combat_state = serialize_combat_state(scenario)

        # Build prompt (with role if specified)
        if self.config.role:
            prompt = build_tactical_prompt_with_role(
                combat_state, self.config.role, self.config.difficulty
            )
        else:
            prompt = build_tactical_prompt(combat_state, self.config.difficulty)

        # Retry loop
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    f"LLM call attempt {attempt + 1}/{self.config.max_retries}"
                )
                llm_output = self._call_llm(prompt)

                # Parse and validate action(s) - supports single action or sequence
                action_inputs, reasoning_fields = parse_llm_action_sequence(
                    llm_output, actor_id, available_actions
                )

                action_summary = ", ".join(
                    f"{ai.action_id} (targets: {ai.target_ids})" for ai in action_inputs
                )
                logger.info(
                    f"Action sequence decided ({len(action_inputs)} actions): {action_summary}"
                )
                return action_inputs, reasoning_fields

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff before retry
                    time.sleep(0.5 * (2**attempt))
                continue

        # All retries failed
        if self.config.fallback_to_random:
            logger.warning("All LLM attempts failed, falling back to random action")
            return self._choose_random_action(actor_id, available_actions)
        else:
            error_msg = (
                f"Failed to decide action after {self.config.max_retries} attempts"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from last_error

    def _choose_random_action(
        self,
        actor_id: str,
        available_actions: List[AvailableAction],
    ) -> tuple[list[ActionExecutionInput], dict]:
        """Choose a random valid action as fallback."""
        # Filter to available actions
        valid_actions = [a for a in available_actions if a.is_available]
        if not valid_actions:
            # If no actions are available, pick any (shouldn't happen)
            valid_actions = available_actions

        action = random.choice(valid_actions)

        # For actions requiring target, we need to pick a target
        # For simplicity, we'll leave target_ids empty and let combat system handle it
        # (or pick a random enemy if available)
        target_ids = []
        if action.requires_target:
            # This is a simplified placeholder - real implementation needs combat state
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

        reasoning_fields = {
            "reasoning": "Random fallback action (LLM failed)",
            "confidence": 0.0,
            "situation_assessment": "",
            "considered_options": "",
            "rationale": "",
        }

        return [action_input], reasoning_fields
