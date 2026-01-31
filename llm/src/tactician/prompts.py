"""Prompt templates for the Lancer Tactical AI."""

import json
from pathlib import Path
from typing import Dict, Any


def _get_aggression_adjective(difficulty: float) -> str:
    """Get aggression adjective based on difficulty (0.0-1.0).

    Returns:
        Adjective: "cautious", "balanced", or "aggressive"
    """
    if difficulty < 0.3:
        return "cautious"
    elif difficulty < 0.7:
        return "balanced"
    else:
        return "aggressive"


def load_system_prompt() -> str:
    """Load the tactical system prompt from markdown file.

    Returns:
        The full system prompt as a string.
    """
    prompt_path = Path(__file__).parent / "prompts" / "tactical_system.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found at {prompt_path}")
    return prompt_path.read_text()


def build_tactical_prompt(
    combat_state: Dict[str, Any],
    difficulty: float = 0.5,
) -> str:
    """Build a complete tactical prompt with combat state and difficulty.

    Args:
        combat_state: Serialized combat state from state_serializer
        difficulty: Difficulty factor (0.0-1.0) influencing aggression

    Returns:
        Complete prompt string ready for LLM consumption
    """
    system_prompt = load_system_prompt()

    # Determine aggression adjective
    aggression = _get_aggression_adjective(difficulty)

    # Add aggression note
    aggression_note = f"\n\n## Tactical Approach\n\nYou are controlling a **{aggression}** mech. Adjust your risk tolerance accordingly."

    # Format combat state as JSON string for inclusion
    combat_state_json = json.dumps(combat_state, indent=2)

    prompt = f"""{system_prompt}{aggression_note}

## Current Combat State

Below is the current combat state as JSON. Analyze this state and choose the best action as a {aggression} mech.

```json
{combat_state_json}
```

## Your Decision

Based on the combat state above, your {aggression} approach, and tactical principles, choose the best action.

Remember to output ONLY a JSON object with the structure specified above."""
    return prompt


def build_tactical_prompt_with_role(
    combat_state: Dict[str, Any],
    npc_role: str,
    difficulty: float = 0.5,
) -> str:
    """Build tactical prompt with explicit NPC role and difficulty.

    Args:
        combat_state: Serialized combat state
        npc_role: One of "striker", "defender", "artillery", "controller"
        difficulty: Difficulty factor (0.0-1.0) influencing aggression

    Returns:
        Complete prompt with role-specific instructions
    """
    system_prompt = load_system_prompt()

    # Determine aggression adjective
    aggression = _get_aggression_adjective(difficulty)

    # Insert role-specific emphasis with aggression
    role_emphasis = f"\n\n## Role-Specific Instructions\n\nYou are controlling a **{aggression} {npc_role}** mech. Focus on the {npc_role} tactics described above. As a {aggression} mech, adjust your risk tolerance accordingly."

    combat_state_json = json.dumps(combat_state, indent=2)

    prompt = f"""{system_prompt}{role_emphasis}

## Current Combat State

Below is the current combat state as JSON. Analyze this state and choose the best action for a {aggression} {npc_role}.

```json
{combat_state_json}
```

## Your Decision

Based on the combat state above, your {aggression} {npc_role} role, and tactical principles, choose the best action.

Remember to output ONLY a JSON object with the structure specified above."""
    return prompt
