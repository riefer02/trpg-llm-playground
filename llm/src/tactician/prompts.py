"""Prompt templates for the Lancer Tactical AI."""

import json
from pathlib import Path
from typing import Dict, Any


def load_system_prompt() -> str:
    """Load the tactical system prompt from markdown file.

    Returns:
        The full system prompt as a string.
    """
    prompt_path = Path(__file__).parent / "prompts" / "tactical_system.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found at {prompt_path}")
    return prompt_path.read_text()


def build_tactical_prompt(combat_state: Dict[str, Any]) -> str:
    """Build a complete tactical prompt with combat state.

    Args:
        combat_state: Serialized combat state from state_serializer

    Returns:
        Complete prompt string ready for LLM consumption
    """
    system_prompt = load_system_prompt()

    # Format combat state as JSON string for inclusion
    combat_state_json = json.dumps(combat_state, indent=2)

    prompt = f"""{system_prompt}

## Current Combat State

Below is the current combat state as JSON. Analyze this state and choose the best action.

```json
{combat_state_json}
```

## Your Decision

Based on the combat state above, your role, and tactical principles, choose the best action.

Remember to output ONLY a JSON object with the structure specified above."""

    return prompt


def build_tactical_prompt_with_role(combat_state: Dict[str, Any], npc_role: str) -> str:
    """Build tactical prompt with explicit NPC role.

    Args:
        combat_state: Serialized combat state
        npc_role: One of "striker", "defender", "artillery", "controller"

    Returns:
        Complete prompt with role-specific instructions
    """
    system_prompt = load_system_prompt()

    # Insert role-specific emphasis
    role_emphasis = f"\n\n## Role-Specific Instructions\n\nYou are controlling a **{npc_role}** mech. Focus on the {npc_role} tactics described above."

    combat_state_json = json.dumps(combat_state, indent=2)

    prompt = f"""{system_prompt}{role_emphasis}

## Current Combat State

Below is the current combat state as JSON. Analyze this state and choose the best action for a {npc_role}.

```json
{combat_state_json}
```

## Your Decision

Based on the combat state above, your {npc_role} role, and tactical principles, choose the best action.

Remember to output ONLY a JSON object with the structure specified above."""

    return prompt
