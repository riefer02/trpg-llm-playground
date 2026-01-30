"""Lancer Tactical AI module.

This module provides AI-driven tactical decision making for Lancer mech combat.
"""

from .state_serializer import serialize_combat_state
from .prompts import (
    load_system_prompt,
    build_tactical_prompt,
    build_tactical_prompt_with_role,
)

__all__ = [
    "serialize_combat_state",
    "load_system_prompt",
    "build_tactical_prompt",
    "build_tactical_prompt_with_role",
]
