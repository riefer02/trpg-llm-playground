"""Unified Character model for Lancer TTRPG.

This module provides the Character model that unifies pilot and mech
as components of a single character, as described in the Lancer Core Book:

> "Your pilot and your mech are effectively two components of the same character."

Main exports:
- Character: The unified character model (pilot + mechs)
- MechConfiguration: A named mech loadout owned by a character
- CharacterValidation: Validation result for a character
- validate_character: Validate a character against game rules
- create_ll0_character: Create a valid LL0 character with defaults
"""

from core.character.character import Character, MechConfiguration
from core.character.validation import (
    CharacterValidation,
    CharacterIssue,
    MechValidationEntry,
    validate_character,
)
from core.character.factory import (
    create_ll0_character,
    create_empty_character,
)

__all__ = [
    # Models
    "Character",
    "MechConfiguration",
    # Validation
    "CharacterValidation",
    "CharacterIssue",
    "MechValidationEntry",
    "validate_character",
    # Factory functions
    "create_ll0_character",
    "create_empty_character",
]
