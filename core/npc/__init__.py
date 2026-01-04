"""NPC system for typed Lancer mechanics.

This module provides schemas for NPC templates, stats, abilities,
and integration with combat state. NPCs use tier-based scaling
and can reuse player compendium gear without license requirements.
"""

from core.npc.enums import (
    NPCTier,
    NPCClass,
    NPCAbilityTriggerType,
)
from core.npc.models import (
    NPCStats,
    NPCTemplate,
    NPCAbility,
    NPCGear,
)
from core.npc.state import (
    NPCState,
    NPCCombatStats,
    scale_npc_stats,
    convert_to_combat_stats,
)
from core.npc.validation import (
    NPCValidationIssue,
    NPCValidation,
    validate_npc_template,
    validate_npc_in_combat,
)
from core.npc.compendium import (
    NPC_TEMPLATES,
    NPC_TEMPLATES_BY_ID,
    get_npc_template,
)

__all__ = [
    # Enums
    "NPCTier",
    "NPCClass",
    "NPCAbilityTriggerType",
    # Models
    "NPCStats",
    "NPCTemplate",
    "NPCAbility",
    "NPCGear",
    # State
    "NPCState",
    "NPCCombatStats",
    "scale_npc_stats",
    "convert_to_combat_stats",
    # Validation
    "NPCValidationIssue",
    "NPCValidation",
    "validate_npc_template",
    "validate_npc_in_combat",
    # Compendium
    "NPC_TEMPLATES",
    "NPC_TEMPLATES_BY_ID",
    "get_npc_template",
]
