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
    NPCRole,
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
    get_templates_by_class,
    get_templates_by_tier,
)
from core.npc.templates import (
    get_templates_by_role,
    get_striker_templates,
    get_defender_templates,
    get_controller_templates,
    get_supporter_templates,
    NPCTemplateVariant,
    create_variant,
    create_elite_variant,
    create_veteran_variant,
    create_boss_variant,
    NPCSpecialClass,
    get_special_class_description,
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
    "NPCRole",
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
    "get_templates_by_class",
    "get_templates_by_tier",
    # Templates
    "get_templates_by_role",
    "get_striker_templates",
    "get_defender_templates",
    "get_controller_templates",
    "get_supporter_templates",
    "NPCTemplateVariant",
    "create_variant",
    "create_elite_variant",
    "create_veteran_variant",
    "create_boss_variant",
    "NPCSpecialClass",
    "get_special_class_description",
]
