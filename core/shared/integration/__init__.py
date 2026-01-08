"""Integration helpers for connecting Lancer game systems.

This module provides bridges between:
- Save resolution and condition application
- Narrative play and mech combat
- NPC behavior patterns for automated decision-making
"""

from core.shared.integration.save_conditions import (
    SaveConditionMapping,
    SaveConditionResult,
    HULL_SAVE_MAPPINGS,
    AGILITY_SAVE_MAPPINGS,
    SYSTEMS_SAVE_MAPPINGS,
    ENGINEERING_SAVE_MAPPINGS,
    COMMON_SAVE_CONDITION_MAPPINGS,
    resolve_save_with_conditions,
    get_default_mappings_for_save_type,
)

from core.shared.integration.narrative_combat import (
    CombatEvent,
    CombatToNarrativeMapper,
    CombatSetup,
    CombatResult,
    NarrativeCombatBridge,
    get_goal_ids,
    get_active_goals,
    DEFAULT_BRIDGE,
)

from core.shared.integration.npc_ai import (
    NPCBehaviorPattern,
    TargetInfo,
    ActionScore,
    NPCActionDecision,
    TargetPriority,
    STRIKER_PATTERN,
    DEFENDER_PATTERN,
    CONTROLLER_PATTERN,
    SUPPORTER_PATTERN,
    NPC_BEHAVIOR_PATTERNS,
    get_role_from_template,
    get_behavior_pattern,
    compute_target_score,
    score_available_actions,
    select_npc_action_with_role,
    is_adjacent,
)

__all__ = [
    # Save → Conditions
    "SaveConditionMapping",
    "SaveConditionResult",
    "HULL_SAVE_MAPPINGS",
    "AGILITY_SAVE_MAPPINGS",
    "SYSTEMS_SAVE_MAPPINGS",
    "ENGINEERING_SAVE_MAPPINGS",
    "COMMON_SAVE_CONDITION_MAPPINGS",
    "resolve_save_with_conditions",
    "get_default_mappings_for_save_type",
    # Narrative → Combat
    "CombatEvent",
    "CombatToNarrativeMapper",
    "CombatSetup",
    "CombatResult",
    "NarrativeCombatBridge",
    "get_goal_ids",
    "get_active_goals",
    "DEFAULT_BRIDGE",
    # NPC AI
    "NPCBehaviorPattern",
    "TargetInfo",
    "ActionScore",
    "NPCActionDecision",
    "TargetPriority",
    "STRIKER_PATTERN",
    "DEFENDER_PATTERN",
    "CONTROLLER_PATTERN",
    "SUPPORTER_PATTERN",
    "NPC_BEHAVIOR_PATTERNS",
    "get_role_from_template",
    "get_behavior_pattern",
    "compute_target_score",
    "score_available_actions",
    "select_npc_action_with_role",
    "is_adjacent",
]
