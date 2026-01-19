"""Status and condition definitions for mech combat."""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.enums import StatusType


StatusCategory = Literal["status", "condition"]
StatusClearTrigger = Literal[
    "end_of_turn",
    "end_of_next_turn",
    "stabilize",
    "rest",
    "full_repair",
    "boot_up",
    "search_success",
    "cover_lost",
    "attack",
    "boost",
    "reaction",
    "invisibility_lost",
    "stand_up",
]

StatusDurationType = Literal["indefinite", "end_of_turn", "end_of_next_turn"]


class StatusInstance(FrozenModel):
    """Tracks an applied status with duration metadata.

    This model enables automatic status expiration based on turn boundaries
    and trigger-based clearing per PR2 rules.

    Duration types:
    - indefinite: Status persists until explicitly cleared (e.g., prone, shutdown)
    - end_of_turn: Status expires at end of current turn
    - end_of_next_turn: Status expires at end of next turn (e.g., braced, stunned)
    """

    status: StatusType
    applied_on_round: int = Field(..., ge=1, description="Round when status was applied")
    applied_by: str | None = Field(
        default=None, description="ID of combatant who applied this status, if any"
    )
    duration_type: StatusDurationType = Field(
        default="indefinite", description="How the status expires"
    )


class ActionRestriction(FrozenModel):
    """Action limitations caused by a status or condition."""

    disallow_actions: bool = False
    disallow_full_actions: bool = False
    disallow_reactions: bool = False
    disallow_free_actions: bool = False
    disallow_move: bool = False
    disallow_overcharge: bool = False
    disallow_boost: bool = False
    disallow_tech_actions: bool = False
    disallow_comms: bool = False
    max_quick_actions: int | None = Field(default=None, ge=0)
    allowed_action_ids: list[str] = Field(default_factory=list)
    allowed_attack_action_ids: list[str] = Field(default_factory=list)



class MovementRestriction(FrozenModel):
    """Movement limitations caused by a status or condition."""

    max_voluntary_speed: int | None = Field(default=None, ge=0)
    only_regular_move: bool = False
    counts_as_difficult_terrain: bool = False



class TargetingRestriction(FrozenModel):
    """Targeting limitations caused by a status or condition."""

    cannot_be_targeted: bool = False
    area_attacks_can_target: bool = True
    miss_chance: float | None = Field(default=None, ge=0.0, le=1.0)



class StatusEffect(FrozenModel):
    """Mechanical effects for a status or condition."""

    ranged_attack_difficulty: int = 0
    all_attack_difficulty: int = 0
    save_difficulty: int = 0
    skill_check_difficulty: int = 0
    attackers_accuracy_bonus: int = 0  # Negative values represent difficulty.
    damage_multiplier: float | None = Field(default=None, ge=0.0)
    ignore_armor: bool = False
    ignore_resistance: bool = False
    immune_to_tech: bool = False
    end_tech_effects: bool = False
    cool_heat_to_zero: bool = False
    end_exposed: bool = False
    max_evasion: int | None = Field(default=None, ge=0)
    auto_fail_hull_checks: bool = False
    auto_fail_agility_checks: bool = False
    auto_fail_hull_saves: bool = False
    auto_fail_agility_saves: bool = False
    consumable_accuracy_bonus: int | None = Field(default=None, ge=0)
    grants_conditions: list[StatusType] = Field(default_factory=list)
    action_restrictions: ActionRestriction = Field(default_factory=ActionRestriction)
    movement_restrictions: MovementRestriction = Field(default_factory=MovementRestriction)
    targeting_restrictions: TargetingRestriction = Field(default_factory=TargetingRestriction)



class StatusDefinition(FrozenModel):
    """Definition for a combat status or condition."""

    status: StatusType
    category: StatusCategory
    effects: StatusEffect = Field(default_factory=StatusEffect)
    clear_triggers: list[StatusClearTrigger] = Field(default_factory=list)



COMBAT_STATUS_DEFINITIONS: list[StatusDefinition] = [
    StatusDefinition(
        status="braced",
        category="status",
        effects=StatusEffect(
            attackers_accuracy_bonus=-1,
            action_restrictions=ActionRestriction(
                disallow_full_actions=True,
                disallow_reactions=True,
                disallow_free_actions=True,
                disallow_move=True,
                disallow_overcharge=True,
                max_quick_actions=1,
            ),
        ),
        clear_triggers=["end_of_next_turn"],
    ),
    StatusDefinition(
        status="engaged",
        category="status",
        effects=StatusEffect(ranged_attack_difficulty=1),
    ),
    StatusDefinition(
        status="exposed",
        category="status",
        effects=StatusEffect(damage_multiplier=2.0),
    ),
    StatusDefinition(
        status="hidden",
        category="status",
        effects=StatusEffect(
            targeting_restrictions=TargetingRestriction(cannot_be_targeted=True, area_attacks_can_target=True),
        ),
        clear_triggers=["attack", "boost", "reaction", "cover_lost", "search_success", "invisibility_lost"],
    ),
    StatusDefinition(
        status="invisible",
        category="status",
        effects=StatusEffect(targeting_restrictions=TargetingRestriction(miss_chance=0.5)),
    ),
    StatusDefinition(
        status="prone",
        category="status",
        effects=StatusEffect(
            attackers_accuracy_bonus=1,
            grants_conditions=["slowed"],
            movement_restrictions=MovementRestriction(counts_as_difficult_terrain=True),
        ),
        clear_triggers=["stand_up"],
    ),
    StatusDefinition(
        status="shutdown",
        category="status",
        effects=StatusEffect(
            action_restrictions=ActionRestriction(
                disallow_actions=True,
                disallow_reactions=True,
                disallow_move=True,
                disallow_free_actions=True,
                disallow_overcharge=True,
                allowed_action_ids=["boot_up", "mount", "dismount", "eject"],
            ),
            immune_to_tech=True,
            end_tech_effects=True,
            cool_heat_to_zero=True,
            end_exposed=True,
        ),
        clear_triggers=["boot_up"],
    ),
    StatusDefinition(
        status="immobilized",
        category="condition",
        effects=StatusEffect(movement_restrictions=MovementRestriction(max_voluntary_speed=0)),
    ),
    StatusDefinition(
        status="impaired",
        category="condition",
        effects=StatusEffect(all_attack_difficulty=1, save_difficulty=1, skill_check_difficulty=1),
    ),
    StatusDefinition(
        status="jammed",
        category="condition",
        effects=StatusEffect(
            action_restrictions=ActionRestriction(
                disallow_tech_actions=True,
                disallow_reactions=True,
                disallow_comms=True,
                allowed_attack_action_ids=["improvised_attack", "grapple", "ram"],
            ),
        ),
    ),
    StatusDefinition(
        status="lock_on",
        category="condition",
        effects=StatusEffect(consumable_accuracy_bonus=1),
    ),
    StatusDefinition(
        status="shredded",
        category="condition",
        effects=StatusEffect(ignore_armor=True, ignore_resistance=True),
    ),
    StatusDefinition(
        status="slowed",
        category="condition",
        effects=StatusEffect(movement_restrictions=MovementRestriction(only_regular_move=True)),
    ),
    StatusDefinition(
        status="stunned",
        category="condition",
        effects=StatusEffect(
            action_restrictions=ActionRestriction(
                disallow_actions=True,
                disallow_reactions=True,
                disallow_free_actions=True,
                disallow_move=True,
                disallow_overcharge=True,
                allowed_action_ids=["mount", "dismount", "eject"],
            ),
            max_evasion=5,
            auto_fail_hull_checks=True,
            auto_fail_agility_checks=True,
            auto_fail_hull_saves=True,
            auto_fail_agility_saves=True,
        ),
        clear_triggers=["end_of_next_turn"],
    ),
]


STATUS_DEFINITIONS_BY_ID = {definition.status: definition for definition in COMBAT_STATUS_DEFINITIONS}


def get_status_definition(status: StatusType) -> StatusDefinition | None:
    """Look up a status or condition definition by ID."""
    return STATUS_DEFINITIONS_BY_ID.get(status)


def get_status_default_duration(status: StatusType) -> StatusDurationType:
    """Get the default duration type for a status based on its clear triggers.

    Duration mapping:
    - Statuses with 'end_of_next_turn' clear trigger → end_of_next_turn
    - Statuses with 'end_of_turn' clear trigger → end_of_turn
    - All others → indefinite (trigger-based or permanent)

    Args:
        status: The status type to look up

    Returns:
        The duration type for automatic expiration handling
    """
    definition = get_status_definition(status)
    if definition is None:
        return "indefinite"

    if "end_of_next_turn" in definition.clear_triggers:
        return "end_of_next_turn"
    if "end_of_turn" in definition.clear_triggers:
        return "end_of_turn"
    return "indefinite"


def get_status_clear_triggers(status: StatusType) -> list[StatusClearTrigger]:
    """Get the clear triggers for a status.

    Args:
        status: The status type to look up

    Returns:
        List of triggers that clear this status
    """
    definition = get_status_definition(status)
    if definition is None:
        return []
    return definition.clear_triggers
