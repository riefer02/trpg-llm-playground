"""Mech rules and penalties for Lancer TTRPG."""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.effects import AccuracyModifier
from core.shared.enums import AttackType, DamageType, StatusType
from core.shared.dice import DiceExpression


class MechPilotingRules(FrozenModel):
    """Rules for piloting a mech."""

    unlicensed_accuracy_penalty: AccuracyModifier = AccuracyModifier(value=-1)
    unlicensed_status_penalties: list[StatusType] = Field(
        default_factory=lambda: ["impaired", "slowed"],
    )



class CorePowerRules(FrozenModel):
    """Core power availability rules."""

    starts_with_core_power: bool = True
    restores_on_mission_start: bool = True
    restores_on_full_repair: bool = True



class SystemPointRules(FrozenModel):
    """System point bonus rules."""

    grit_bonus: bool = True
    systems_per_bonus_sp: int = Field(default=2, ge=1)



DEFAULT_MECH_PILOTING_RULES = MechPilotingRules()
DEFAULT_CORE_POWER_RULES = CorePowerRules()
DEFAULT_SYSTEM_POINT_RULES = SystemPointRules()


class ActionEconomyRules(FrozenModel):
    """Action economy rules for mech combat."""

    quick_actions_per_turn: int = Field(default=2, ge=0)
    full_actions_per_turn: int = Field(default=1, ge=0)
    allows_duplicate_actions: bool = False
    duplicate_requires_free_action: bool = True



class CoverRules(FrozenModel):
    """Cover modifiers for ranged attacks."""

    soft_cover_difficulty: int = Field(default=1, ge=0)
    hard_cover_difficulty: int = Field(default=2, ge=0)
    hard_cover_requires_adjacency: bool = True
    hard_cover_flanking_negates: bool = True
    hard_cover_requires_size_match: bool = True
    cover_types_do_not_stack: bool = True
    characters_provide_cover: bool = False



class AttackRules(FrozenModel):
    """Attack targeting rules."""

    melee_target_stat: Literal["evasion"] = "evasion"
    ranged_target_stat: Literal["evasion"] = "evasion"
    tech_target_stat: Literal["e_defense"] = "e_defense"
    ranged_engaged_difficulty: int = Field(default=1, ge=0)



class CriticalHitRules(FrozenModel):
    """Critical hit rules for attacks."""

    threshold: int = Field(default=20, ge=0)
    applies_to: list[AttackType] = Field(default_factory=lambda: ["melee", "ranged"])
    roll_damage_twice_take_highest: bool = True



class BonusDamageRules(FrozenModel):
    """Bonus damage limits and scaling."""

    allowed_types: list[DamageType] = Field(default_factory=lambda: ["kinetic", "explosive", "energy"])
    halve_on_multi_target: bool = True



class ThreatRules(FrozenModel):
    """Threat range defaults."""

    default_threat: int = Field(default=1, ge=0)



class HeatRules(FrozenModel):
    """Heat and overheat behavior."""

    heat_is_damage: bool = False
    heat_affected_by_resistance: bool = True
    overheat_on_exceeding_cap: bool = True



class BurnRules(FrozenModel):
    """Burn behavior for mechs."""

    immediate_damage_ignores_armor: bool = True
    end_turn_clear_on_success: bool = True
    end_turn_check_skill: Literal["engineering"] = "engineering"
    failure_damage_equals_burn_total: bool = True



class OverchargeRules(FrozenModel):
    """Overcharge heat escalation rules."""

    costs: list[int | DiceExpression] = Field(
        default_factory=lambda: [
            1,
            DiceExpression.parse("1d3"),
            DiceExpression.parse("1d6"),
            DiceExpression.parse("1d6+4"),
        ]
    )
    resets_on_full_repair: bool = True



class InvisibilityRules(FrozenModel):
    """Invisibility targeting rules."""

    miss_chance: float = Field(default=0.5, ge=0.0, le=1.0)
    can_always_hide: bool = True



class ObjectRules(FrozenModel):
    """Defaults for objects in mech combat."""

    base_evasion: int = Field(default=5, ge=0)
    hp_per_size: int = Field(default=10, ge=0)
    section_hp: int = Field(default=10, ge=0)



class StabilizeRules(FrozenModel):
    """Conditions that can be cleared by Stabilize."""

    removable_conditions: list[StatusType] = Field(
        default_factory=lambda: [
            "impaired",
            "shredded",
            "jammed",
            "slowed",
            "immobilized",
            "stunned",
            "lock_on",
        ]
    )



class ReactionRules(FrozenModel):
    """Reaction limits and defaults."""

    max_reactions_per_turn: int = Field(default=1, ge=0)
    default_reactions_per_round: list[str] = Field(default_factory=lambda: ["brace", "overwatch"])



DEFAULT_ACTION_ECONOMY_RULES = ActionEconomyRules()
DEFAULT_COVER_RULES = CoverRules()
DEFAULT_ATTACK_RULES = AttackRules()
DEFAULT_CRITICAL_HIT_RULES = CriticalHitRules()
DEFAULT_BONUS_DAMAGE_RULES = BonusDamageRules()
DEFAULT_THREAT_RULES = ThreatRules()
DEFAULT_HEAT_RULES = HeatRules()
DEFAULT_BURN_RULES = BurnRules()
DEFAULT_OVERCHARGE_RULES = OverchargeRules()
DEFAULT_INVISIBILITY_RULES = InvisibilityRules()
DEFAULT_OBJECT_RULES = ObjectRules()
DEFAULT_STABILIZE_RULES = StabilizeRules()
DEFAULT_REACTION_RULES = ReactionRules()
