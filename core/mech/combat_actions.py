"""Action rule definitions for mech combat."""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.dice import DiceExpression
from core.shared.enums import ActionType, AttackType, DamageType, StatusType
from core.mech.actions import ActionScope


ActionCategory = Literal[
    "attack",
    "movement",
    "tech",
    "utility",
    "defense",
    "reaction",
    "protocol",
]


class AttackActionProfile(FrozenModel):
    """Attack profile for an action."""

    attack_type: AttackType | None = None
    attack_type_follows_weapon: bool = False
    uses_weapon: bool = True
    weapon_count: int = Field(default=1, ge=1)
    allow_superheavy: bool = True
    allow_aux_bonus_attack: bool = False
    aux_cannot_bonus_damage: bool = False
    ignores_cover: bool = False
    fixed_damage: DiceExpression | None = None
    on_hit_statuses: list[StatusType] = Field(default_factory=list)
    knockback_spaces: int | None = Field(default=None, ge=0)



class MovementActionProfile(FrozenModel):
    """Movement profile for an action."""

    distance_multiplier: float = Field(default=1.0, ge=0.0)
    ignore_engagement: bool = False
    ignore_reactions: bool = False
    counts_as_boost: bool = False



class TechActionProfile(FrozenModel):
    """Tech action profile."""

    is_attack: bool = False
    heat_on_hit: int | None = Field(default=None, ge=0)
    grants_lock_on: bool = False
    bolster_accuracy: int | None = Field(default=None, ge=0)
    scan_options: list[Literal["stats", "hidden_info", "public_info"]] = Field(default_factory=list)
    inflicts_conditions: list[StatusType] = Field(default_factory=list)
    duration: Literal["end_of_next_turn", "until_cleared"] | None = None
    options_per_action: int | None = Field(default=None, ge=1)
    repeat_same_option_allowed: bool = True
    repeat_same_option_requires_free_action: bool = False



CheckStat = Literal["systems", "agility", "skill_check"]


class ContestedCheckRule(FrozenModel):
    """Contested check definition for an action."""

    attacker_stat: CheckStat
    defender_stat: CheckStat
    tie_breaker: Literal["attacker", "defender"] = "attacker"



class LockOnRule(FrozenModel):
    """Lock on action effect and consumption behavior."""

    condition: StatusType = "lock_on"
    duration: Literal["until_consumed", "until_cleared", "end_of_next_turn"] = "until_consumed"
    consumable: bool = True
    consumed_by: Literal["hostile_attack"] = "hostile_attack"
    accuracy_bonus: int = Field(default=1, ge=0)



class HideRule(FrozenModel):
    """Hide action requirements."""

    grants_condition: StatusType = "hidden"
    disallow_if_engaged: bool = True
    requires_hard_or_area_soft_cover: bool = True
    cover_must_conceal_size: bool = True
    allow_without_cover_if_no_los: bool = True
    allow_without_cover_if_invisible: bool = True



class SearchRule(FrozenModel):
    """Search contested check and targeting requirements."""

    requires_hidden_target: bool = True
    contested_check: ContestedCheckRule = Field(
        default_factory=lambda: ContestedCheckRule(attacker_stat="systems", defender_stat="agility")
    )
    pilot_contested_check: ContestedCheckRule = Field(
        default_factory=lambda: ContestedCheckRule(attacker_stat="skill_check", defender_stat="agility")
    )
    pilot_range: int = Field(default=5, ge=0)
    reveals_on_success: bool = True



class GrappleRule(FrozenModel):
    """Grapple-specific restrictions."""

    no_boost_or_reactions: bool = True
    immobilizes_smaller: bool = True
    smaller_moves_with_larger: bool = True
    breaks_on_adjacency_loss: bool = True
    attacker_end_free_action: bool = True
    defender_end_quick_action: bool = True
    defender_contested_hull: bool = True
    equal_size_contested_hull_each_turn: bool = True



class StabilizeRule(FrozenModel):
    """Stabilize action options."""

    primary_options: list[Literal["cool_heat", "spend_repair_full_hp"]] = Field(
        default_factory=lambda: ["cool_heat", "spend_repair_full_hp"]
    )
    repair_cost: int = Field(default=1, ge=0)
    cool_heat_clears_exposed: bool = True
    secondary_options: list[
        Literal["reload_loading", "clear_burn", "clear_condition"]
    ] = Field(default_factory=lambda: ["reload_loading", "clear_burn", "clear_condition"])
    condition_clear_allows_adjacent_ally: bool = True
    condition_clear_disallow_self_sourced: bool = True
    clearable_conditions: list[StatusType] = Field(
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



class PrepareRule(FrozenModel):
    """Prepared action rule."""

    held_action_type: ActionType = "quick"
    blocks_other_actions: bool = True
    blocks_reactions: bool = True
    requires_trigger: bool = True
    expires_start_next_turn: bool = True



class ShutdownRule(FrozenModel):
    """Shutdown action effect."""

    applies_status: StatusType = "shutdown"
    clears_exposed: bool = True
    cools_heat_to_zero: bool = True
    ends_tech_effects: bool = True
    immune_to_tech: bool = True



class BootUpRule(FrozenModel):
    """Boot up action effect."""

    clears_status: StatusType = "shutdown"



class MountRule(FrozenModel):
    """Mount/dismount/eject rules."""

    requires_adjacent: bool = True
    eject_distance: int | None = Field(default=None, ge=0)
    eject_causes_impaired_until_full_repair: bool = False



class SelfDestructRule(FrozenModel):
    """Self-destruct rules."""

    delay_turns_min: int = Field(default=1, ge=0)
    delay_turns_max: int = Field(default=2, ge=0)
    burst_radius: int = Field(default=2, ge=0)
    damage: DiceExpression = Field(default_factory=lambda: DiceExpression.parse("4d6"))
    damage_type: DamageType = "explosive"
    save_skill: Literal["agility"] = "agility"
    save_halves_damage: bool = True



class OverchargeActionRule(FrozenModel):
    """Overcharge action effect."""

    grants_extra_quick_action: bool = True



class BraceRule(FrozenModel):
    """Brace reaction effects."""

    grants_status: StatusType = "braced"
    resist_all_damage_from_trigger: bool = True
    resist_heat_from_trigger: bool = True
    resist_burn_from_trigger: bool = True



class OverwatchRule(FrozenModel):
    """Overwatch reaction trigger and behavior."""

    trigger: Literal["enemy_starts_movement_in_threat"] = "enemy_starts_movement_in_threat"
    uses_weapon_threat: bool = True
    uses_skirmish_attack: bool = True
    uses_per_round: int = Field(default=1, ge=1)



class FightRule(FrozenModel):
    """Pilot fight action rule."""

    uses_pilot_weapon: bool = True
    sidearm_can_be_quick: bool = True



class JockeyOption(FrozenModel):
    """Jockey follow-up option."""

    name: Literal["distract", "shred", "damage"]
    inflicts_conditions: list[StatusType] = Field(default_factory=list)
    heat: int = 0
    damage: int = 0
    damage_type: DamageType | None = None



class JockeyRule(FrozenModel):
    """Pilot jockey action rule."""

    contested_check: Literal["grit_vs_hull"] = "grit_vs_hull"
    options: list[JockeyOption] = Field(default_factory=list)



class ActionRule(FrozenModel):
    """Rule definition for a combat action."""

    id: str
    name: str
    action_type: ActionType
    alternate_action_types: list[ActionType] = Field(default_factory=list)
    scope: ActionScope = "mech"
    category: ActionCategory
    requires_target: bool = False
    requires_adjacent_target: bool = False
    requires_line_of_sight: bool = True
    uses_sensor_range: bool = False
    attack: AttackActionProfile | None = None
    movement: MovementActionProfile | None = None
    tech: TechActionProfile | None = None
    grapple: GrappleRule | None = None
    stabilize: StabilizeRule | None = None
    prepare: PrepareRule | None = None
    shutdown: ShutdownRule | None = None
    boot_up: BootUpRule | None = None
    mount: MountRule | None = None
    self_destruct: SelfDestructRule | None = None
    overcharge: OverchargeActionRule | None = None
    brace: BraceRule | None = None
    overwatch: OverwatchRule | None = None
    fight: FightRule | None = None
    jockey: JockeyRule | None = None
    lock_on: LockOnRule | None = None
    hide: HideRule | None = None
    search: SearchRule | None = None



COMBAT_ACTION_RULES: list[ActionRule] = [
    ActionRule(
        id="skirmish",
        name="Skirmish",
        action_type="quick",
        category="attack",
        requires_target=True,
        attack=AttackActionProfile(
            attack_type=None,
            attack_type_follows_weapon=True,
            weapon_count=1,
            allow_superheavy=False,
            allow_aux_bonus_attack=True,
            aux_cannot_bonus_damage=True,
        ),
    ),
    ActionRule(
        id="barrage",
        name="Barrage",
        action_type="full",
        category="attack",
        requires_target=True,
        attack=AttackActionProfile(
            attack_type=None,
            attack_type_follows_weapon=True,
            weapon_count=2,
            allow_superheavy=True,
            allow_aux_bonus_attack=True,
            aux_cannot_bonus_damage=True,
        ),
    ),
    ActionRule(
        id="boost",
        name="Boost",
        action_type="quick",
        category="movement",
        movement=MovementActionProfile(distance_multiplier=1.0, counts_as_boost=True),
    ),
    ActionRule(
        id="ram",
        name="Ram",
        action_type="quick",
        category="attack",
        requires_target=True,
        requires_adjacent_target=True,
        attack=AttackActionProfile(
            attack_type="melee",
            uses_weapon=False,
            weapon_count=1,
            on_hit_statuses=["prone"],
            knockback_spaces=1,
        ),
    ),
    ActionRule(
        id="grapple",
        name="Grapple",
        action_type="quick",
        category="attack",
        requires_target=True,
        requires_adjacent_target=True,
        attack=AttackActionProfile(attack_type="melee", uses_weapon=False),
        grapple=GrappleRule(),
    ),
    ActionRule(
        id="quick_tech",
        name="Quick Tech",
        action_type="quick",
        category="tech",
        uses_sensor_range=True,
        tech=TechActionProfile(
            options_per_action=1,
            repeat_same_option_allowed=False,
            repeat_same_option_requires_free_action=True,
        ),
    ),
    ActionRule(
        id="bolster",
        name="Bolster",
        action_type="quick",
        category="tech",
        requires_target=True,
        uses_sensor_range=True,
        tech=TechActionProfile(bolster_accuracy=2, duration="end_of_next_turn"),
    ),
    ActionRule(
        id="scan",
        name="Scan",
        action_type="quick",
        category="tech",
        requires_target=True,
        uses_sensor_range=True,
        tech=TechActionProfile(scan_options=["stats", "hidden_info", "public_info"]),
    ),
    ActionRule(
        id="lock_on",
        name="Lock On",
        action_type="quick",
        category="tech",
        requires_target=True,
        uses_sensor_range=True,
        tech=TechActionProfile(grants_lock_on=True),
        lock_on=LockOnRule(),
    ),
    ActionRule(
        id="invade",
        name="Invade",
        action_type="quick",
        category="tech",
        requires_target=True,
        uses_sensor_range=True,
        tech=TechActionProfile(
            is_attack=True,
            heat_on_hit=2,
            inflicts_conditions=["impaired", "slowed"],
            duration="end_of_next_turn",
        ),
    ),
    ActionRule(
        id="hide",
        name="Hide",
        action_type="quick",
        category="utility",
        requires_target=False,
        hide=HideRule(),
    ),
    ActionRule(
        id="search",
        name="Search",
        action_type="quick",
        category="utility",
        requires_target=True,
        uses_sensor_range=True,
        requires_line_of_sight=False,
        search=SearchRule(),
    ),
    ActionRule(
        id="full_tech",
        name="Full Tech",
        action_type="full",
        category="tech",
        uses_sensor_range=True,
        tech=TechActionProfile(options_per_action=2),
    ),
    ActionRule(
        id="improvised_attack",
        name="Improvised Attack",
        action_type="full",
        category="attack",
        requires_target=True,
        attack=AttackActionProfile(
            attack_type="melee",
            uses_weapon=False,
            fixed_damage=DiceExpression.parse("1d6"),
        ),
    ),
    ActionRule(
        id="stabilize",
        name="Stabilize",
        action_type="full",
        category="utility",
        stabilize=StabilizeRule(),
    ),
    ActionRule(
        id="disengage",
        name="Disengage",
        action_type="full",
        category="movement",
        movement=MovementActionProfile(
            distance_multiplier=1.0,
            ignore_engagement=True,
            ignore_reactions=True,
        ),
    ),
    ActionRule(
        id="activate",
        name="Activate",
        action_type="quick",
        category="utility",
        alternate_action_types=["full"],
    ),
    ActionRule(
        id="shutdown",
        name="Shut Down",
        action_type="quick",
        category="utility",
        shutdown=ShutdownRule(),
    ),
    ActionRule(
        id="boot_up",
        name="Boot Up",
        action_type="full",
        category="utility",
        boot_up=BootUpRule(),
    ),
    ActionRule(
        id="mount",
        name="Mount",
        action_type="full",
        scope="both",
        category="utility",
        mount=MountRule(requires_adjacent=True),
    ),
    ActionRule(
        id="dismount",
        name="Dismount",
        action_type="full",
        scope="both",
        category="utility",
        mount=MountRule(requires_adjacent=True),
    ),
    ActionRule(
        id="eject",
        name="Eject",
        action_type="quick",
        scope="both",
        category="utility",
        mount=MountRule(
            requires_adjacent=False,
            eject_distance=6,
            eject_causes_impaired_until_full_repair=True,
        ),
    ),
    ActionRule(
        id="self_destruct",
        name="Self Destruct",
        action_type="quick",
        category="utility",
        self_destruct=SelfDestructRule(),
    ),
    ActionRule(
        id="prepare",
        name="Prepare",
        action_type="quick",
        scope="both",
        category="utility",
        prepare=PrepareRule(),
    ),
    ActionRule(
        id="skill_check",
        name="Skill Check",
        action_type="full",
        scope="both",
        category="utility",
    ),
    ActionRule(
        id="overcharge",
        name="Overcharge",
        action_type="free",
        category="utility",
        overcharge=OverchargeActionRule(),
    ),
    ActionRule(
        id="brace",
        name="Brace",
        action_type="reaction",
        category="defense",
        brace=BraceRule(),
    ),
    ActionRule(
        id="overwatch",
        name="Overwatch",
        action_type="reaction",
        category="reaction",
        requires_target=True,
        attack=AttackActionProfile(
            attack_type=None,
            attack_type_follows_weapon=True,
            weapon_count=1,
        ),
        overwatch=OverwatchRule(),
    ),
    ActionRule(
        id="fight",
        name="Fight",
        action_type="full",
        scope="pilot",
        category="attack",
        requires_target=True,
        fight=FightRule(),
    ),
    ActionRule(
        id="jockey",
        name="Jockey",
        action_type="full",
        scope="pilot",
        category="attack",
        requires_target=True,
        requires_adjacent_target=True,
        jockey=JockeyRule(
            options=[
                JockeyOption(name="distract", inflicts_conditions=["impaired", "slowed"]),
                JockeyOption(name="shred", heat=2),
                JockeyOption(name="damage", damage=4, damage_type="kinetic"),
            ]
        ),
    ),
    ActionRule(
        id="stand_up",
        name="Stand Up",
        action_type="quick",
        scope="both",
        category="utility",
    ),
]


ACTION_RULES_BY_ID = {rule.id: rule for rule in COMBAT_ACTION_RULES}


def get_action_rule(action_id: str) -> ActionRule | None:
    """Look up a combat action rule by ID."""
    return ACTION_RULES_BY_ID.get(action_id)
