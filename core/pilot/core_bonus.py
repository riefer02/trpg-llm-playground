"""Core bonus types for Lancer TTRPG.

Core bonuses are powerful permanent upgrades earned by maxing
out licenses (3 ranks in any single manufacturer's license).
Each manufacturer has unique core bonuses.

Note: This module contains only mechanical definitions (allowed under
the Lancer Third Party License). No copyrighted flavor text.
"""

from pydantic import Field
from core.shared.models import FrozenModel

from core.pilot.license import Manufacturer
from core.shared.dice import DiceExpression
from core.shared.effects import (
    MechanicalEffect,
    StatModifier,
    DamageModifier,
    Immunity,
    Resistance,
    AccuracyModifier,
    DirectDamage,
    RangeModifier,
    CheckModifierEffect,
    ResourceChange,
    OverchargeCostCapEffect,
    LimitedUseBonusEffect,
    IntegratedWeaponEffect,
    MountSlotGrant,
    MountSlotReplacement,
    RandomCheckEffect,
    ActionGrant,
    CoverGrant,
    MovementGrant,
    MovementOverrideEffect,
    MovementSurfaceEffect,
    MovementModeAccessEffect,
    JumpDistanceEffect,
    OutOfPlayEffect,
    StatusBreakCondition,
    StatusGrant,
    TriggeredEffect,
    WeaponModEffect,
    ZeroHpSurvivalEffect,
    AISystemLimitEffect,
    AIControlTransferEffect,
)


class CoreBonusDefinition(FrozenModel):
    """
    A core bonus definition - the template for a learnable core bonus.
    
    Core bonuses are powerful, permanent upgrades to a pilot's mech.
    They're earned by reaching LL3 in any manufacturer's licenses
    (3 total license levels with that manufacturer).
    """
    
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    manufacturer: Manufacturer
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)
    


class CoreBonus(FrozenModel):
    """A core bonus that a pilot has earned."""
    
    core_bonus_id: str = Field(..., description="ID of the core bonus definition")
    


# GMS Core Bonuses (available to all pilots)
# Note: Only mechanical effects, no flavor text
GMS_CORE_BONUSES: list[CoreBonusDefinition] = [
    CoreBonusDefinition(
        id="gms_auto_stabilizing_hardpoints",
        name="Auto-Stabilizing Hardpoints",
        manufacturer="GMS",
        effects=MechanicalEffect(
            accuracy_mods=[
                AccuracyModifier(value=1, applies_to="all", condition="selected_mount")
            ],
        ),
    ),
    CoreBonusDefinition(
        id="gms_burnout_insulation",
        name="Burnout Insulation",
        manufacturer="GMS",
        effects=MechanicalEffect(
            damage_mods=[
                DamageModifier(
                    dice=DiceExpression.parse("1d6"),
                    condition="selected_weapon",
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="gms_improved_armament",
        name="Improved Armament",
        manufacturer="GMS",
        effects=MechanicalEffect(
            mount_slot_grants=[
                MountSlotGrant(
                    slot_type="flexible",
                    count=1,
                    requires_mount_count_lt=3,
                    condition="exclude_integrated_mounts",
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="gms_integrated_weapon",
        name="Integrated Weapon",
        manufacturer="GMS",
        effects=MechanicalEffect(
            integrated_weapons=[
                IntegratedWeaponEffect(
                    weapon_size="aux",
                    free_attack_action_type="free",
                    free_attack_uses_per="round",
                    free_attack_trigger="on_attack_roll",
                    requires_other_weapon_attack=True,
                    cannot_be_modified=True,
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="gms_mount_retrofitting",
        name="Mount Retrofitting",
        manufacturer="GMS",
        effects=MechanicalEffect(
            mount_slot_replacements=[
                MountSlotReplacement(new_slot_type="main_aux", count=1)
            ],
        ),
    ),
    CoreBonusDefinition(
        id="gms_reserve_capacitors",
        name="Reserve Capacitors",
        manufacturer="GMS",
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_core_power_spent",
                    effect=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(resource="hp", amount="full", direction="set", target="self"),
                            ResourceChange(resource="heat", amount=0, direction="set", target="self"),
                        ],
                    ),
                )
            ],
            random_checks=[
                RandomCheckEffect(
                    trigger="on_core_power_spent",
                    roll=DiceExpression.parse("1d20"),
                    success_threshold=20,
                    target="self",
                    on_success=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(resource="core_power", amount=1, direction="gain", target="self")
                        ],
                    ),
                )
            ],
        ),
    ),
]

# IPS-N Core Bonuses
IPSN_CORE_BONUSES: list[CoreBonusDefinition] = [
    CoreBonusDefinition(
        id="ipsn_briareos_frame",
        name="Briareos Frame Reinforcement",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            resistances=[
                Resistance(damage_type="all", condition="structure_1_or_less")
            ],
            zero_hp_survival_effects=[
                ZeroHpSurvivalEffect(condition="zero_hp")
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ipsn_fomorian_frame",
        name="Fomorian Frame Reinforcement",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="size", value=1)],
            immunities=[
                Immunity(target="knockback", condition="from_smaller"),
                Immunity(target="prone", condition="from_smaller"),
                Immunity(target="pull", condition="from_smaller"),
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ipsn_gyges_frame",
        name="Gyges Frame Reinforcement",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            check_mods=[
                CheckModifierEffect(
                    value=1,
                    check_types=["hull"],
                    check_kinds=["check", "save"],
                )
            ],
            range_mods=[
                RangeModifier(range_type="threat", value=1, condition="melee_attack")
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ipsn_reinforced_frame",
        name="Reinforced Frame",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="hp", value=5)],
        ),
    ),
    CoreBonusDefinition(
        id="ipsn_sloped_plating",
        name="Sloped Plating",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="armor", value=1)],
        ),
    ),
    CoreBonusDefinition(
        id="ipsn_titanomachy_mesh",
        name="Titanomachy Mesh",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_hit",
                    condition="ram_or_grapple",
                    uses_per="round",
                    effect=MechanicalEffect(
                        action_grants=[
                            ActionGrant(
                                action_type="free",
                                name="Bonus Ram/Grapple",
                                trigger="on_hit",
                            )
                        ]
                    ),
                )
            ],
            forced_movements=[
                ForcedMovement(
                    direction="push",
                    distance=1,
                    target="enemy",
                    condition="melee_knockback",
                )
            ],
        ),
    ),
]

# SSC Core Bonuses
SSC_CORE_BONUSES: list[CoreBonusDefinition] = [
    CoreBonusDefinition(
        id="ssc_full_subjectivity_sync",
        name="Full Subjectivity Sync",
        manufacturer="SSC",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="evasion", value=2)],
        ),
    ),
    CoreBonusDefinition(
        id="ssc_ghostweave",
        name="Ghostweave",
        manufacturer="SSC",
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_turn_start",
                    effect=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="invisible",
                                target="self",
                                duration="end_of_turn",
                            )
                        ],
                    ),
                ),
                TriggeredEffect(
                    trigger="on_turn_end",
                    condition="turn_only_move_hide_boost",
                    effect=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="invisible",
                                target="self",
                                duration="start_of_next_turn",
                            )
                        ],
                    ),
                ),
            ],
            status_breaks=[
                StatusBreakCondition(
                    status="invisible",
                    target="self",
                    break_triggers=["reaction"],
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ssc_integrated_nerveweave",
        name="Integrated Nerveweave",
        manufacturer="SSC",
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_move",
                    condition="after_boost",
                    effect=MechanicalEffect(
                        movement_grants=[
                            MovementGrant(spaces=2, movement_type="boost")
                        ],
                    ),
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ssc_kai_bioplating",
        name="Kai Bioplating",
        manufacturer="SSC",
        effects=MechanicalEffect(
            check_mods=[
                CheckModifierEffect(
                    value=1,
                    check_types=["agility"],
                    check_kinds=["check"],
                )
            ],
            movement_surface_effects=[
                MovementSurfaceEffect(ignore_difficult_terrain=True),
            ],
            movement_mode_accesses=[
                MovementModeAccessEffect(
                    climb_at_full_speed=True,
                    swim_at_full_speed=True,
                )
            ],
            jump_distance_effects=[
                JumpDistanceEffect(horizontal_multiplier=1.0, vertical_multiplier=0.5)
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ssc_neurolinked_targeting",
        name="Neuro-Linked Targeting",
        manufacturer="SSC",
        effects=MechanicalEffect(
            range_mods=[
                RangeModifier(range_type="range", value=3, condition="ranged_attack")
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ssc_all_theater_movement",
        name="All-Theater Movement Suite",
        manufacturer="SSC",
        effects=MechanicalEffect(
            movement_overrides=[
                MovementOverrideEffect(
                    movement_modes=["move", "boost"],
                    override_type="fly",
                    duration="scene",
                )
            ],
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_turn_end",
                    condition="flying_this_turn",
                    effect=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(resource="heat", amount=1, direction="gain", target="self")
                        ],
                    ),
                )
            ],
        ),
    ),
]

# HORUS Core Bonuses
HORUS_CORE_BONUSES: list[CoreBonusDefinition] = [
    CoreBonusDefinition(
        id="horus_the_lesson_of_disbelief",
        name="The Lesson of Disbelief",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="e_defense", value=2)],
            check_mods=[
                CheckModifierEffect(
                    value=1,
                    check_types=["systems"],
                    check_kinds=["check", "save"],
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="horus_the_lesson_of_the_open_door",
        name="The Lesson of the Open Door",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="save_target", value=2)],
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_target_failed_save",
                    condition="caused_by_self",
                    uses_per="round",
                    effect=MechanicalEffect(
                        direct_damages=[
                            DirectDamage(damage_type="heat", flat=2, target="enemy")
                        ]
                    ),
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="horus_the_lesson_of_the_held_image",
        name="The Lesson of the Held Image",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            action_grants=[
                ActionGrant(
                    action_type="reaction",
                    name="Lock On (Held Image)",
                    trigger="on_ally_turn_start",
                    uses_per="round",
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="horus_the_lesson_of_transubstantiation",
        name="The Lesson of Transubstantiation",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_structure_loss",
                    effect=MechanicalEffect(
                        out_of_play_effects=[
                            OutOfPlayEffect(duration="start_of_next_turn")
                        ],
                    ),
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="horus_the_lesson_of_thinking_tomorrows_thought",
        name="The Lesson of Thinking-Tomorrow's-Thought",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_tech_attack_hit",
                    effect=MechanicalEffect(
                        accuracy_mods=[
                            AccuracyModifier(
                                value=1,
                                applies_to="melee",
                                condition="next_melee_vs_tech_hit_target",
                            )
                        ],
                        weapon_mods=[
                            WeaponModEffect(
                                allowed_weapon_types=["melee"],
                                damage_unreducible=True,
                                condition="next_melee_vs_tech_hit_target",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="horus_the_lesson_of_shaping",
        name="The Lesson of Shaping",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            ai_system_limits=[
                AISystemLimitEffect(bonus_systems=1, max_ai_systems=2)
            ],
            ai_control_transfers=[AIControlTransferEffect()],
        ),
    ),
]

# Harrison Armory Core Bonuses
HA_CORE_BONUSES: list[CoreBonusDefinition] = [
    CoreBonusDefinition(
        id="ha_superior_by_design",
        name="Superior by Design",
        manufacturer="HA",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="heat_cap", value=2)],
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_overheat",
                    effect=MechanicalEffect(
                        direct_damages=[
                            DirectDamage(
                                damage_type="energy",
                                flat=2,
                                target="adjacent",
                            )
                        ]
                    ),
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ha_ammofeeds",
        name="Ammofeeds",
        manufacturer="HA",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="limited_bonus", value=2)],
            reload_restrictions=[
                ReloadRestrictionEffect(
                    applies_to="limited",
                    disallow_reload=True,
                )
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ha_burnout_insulation",
        name="Burnout Insulation",
        manufacturer="HA",
        effects=MechanicalEffect(
            immunities=[Immunity(target="burn")],
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all", condition="deals_burn_or_heat")],
        ),
    ),
    CoreBonusDefinition(
        id="ha_integrated_nervesuit",
        name="Integrated Nervesuit",
        manufacturer="HA",
        effects=MechanicalEffect(
            immunities=[Immunity(target="reactions_from_movement")],
            check_value_mods=[
                CheckValueModifierEffect(
                    value=2,
                    check_kinds=["save"],
                    condition="avoid_immobilized",
                )
            ],
        ),
    ),
]

# All core bonuses combined
ALL_CORE_BONUSES: list[CoreBonusDefinition] = (
    GMS_CORE_BONUSES + IPSN_CORE_BONUSES + SSC_CORE_BONUSES + 
    HORUS_CORE_BONUSES + HA_CORE_BONUSES
)


def get_core_bonus_definition(core_bonus_id: str) -> CoreBonusDefinition | None:
    """Look up a core bonus definition by ID."""
    for cb in ALL_CORE_BONUSES:
        if cb.id == core_bonus_id:
            return cb
    return None


def get_core_bonuses_by_manufacturer(manufacturer: Manufacturer) -> list[CoreBonusDefinition]:
    """Get all core bonuses from a specific manufacturer."""
    return [cb for cb in ALL_CORE_BONUSES if cb.manufacturer == manufacturer]
