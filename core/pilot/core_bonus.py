"""Core bonus types for Lancer TTRPG.

Core bonuses are powerful permanent upgrades earned by maxing
out licenses (3 ranks in any single manufacturer's license).
Each manufacturer has unique core bonuses.

Note: This module contains only mechanical definitions (allowed under
the Lancer Third Party License). No copyrighted flavor text.
"""

from pydantic import BaseModel, Field

from core.pilot.license import Manufacturer
from core.shared.dice import DiceExpression
from core.shared.effects import (
    MechanicalEffect,
    StatModifier,
    DamageModifier,
    Immunity,
    Resistance,
    AccuracyModifier,
    RangeModifier,
)


class CoreBonusDefinition(BaseModel):
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
    
    model_config = {"frozen": True}


class CoreBonus(BaseModel):
    """A core bonus that a pilot has earned."""
    
    core_bonus_id: str = Field(..., description="ID of the core bonus definition")
    
    model_config = {"frozen": True}


# GMS Core Bonuses (available to all pilots)
# Note: Only mechanical effects, no flavor text
GMS_CORE_BONUSES: list[CoreBonusDefinition] = [
    CoreBonusDefinition(
        id="gms_reinforced_frame",
        name="Reinforced Frame",
        manufacturer="GMS",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="hp", value=5)],
        ),
    ),
    CoreBonusDefinition(
        id="gms_improved_armament",
        name="Improved Armament",
        manufacturer="GMS",
        effects=MechanicalEffect(
            special="mount_extra_any",  # Gain extra mount of any size
        ),
    ),
    CoreBonusDefinition(
        id="gms_integrated_weapon",
        name="Integrated Weapon",
        manufacturer="GMS",
        effects=MechanicalEffect(
            special="weapon_integrated_main_or_aux",  # Main/Aux weapon becomes integrated
        ),
    ),
    CoreBonusDefinition(
        id="gms_mount_retrofitting",
        name="Mount Retrofitting",
        manufacturer="GMS",
        effects=MechanicalEffect(
            special="mount_size_plus_1",  # Fit weapons one size larger
        ),
    ),
]

# IPS-N Core Bonuses
IPSN_CORE_BONUSES: list[CoreBonusDefinition] = [
    CoreBonusDefinition(
        id="ipsn_siege_ram",
        name="Siege Ram",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            damage_mods=[DamageModifier(dice=DiceExpression.parse("1d6"), damage_type="kinetic", condition=None)],
            special="ram_knockdown",  # Ram knocks prone on hit
        ),
    ),
    CoreBonusDefinition(
        id="ipsn_gyges_frame",
        name="Gyges Frame",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            stat_mods=[
                StatModifier(stat="size", value=1),
                StatModifier(stat="hp", value=5),
            ],
            immunities=[
                Immunity(target="knockback", condition="from_smaller"),
                Immunity(target="prone", condition="from_smaller"),
            ],
        ),
    ),
    CoreBonusDefinition(
        id="ipsn_briareos_frame",
        name="Briareos Frame",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all", condition="aux_weapon")],
            special="mount_aux_extra_2",  # +2 Aux mounts
        ),
    ),
    CoreBonusDefinition(
        id="ipsn_titanomachy_mesh",
        name="Titanomachy Mesh",
        manufacturer="IPS-N",
        effects=MechanicalEffect(
            special="damage_reduction_2",  # Reduce all damage by 2 (min 1)
        ),
    ),
]

# SSC Core Bonuses
SSC_CORE_BONUSES: list[CoreBonusDefinition] = [
    CoreBonusDefinition(
        id="ssc_ghostweave",
        name="Ghostweave",
        manufacturer="SSC",
        effects=MechanicalEffect(
            special="hide_grants_invisible",  # Invisible when hidden until attack/save
        ),
    ),
    CoreBonusDefinition(
        id="ssc_full_subjectivity_sync",
        name="Full Subjectivity Sync",
        manufacturer="SSC",
        effects=MechanicalEffect(
            stat_mods=[
                StatModifier(stat="evasion", value=2),
                StatModifier(stat="e_defense", value=2),
            ],
            accuracy_mods=[AccuracyModifier(value=-1, applies_to="all")],
        ),
    ),
    CoreBonusDefinition(
        id="ssc_sculpted_light",
        name="Sculpted Light",
        manufacturer="SSC",
        effects=MechanicalEffect(
            special="teleport_3_after_boost_1_per_round",
        ),
    ),
    CoreBonusDefinition(
        id="ssc_neurolink_targeting",
        name="Neurolink Targeting",
        manufacturer="SSC",
        effects=MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="ranged", condition="target_has_lock_on")],
        ),
    ),
]

# HORUS Core Bonuses
HORUS_CORE_BONUSES: list[CoreBonusDefinition] = [
    CoreBonusDefinition(
        id="horus_the_lesson_of_the_open_door",
        name="The Lesson of the Open Door",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            range_mods=[RangeModifier(range_type="range", value=50)],
            special="hack_no_los",  # Hack at range 50, no LoS required
        ),
    ),
    CoreBonusDefinition(
        id="horus_the_lesson_of_disbelief",
        name="The Lesson of Disbelief",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            special="damage_negate_d6_5_plus_1_per_round",  # 1/round, d6, 5+ = ignore damage
        ),
    ),
    CoreBonusDefinition(
        id="horus_the_lesson_of_the_held_image",
        name="The Lesson of the Held Image",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            special="holographic_duplicate",
        ),
    ),
    CoreBonusDefinition(
        id="horus_the_lesson_of_transubstantiation",
        name="The Lesson of Transubstantiation",
        manufacturer="HORUS",
        effects=MechanicalEffect(
            special="swap_position_ally_sensors_1_per_scene",
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
            special="overheat_deals_2_energy_adjacent",
        ),
    ),
    CoreBonusDefinition(
        id="ha_ammofeeds",
        name="Ammofeeds",
        manufacturer="HA",
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="limited_bonus", value=2)],
            special="limited_no_reload",  # Can no longer reload Limited weapons
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
            special="save_immobilized_plus_2",
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
