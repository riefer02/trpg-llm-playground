from core.mech.compendium import (
    FRAME_DEFINITIONS_BY_ID,
    SYSTEM_DEFINITIONS_BY_ID,
    WEAPON_DEFINITIONS_BY_ID,
)
from core.shared.effects import (
    AttackContextCondition,
    ConditionGroup,
    ReactionCondition,
    SpatialCondition,
    SizeCondition,
)


def test_low_profile_and_flechette_conditions() -> None:
    low_profile = SYSTEM_DEFINITIONS_BY_ID["ssc_low_profile"]
    accuracy = low_profile.effects.accuracy_mods[0]

    assert isinstance(accuracy.condition, AttackContextCondition)
    assert set(accuracy.condition.attack_types) == {"ranged", "tech"}

    check_mod = low_profile.effects.check_mods[0]
    assert check_mod.check_kinds == ["search"]
    assert check_mod.target == "enemy"
    assert check_mod.condition == "hidden"

    flechette = WEAPON_DEFINITIONS_BY_ID["ipsn_flechette_launcher"]
    trigger = flechette.effects.triggered_effects[0]

    assert isinstance(trigger.condition, ConditionGroup)
    assert "target_grappled" in trigger.condition.any_of
    assert "target_biological" in trigger.condition.any_of


def test_armor_lock_supercharger_and_line_conditions() -> None:
    armor_lock = SYSTEM_DEFINITIONS_BY_ID["ipsn_armor_lock_system"]
    armor_effect = armor_lock.effects.triggered_effects[0].effect
    accuracy_condition = armor_effect.accuracy_mods[0].condition

    assert isinstance(accuracy_condition, ConditionGroup)
    assert "braced" in accuracy_condition.all_of
    assert any(
        isinstance(condition, AttackContextCondition)
        for condition in accuracy_condition.all_of
    )

    for immunity in armor_effect.immunities:
        condition = immunity.condition
        assert isinstance(condition, ConditionGroup)
        assert "braced" in condition.all_of
        assert any(isinstance(part, SizeCondition) for part in condition.all_of)

    lancaster = FRAME_DEFINITIONS_BY_ID["ipsn_lancaster"]
    core_power = lancaster.core_system.effects.core_powers[0]
    assert core_power.effects.triggered_effects[0].condition is None
    assert isinstance(core_power.effects.triggered_effects[1].condition, ConditionGroup)

    accuracy = core_power.effects.accuracy_mods[0]
    assert accuracy.target == "ally"

    check_mod = core_power.effects.check_mods[0]
    assert check_mod.target == "ally"

    assault_grapples = FRAME_DEFINITIONS_BY_ID["ipsn_blackbeard"]
    targeting = assault_grapples.core_system.effects.core_powers[0].effects.targetings[0]
    assert isinstance(targeting.condition, SpatialCondition)
    assert targeting.condition.relation == "within_range"

    mag_cannon = WEAPON_DEFINITIONS_BY_ID["ssc_mag_cannon"]
    mag_save = mag_cannon.effects.save_checks[0]
    assert isinstance(mag_save.condition, AttackContextCondition)
    assert mag_save.condition.area_shapes == ["line"]

    veil_rifle = WEAPON_DEFINITIONS_BY_ID["ssc_veil_rifle"]
    cover_grant = veil_rifle.effects.cover_grants[0]
    assert isinstance(cover_grant.condition, SpatialCondition)
    assert cover_grant.condition.relation == "line_of_attack"


def test_reaction_and_line_of_sight_conditions() -> None:
    tortuga = FRAME_DEFINITIONS_BY_ID["ipsn_tortuga"]
    overwatch_trigger = tortuga.core_system.effects.triggered_effects[0]
    assert isinstance(overwatch_trigger.condition, ReactionCondition)
    assert overwatch_trigger.condition.reaction_id == "overwatch"
    assert overwatch_trigger.condition.is_attack is True

    reaction_accuracy = tortuga.traits[0].effects.accuracy_mods[0].condition
    assert isinstance(reaction_accuracy, ReactionCondition)
    assert reaction_accuracy.is_attack is True

    armor_lock = SYSTEM_DEFINITIONS_BY_ID["ipsn_armor_lock_system"]
    brace_trigger = armor_lock.effects.triggered_effects[0]
    assert isinstance(brace_trigger.condition, ReactionCondition)
    assert brace_trigger.condition.reaction_id == "brace"

    swallowtail = FRAME_DEFINITIONS_BY_ID["ssc_swallowtail"]
    tacsim = swallowtail.core_system.effects.random_checks[0]
    assert isinstance(tacsim.condition, ConditionGroup)
    assert any(
        isinstance(condition, SpatialCondition)
        and condition.relation == "line_of_sight"
        for condition in tacsim.condition.all_of
    )

    monarch = FRAME_DEFINITIONS_BY_ID["ssc_monarch"]
    avenger_trait = next(trait for trait in monarch.traits if trait.name == "Avenger Silos")
    avenger_trigger = avenger_trait.effects.triggered_effects[0]
    assert isinstance(avenger_trigger.condition, ConditionGroup)
    assert any(
        isinstance(condition, SpatialCondition)
        and condition.relation == "within_range"
        and condition.range == 15
        and condition.requires_line_of_sight is True
        for condition in avenger_trigger.condition.all_of
    )
