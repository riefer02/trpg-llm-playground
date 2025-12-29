from core.mech.compendium import SYSTEM_DEFINITIONS_BY_ID


def test_deployment_behavior_fields_present() -> None:
    system = SYSTEM_DEFINITIONS_BY_ID["ipsn_portable_bunker"]
    deployment = system.effects.deployments[0]

    assert deployment.placement_requires_free_space is True
    assert deployment.activation_condition == "on_prime"
    assert deployment.open_topped is True
    assert deployment.immobile is True
    assert deployment.can_deactivate is False


def test_zone_effect_cap_behavior_fields_present() -> None:
    system = SYSTEM_DEFINITIONS_BY_ID["ipsn_aegis_shield_generator"]
    zone = system.effects.zones[0]

    assert zone.total_effect_cap == 20
    assert zone.deactivate_on_effect_cap is True
    assert zone.ends_on_source_destroyed is True
