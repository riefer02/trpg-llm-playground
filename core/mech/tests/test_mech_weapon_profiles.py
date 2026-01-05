import pytest

from core.mech.compendium import FRAME_DEFINITIONS_BY_ID, WEAPON_DEFINITIONS_BY_ID
from core.mech.weapon import resolve_weapon_profile


def test_ghoul_nexus_profile_choice() -> None:
    ghoul = WEAPON_DEFINITIONS_BY_ID["horus_ghoul_nexus"]
    assert ghoul.dynamic is not None
    choice = ghoul.dynamic.profile_choice
    assert choice is not None
    assert choice.default_profile_id == "kinetic"

    profile_ids = {profile.profile_id for profile in choice.profiles}
    assert profile_ids == {"kinetic", "energy", "explosive"}

    default_profile = resolve_weapon_profile(ghoul)
    assert default_profile.profile_id == "kinetic"

    for profile_id in profile_ids:
        profile = resolve_weapon_profile(ghoul, profile_id=profile_id)
        assert profile.damage
        assert profile.damage[0].damage_type == profile_id
        assert profile.damage_type == profile_id
        assert profile.ranges and profile.ranges[0].value == 10
        assert any(tag.tag == "smart" for tag in profile.tags)


def test_ghoul_nexus_invalid_profile() -> None:
    ghoul = WEAPON_DEFINITIONS_BY_ID["horus_ghoul_nexus"]
    with pytest.raises(ValueError):
        resolve_weapon_profile(ghoul, profile_id="void")


def test_mimic_gun_dynamic_rule() -> None:
    mimic = WEAPON_DEFINITIONS_BY_ID["horus_mimic_gun"]
    assert mimic.dynamic is not None
    assert mimic.dynamic.mimic_gun is not None
    rule = mimic.dynamic.mimic_gun

    assert rule.roll_count == 3
    assert rule.damage_divisor == 2
    assert rule.damage_rounding == "up"
    assert rule.damage_bonus == 1


def test_ushabti_mountless_profile() -> None:
    pegasus = FRAME_DEFINITIONS_BY_ID["horus_pegasus"]
    core_system = pegasus.core_system
    assert core_system is not None

    mountless = core_system.mountless_weapons
    assert mountless
    omnigun = mountless[0]
    assert omnigun.id == "horus_ushabti_omnigun"
    assert omnigun.name == "Ushabti Omnigun"
    assert omnigun.action_type == "free"
    assert omnigun.uses_per == "round"
    assert omnigun.target == "enemy"
    assert omnigun.requires_line_of_sight is True
    assert omnigun.counts_as_attack is False
    assert omnigun.auto_hit is True
    assert omnigun.ignores_cover is True
    assert omnigun.damage_unreducible is True
    assert omnigun.counts_as_weapon is False
    assert omnigun.modifiable is False
    assert omnigun.benefits_from_talents is False

    profile = omnigun.profile
    assert profile.ranges and profile.ranges[0].value == 15
    assert profile.damage and profile.damage[0].flat == 1
    assert profile.damage[0].damage_type == "kinetic"
    assert any(tag.tag == "ap" for tag in profile.tags)
    assert "horus_ushabti_omnigun" not in WEAPON_DEFINITIONS_BY_ID
