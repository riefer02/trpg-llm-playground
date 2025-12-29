from core.mech.compendium import FRAME_DEFINITIONS_BY_ID
from core.shared.effects import AttackContextCondition


def test_mjolnir_chamber_pool() -> None:
    frame = FRAME_DEFINITIONS_BY_ID["ipsn_raleigh"]
    core_power = frame.core_system.effects.core_powers[0]
    pools = core_power.effects.dice_pools

    assert len(pools) == 1
    pool = pools[0]
    assert pool.pool_name == "mjolnir_chambers"
    assert pool.weapon_id == "ipsn_m35_mjolnir"
    assert pool.max_dice == 6

    gain = pool.gain_triggers[0]
    assert gain.trigger == "on_turn_end"
    assert gain.amount == 2
    assert gain.requires_no_spend_this_turn is True

    spend = pool.spend_options[0]
    assert spend.spend_all is True
    assert isinstance(spend.condition, AttackContextCondition)
    assert spend.effect_per_die is not None
    assert spend.bonus_requires_spend_at_least == 4
    assert spend.bonus_effect is not None
    assert any(
        tag.tag == "ap"
        for mod in spend.bonus_effect.weapon_mods
        for tag in mod.add_tags
    )
    assert any(
        grant.status == "shredded" for grant in spend.bonus_effect.status_grants
    )
