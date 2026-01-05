"""Tests for pilot gear models and tag effect helpers."""

import pytest
from core.pilot.gear import (
    PilotGearTag,
    PilotGearItemDefinition,
    PilotDamageSpec,
    PilotAreaEffect,
    PilotWeaponProfile,
    get_pilot_gear_definition,
    is_sidearm_weapon,
    is_archaic_weapon,
    is_inaccurate_weapon,
    is_loading_weapon,
    is_ordnance_weapon,
    get_pilot_weapon_action_type,
    get_pilot_weapon_difficulty_modifier,
    can_pilot_weapon_damage_target,
    PILOT_GEAR_DEFINITIONS_BY_ID,
)


class TestPilotAreaEffectObjectDamage:
    """Test PilotAreaEffect supports object damage fields."""

    def test_pilot_area_effect_has_object_damage_field(self):
        """PilotAreaEffect should accept object_damage field."""
        effect = PilotAreaEffect(
            pattern="blast",
            size=1,
            damage=PilotDamageSpec(damage_type="energy", flat=3),
            object_damage=PilotDamageSpec(damage_type="energy", flat=10),
        )
        assert effect.object_damage is not None
        assert effect.object_damage.flat == 10

    def test_pilot_area_effect_has_objects_auto_hit_field(self):
        """PilotAreaEffect should accept objects_auto_hit field."""
        effect = PilotAreaEffect(
            pattern="blast",
            size=1,
            objects_auto_hit=True,
        )
        assert effect.objects_auto_hit is True

    def test_pilot_area_effect_defaults_objects_auto_hit_to_false(self):
        """objects_auto_hit should default to False."""
        effect = PilotAreaEffect(
            pattern="blast",
            size=1,
        )
        assert effect.objects_auto_hit is False

    def test_pilot_area_effect_full_thermal_charge_definition(self):
        """Thermal charge should have object damage and auto-hit."""
        thermal_charge = get_pilot_gear_definition("thermal_charge")
        assert thermal_charge is not None
        assert len(thermal_charge.charges) == 1
        area = thermal_charge.charges[0].area
        assert area.object_damage is not None
        assert area.object_damage.flat == 10
        assert area.object_damage.ap is True
        assert area.objects_auto_hit is True


class TestPilotWeaponProfileLoadedField:
    """Test PilotWeaponProfile tracks loading state."""

    def test_pilot_weapon_profile_has_loaded_field(self):
        """PilotWeaponProfile should have loaded field."""
        profile = PilotWeaponProfile(
            range_type="range",
            range=5,
            damage=PilotDamageSpec(damage_type="kinetic", flat=1),
        )
        assert profile.loaded is True

    def test_pilot_weapon_profile_loaded_defaults_to_true(self):
        """loaded should default to True for weapons without loading tag."""
        profile = PilotWeaponProfile(
            range_type="threat",
            range=1,
            damage=PilotDamageSpec(damage_type="kinetic", flat=2),
        )
        assert profile.loaded is True

    def test_signature_weapon_heavy_has_loading_tag_and_not_loaded(self):
        """Signature Weapon (Heavy) should have loading tag and loaded=False."""
        heavy = get_pilot_gear_definition("signature_weapon_heavy")
        assert heavy is not None
        assert heavy.category == "weapon"
        assert heavy.weapon_profile is not None
        assert heavy.weapon_profile.loaded is False
        assert is_loading_weapon(heavy.tags)

    def test_sidearm_weapon_is_loaded_by_default(self):
        """Sidearm weapons should be loaded by default."""
        sidearm = get_pilot_gear_definition("signature_weapon_sidearm")
        assert sidearm is not None
        assert sidearm.weapon_profile is not None
        assert sidearm.weapon_profile.loaded is True


class TestTagQueryHelpers:
    """Test pilot weapon tag query helper functions."""

    def test_is_sidearm_weapon_with_sidearm_tag(self):
        """is_sidearm_weapon returns True for sidearm tag."""
        tags = [PilotGearTag(tag="sidearm")]
        assert is_sidearm_weapon(tags) is True

    def test_is_sidearm_weapon_without_sidearm_tag(self):
        """is_sidearm_weapon returns False without sidearm tag."""
        tags = [PilotGearTag(tag="inaccurate")]
        assert is_sidearm_weapon(tags) is False

    def test_is_archaic_weapon_with_archaic_tag(self):
        """is_archaic_weapon returns True for archaic tag."""
        tags = [PilotGearTag(tag="archaic")]
        assert is_archaic_weapon(tags) is True

    def test_is_archaic_weapon_without_archaic_tag(self):
        """is_archaic_weapon returns False without archaic tag."""
        tags = [PilotGearTag(tag="sidearm")]
        assert is_archaic_weapon(tags) is False

    def test_is_inaccurate_weapon_with_inaccurate_tag(self):
        """is_inaccurate_weapon returns True for inaccurate tag."""
        tags = [PilotGearTag(tag="inaccurate")]
        assert is_inaccurate_weapon(tags) is True

    def test_is_inaccurate_weapon_without_inaccurate_tag(self):
        """is_inaccurate_weapon returns False without inaccurate tag."""
        tags = [PilotGearTag(tag="sidearm")]
        assert is_inaccurate_weapon(tags) is False

    def test_is_loading_weapon_with_loading_tag(self):
        """is_loading_weapon returns True for loading tag."""
        tags = [PilotGearTag(tag="loading")]
        assert is_loading_weapon(tags) is True

    def test_is_loading_weapon_without_loading_tag(self):
        """is_loading_weapon returns False without loading tag."""
        tags = [PilotGearTag(tag="ordnance")]
        assert is_loading_weapon(tags) is False

    def test_is_ordnance_weapon_with_ordnance_tag(self):
        """is_ordnance_weapon returns True for ordnance tag."""
        tags = [PilotGearTag(tag="ordnance")]
        assert is_ordnance_weapon(tags) is True

    def test_is_ordnance_weapon_without_ordnance_tag(self):
        """is_ordnance_weapon returns False without ordnance tag."""
        tags = [PilotGearTag(tag="loading")]
        assert is_ordnance_weapon(tags) is False

    def test_multiple_tags_handled_correctly(self):
        """Functions should correctly identify presence/absence with multiple tags."""
        tags = [
            PilotGearTag(tag="loading"),
            PilotGearTag(tag="ordnance"),
        ]
        assert is_loading_weapon(tags) is True
        assert is_ordnance_weapon(tags) is True
        assert is_sidearm_weapon(tags) is False
        assert is_archaic_weapon(tags) is False
        assert is_inaccurate_weapon(tags) is False


class TestWeaponActionType:
    """Test get_pilot_weapon_action_type helper."""

    def test_sidearm_returns_quick_action(self):
        """Sidearm weapons return 'quick' action type."""
        tags = [PilotGearTag(tag="sidearm")]
        assert get_pilot_weapon_action_type(tags) == "quick"

    def test_non_sidearm_returns_full_action(self):
        """Non-sidearm weapons return 'full' action type."""
        tags = [PilotGearTag(tag="inaccurate")]
        assert get_pilot_weapon_action_type(tags) == "full"

    def test_heavy_weapon_with_loading_returns_full(self):
        """Heavy ordnance weapon returns 'full' action type."""
        heavy = get_pilot_gear_definition("signature_weapon_heavy")
        assert heavy is not None
        assert get_pilot_weapon_action_type(heavy.tags) == "full"

    def test_combat_weapon_returns_full(self):
        """Combat weapon returns 'full' action type."""
        combat = get_pilot_gear_definition("alloy_composite_combat")
        assert combat is not None
        assert get_pilot_weapon_action_type(combat.tags) == "full"


class TestWeaponDifficultyModifier:
    """Test get_pilot_weapon_difficulty_modifier helper."""

    def test_inaccurate_weapon_returns_plus_one(self):
        """Inaccurate weapons add +1 Difficulty."""
        tags = [PilotGearTag(tag="inaccurate")]
        assert get_pilot_weapon_difficulty_modifier(tags) == 1

    def test_heavy_alloy_composite_is_inaccurate(self):
        """Alloy/Composite Weapon (Heavy) has inaccurate tag."""
        heavy = get_pilot_gear_definition("alloy_composite_heavy")
        assert heavy is not None
        assert get_pilot_weapon_difficulty_modifier(heavy.tags) == 1

    def test_accurate_weapon_returns_zero(self):
        """Weapons without inaccurate tag return 0."""
        tags = [PilotGearTag(tag="sidearm")]
        assert get_pilot_weapon_difficulty_modifier(tags) == 0

    def test_combat_weapon_returns_zero(self):
        """Combat weapon returns 0 difficulty modifier."""
        combat = get_pilot_gear_definition("alloy_composite_combat")
        assert combat is not None
        assert get_pilot_weapon_difficulty_modifier(combat.tags) == 0


class TestCanPilotWeaponDamageTarget:
    """Test can_pilot_weapon_damage_target enforcement helper."""

    def test_archaic_weapon_cannot_damage_mech(self):
        """Archaic weapons cannot harm mechs."""
        archaic = get_pilot_gear_definition("archaic_melee")
        assert archaic is not None
        can_damage, reason = can_pilot_weapon_damage_target(
            archaic, target_is_mech=True
        )
        assert can_damage is False
        assert "archaic" in reason.lower()

    def test_archaic_weapon_can_damage_pilot(self):
        """Archaic weapons can damage non-mech targets."""
        archaic = get_pilot_gear_definition("archaic_melee")
        assert archaic is not None
        can_damage, reason = can_pilot_weapon_damage_target(
            archaic, target_is_mech=False
        )
        assert can_damage is True
        assert reason == ""

    def test_archaic_ranged_cannot_damage_mech(self):
        """Archaic Ranged weapon cannot harm mechs."""
        archaic_ranged = get_pilot_gear_definition("archaic_ranged")
        assert archaic_ranged is not None
        can_damage, reason = can_pilot_weapon_damage_target(
            archaic_ranged, target_is_mech=True
        )
        assert can_damage is False

    def test_non_archaic_weapon_can_damage_mech(self):
        """Non-archaic weapons can damage mechs."""
        sidearm = get_pilot_gear_definition("signature_weapon_sidearm")
        assert sidearm is not None
        can_damage, reason = can_pilot_weapon_damage_target(
            sidearm, target_is_mech=True
        )
        assert can_damage is True
        assert reason == ""

    def test_combat_weapon_can_damage_mech(self):
        """Combat weapons can damage mechs."""
        combat = get_pilot_gear_definition("alloy_composite_combat")
        assert combat is not None
        can_damage, reason = can_pilot_weapon_damage_target(combat, target_is_mech=True)
        assert can_damage is True
        assert reason == ""

    def test_heavy_weapon_can_damage_mech(self):
        """Heavy weapons can damage mechs."""
        heavy = get_pilot_gear_definition("alloy_composite_heavy")
        assert heavy is not None
        can_damage, reason = can_pilot_weapon_damage_target(heavy, target_is_mech=True)
        assert can_damage is True


class TestThermalChargeObjectDamage:
    """Test thermal charge object damage per PR2 rules."""

    def test_thermal_charge_has_10_ap_object_damage(self):
        """Thermal charge deals 10 AP damage to objects (PR2 146)."""
        thermal_charge = get_pilot_gear_definition("thermal_charge")
        assert thermal_charge is not None
        assert len(thermal_charge.charges) == 1
        area = thermal_charge.charges[0].area
        assert area.object_damage is not None
        assert area.object_damage.damage_type == "energy"
        assert area.object_damage.flat == 10
        assert area.object_damage.ap is True

    def test_thermal_charge_objects_auto_hit(self):
        """Thermal charge automatically hits objects (PR2 146)."""
        thermal_charge = get_pilot_gear_definition("thermal_charge")
        assert thermal_charge is not None
        area = thermal_charge.charges[0].area
        assert area.objects_auto_hit is True

    def test_thermal_charge_has_standard_damage_vs_creatures(self):
        """Thermal charge deals 3 AP energy damage vs. evasion."""
        thermal_charge = get_pilot_gear_definition("thermal_charge")
        assert thermal_charge is not None
        area = thermal_charge.charges[0].area
        assert area.attack_vs == "evasion"
        assert area.damage is not None
        assert area.damage.flat == 3
        assert area.damage.ap is True


class TestAllPilotWeaponsHaveProfiles:
    """Verify all pilot weapon definitions have valid profiles."""

    @pytest.mark.parametrize(
        "weapon_id",
        [
            "archaic_melee",
            "alloy_composite_light",
            "alloy_composite_combat",
            "alloy_composite_heavy",
            "archaic_ranged",
            "signature_weapon_sidearm",
            "signature_weapon_combat",
            "signature_weapon_heavy",
        ],
    )
    def test_weapon_has_valid_profile(self, weapon_id):
        """Each weapon should have a valid weapon_profile."""
        weapon = get_pilot_gear_definition(weapon_id)
        assert weapon is not None, f"Weapon {weapon_id} not found"
        assert weapon.category == "weapon"
        assert weapon.weapon_profile is not None

    @pytest.mark.parametrize(
        "weapon_id",
        [
            "archaic_melee",
            "archaic_ranged",
        ],
    )
    def test_archaic_weapons_have_archaic_tag(self, weapon_id):
        """Archaic weapons should have the archaic tag."""
        weapon = get_pilot_gear_definition(weapon_id)
        assert weapon is not None
        assert is_archaic_weapon(weapon.tags)

    def test_heavy_alloy_composite_has_inaccurate_tag(self):
        """Alloy/Composite Weapon (Heavy) should have inaccurate tag."""
        weapon = get_pilot_gear_definition("alloy_composite_heavy")
        assert weapon is not None
        assert is_inaccurate_weapon(weapon.tags)

    def test_sidearm_has_sidearm_tag(self):
        """Signature Weapon (Sidearm) should have sidearm tag."""
        weapon = get_pilot_gear_definition("signature_weapon_sidearm")
        assert weapon is not None
        assert is_sidearm_weapon(weapon.tags)
