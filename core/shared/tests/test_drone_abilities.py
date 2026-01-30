"""
Tests for Drone Abilities Module

Tests all drone ability resolvers per PR2 5070-5088:
- Turret Drone: Reaction attack within range 10
- Restock Drone: Cool 1d6 heat, reload, clear condition
- Latch Drone: Mount attack with heal OR active buff + immunity
- ICEOUT Drone: Burst 1 zone with tech immunity
- Tracking Drone: Tech attack revealing info, negating hide/invis
- Hive Drone: Burst 2 zone with soft cover and entry damage
"""

import unittest
from core.shared.drone_abilities import (
    TurretDroneInput,
    TurretDroneResult,
    resolve_turret_drone,
    RestockDroneInput,
    RestockDroneResult,
    resolve_restock_drone,
    LatchDroneInput,
    LatchDroneResult,
    resolve_latch_drone,
    ICEOUTDroneInput,
    ICEOUTDroneResult,
    resolve_iceout_drone,
    TrackingDroneInput,
    TrackingDroneResult,
    resolve_tracking_drone,
    HiveDroneInput,
    HiveDroneResult,
    resolve_hive_drone,
    resolve_drone_ability,
    get_damage_for_tier,
    get_dc_for_tier,
)
from core.mech.grid import HexCoord, HexPosition


class TestGetDamageForTier(unittest.TestCase):
    """Tests for damage scaling by tier."""

    def test_tier_1_base_damage(self):
        """Tier 1 returns base damage."""
        self.assertEqual(get_damage_for_tier(3, 1), 3)
        self.assertEqual(get_damage_for_tier(1, 1), 1)

    def test_tier_2_scaling(self):
        """Tier 2 adds +2 to base damage."""
        self.assertEqual(get_damage_for_tier(3, 2), 5)
        self.assertEqual(get_damage_for_tier(1, 2), 3)

    def test_tier_3_scaling(self):
        """Tier 3 adds +4 to base damage."""
        self.assertEqual(get_damage_for_tier(3, 3), 7)
        self.assertEqual(get_damage_for_tier(1, 3), 5)


class TestGetDCForTier(unittest.TestCase):
    """Tests for DC scaling by tier."""

    def test_tier_1_base_dc(self):
        """Tier 1 returns base DC."""
        self.assertEqual(get_dc_for_tier(8, 1), 8)
        self.assertEqual(get_dc_for_tier(12, 1), 12)

    def test_tier_2_scaling(self):
        """Tier 2 adds +2 to base DC."""
        self.assertEqual(get_dc_for_tier(8, 2), 10)
        self.assertEqual(get_dc_for_tier(12, 2), 14)

    def test_tier_3_scaling(self):
        """Tier 3 adds +4 to base DC."""
        self.assertEqual(get_dc_for_tier(8, 3), 12)
        self.assertEqual(get_dc_for_tier(12, 3), 16)


class TestTurretDrone(unittest.TestCase):
    """Tests for Turret Drone ability per PR2 7344-7358."""

    def _make_positions(
        self, turret_q: int, turret_r: int, ally_q: int, ally_r: int
    ) -> tuple[HexPosition, HexPosition]:
        """Create turret and ally positions."""
        turret_pos = HexPosition(coord=HexCoord(q=turret_q, r=turret_r))
        ally_pos = HexPosition(coord=HexCoord(q=ally_q, r=ally_r))
        return turret_pos, ally_pos

    def test_turret_damage_on_hit_within_range(self):
        """Turret deals damage when ally hits within range 10."""
        turret_pos, ally_pos = self._make_positions(5, 5, 5, 6)

        result = resolve_turret_drone(
            TurretDroneInput(
                drone_id="turret-1",
                owner_id="mech-1",
                tier=1,
                ally_attack_hit=True,
                target_id="enemy-1",
                ally_position=ally_pos,
                turret_position=turret_pos,
                turret_range=10,
                base_damage=3,
            )
        )

        self.assertEqual(result.ability_type, "turret")
        self.assertTrue(result.success)
        self.assertTrue(result.damage_dealt)
        self.assertEqual(result.damage_amount, 3)
        self.assertEqual(result.damage_type, "kinetic")
        self.assertEqual(result.target_ids, ["enemy-1"])
        self.assertTrue(result.range_check)

    def test_turret_no_damage_on_miss(self):
        """Turret does not deal damage when ally misses."""
        turret_pos, ally_pos = self._make_positions(5, 5, 5, 6)

        result = resolve_turret_drone(
            TurretDroneInput(
                drone_id="turret-1",
                owner_id="mech-1",
                tier=1,
                ally_attack_hit=False,
                target_id="enemy-1",
                ally_position=ally_pos,
                turret_position=turret_pos,
                turret_range=10,
                base_damage=3,
            )
        )

        self.assertTrue(result.success)
        self.assertFalse(result.damage_dealt)
        self.assertEqual(result.damage_amount, 0)

    def test_turret_no_damage_outside_range(self):
        """Turret does not deal damage when target is beyond range."""
        turret_pos, ally_pos = self._make_positions(0, 0, 20, 20)

        result = resolve_turret_drone(
            TurretDroneInput(
                drone_id="turret-1",
                owner_id="mech-1",
                tier=1,
                ally_attack_hit=True,
                target_id="enemy-1",
                ally_position=ally_pos,
                turret_position=turret_pos,
                turret_range=10,
                base_damage=3,
            )
        )

        self.assertTrue(result.success)
        self.assertFalse(result.damage_dealt)
        self.assertFalse(result.range_check)

    def test_turret_damage_tier_2(self):
        """Turret damage scales with tier 2 (+2)."""
        turret_pos, ally_pos = self._make_positions(5, 5, 5, 6)

        result = resolve_turret_drone(
            TurretDroneInput(
                drone_id="turret-1",
                owner_id="mech-1",
                tier=2,
                ally_attack_hit=True,
                target_id="enemy-1",
                ally_position=ally_pos,
                turret_position=turret_pos,
                turret_range=10,
                base_damage=3,
            )
        )

        self.assertTrue(result.damage_dealt)
        self.assertEqual(result.damage_amount, 5)

    def test_turret_damage_tier_3(self):
        """Turret damage scales with tier 3 (+4)."""
        turret_pos, ally_pos = self._make_positions(5, 5, 5, 6)

        result = resolve_turret_drone(
            TurretDroneInput(
                drone_id="turret-1",
                owner_id="mech-1",
                tier=3,
                ally_attack_hit=True,
                target_id="enemy-1",
                ally_position=ally_pos,
                turret_position=turret_pos,
                turret_range=10,
                base_damage=3,
            )
        )

        self.assertTrue(result.damage_dealt)
        self.assertEqual(result.damage_amount, 7)

    def test_turret_at_exact_range(self):
        """Turret triggers at exact range 10."""
        turret_pos, ally_pos = self._make_positions(0, 0, 10, 0)

        result = resolve_turret_drone(
            TurretDroneInput(
                drone_id="turret-1",
                owner_id="mech-1",
                tier=1,
                ally_attack_hit=True,
                target_id="enemy-1",
                ally_position=ally_pos,
                turret_position=turret_pos,
                turret_range=10,
                base_damage=3,
            )
        )

        self.assertTrue(result.damage_dealt)
        self.assertTrue(result.range_check)

    def test_turret_at_one_beyond_range(self):
        """Turret does not trigger one space beyond range."""
        turret_pos, ally_pos = self._make_positions(0, 0, 11, 0)

        result = resolve_turret_drone(
            TurretDroneInput(
                drone_id="turret-1",
                owner_id="mech-1",
                tier=1,
                ally_attack_hit=True,
                target_id="enemy-1",
                ally_position=ally_pos,
                turret_position=turret_pos,
                turret_range=10,
                base_damage=3,
            )
        )

        self.assertFalse(result.damage_dealt)
        self.assertFalse(result.range_check)


class TestRestockDrone(unittest.TestCase):
    """Tests for Restock Drone ability per PR2 7833-7843."""

    def _make_positions(
        self, drone_q: int, drone_r: int, combatant_q: int, combatant_r: int
    ) -> tuple[HexPosition, HexPosition]:
        """Create drone and combatant positions."""
        drone_pos = HexPosition(coord=HexCoord(q=drone_q, r=drone_r))
        combatant_pos = HexPosition(coord=HexCoord(q=combatant_q, r=combatant_r))
        return drone_pos, combatant_pos

    def test_restock_cool_heat(self):
        """Restock drone can cool 1d6 heat when adjacent and primed."""
        drone_pos, combatant_pos = self._make_positions(5, 5, 5, 6)

        result = resolve_restock_drone(
            RestockDroneInput(
                drone_id="restock-1",
                owner_id="mech-1",
                tier=1,
                activating_combatant_id="mech-2",
                activating_combatant_position=combatant_pos,
                drone_position=drone_pos,
                action_choice="cool",
                is_primed=True,
                current_heat=5,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.can_activate)
        self.assertIsNotNone(result.heat_cooled)
        self.assertGreaterEqual(result.heat_cooled, 1)
        self.assertLessEqual(result.heat_cooled, 6)
        self.assertTrue(result.drone_consumed)

    def test_restock_cool_with_force_roll(self):
        """Restock drone uses forced roll for testing."""
        drone_pos, combatant_pos = self._make_positions(5, 5, 5, 6)

        result = resolve_restock_drone(
            RestockDroneInput(
                drone_id="restock-1",
                owner_id="mech-1",
                tier=1,
                activating_combatant_id="mech-2",
                activating_combatant_position=combatant_pos,
                drone_position=drone_pos,
                action_choice="cool",
                is_primed=True,
                force_cool_roll=4,
            )
        )

        self.assertEqual(result.heat_cooled, 4)

    def test_restock_reload(self):
        """Restock drone can reload a loading weapon."""
        drone_pos, combatant_pos = self._make_positions(5, 5, 5, 6)

        result = resolve_restock_drone(
            RestockDroneInput(
                drone_id="restock-1",
                owner_id="mech-1",
                tier=1,
                activating_combatant_id="mech-2",
                activating_combatant_position=combatant_pos,
                drone_position=drone_pos,
                action_choice="reload",
                is_primed=True,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.can_activate)
        self.assertTrue(result.weapon_reloaded)
        self.assertIsNone(result.heat_cooled)
        self.assertTrue(result.drone_consumed)

    def test_restock_clear_condition(self):
        """Restock drone can clear one condition."""
        drone_pos, combatant_pos = self._make_positions(5, 5, 5, 6)

        result = resolve_restock_drone(
            RestockDroneInput(
                drone_id="restock-1",
                owner_id="mech-1",
                tier=1,
                activating_combatant_id="mech-2",
                activating_combatant_position=combatant_pos,
                drone_position=drone_pos,
                action_choice="clear_condition",
                is_primed=True,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.can_activate)
        self.assertEqual(result.condition_cleared, "any")
        self.assertTrue(result.drone_consumed)

    def test_restock_not_adjacent(self):
        """Restock drone cannot activate if not adjacent."""
        drone_pos, combatant_pos = self._make_positions(0, 0, 5, 5)

        result = resolve_restock_drone(
            RestockDroneInput(
                drone_id="restock-1",
                owner_id="mech-1",
                tier=1,
                activating_combatant_id="mech-2",
                activating_combatant_position=combatant_pos,
                drone_position=drone_pos,
                action_choice="cool",
                is_primed=True,
            )
        )

        self.assertFalse(result.success)
        self.assertFalse(result.can_activate)
        self.assertFalse(result.drone_consumed)

    def test_restock_not_primed(self):
        """Restock drone cannot activate if not primed."""
        drone_pos, combatant_pos = self._make_positions(5, 5, 5, 6)

        result = resolve_restock_drone(
            RestockDroneInput(
                drone_id="restock-1",
                owner_id="mech-1",
                tier=1,
                activating_combatant_id="mech-2",
                activating_combatant_position=combatant_pos,
                drone_position=drone_pos,
                action_choice="cool",
                is_primed=False,
            )
        )

        self.assertFalse(result.success)
        self.assertFalse(result.can_activate)
        self.assertFalse(result.drone_consumed)

    def test_restock_adjacent_at_distance_1(self):
        """Restock drone activates when combatant is at distance 1."""
        drone_pos, combatant_pos = self._make_positions(5, 5, 6, 5)

        result = resolve_restock_drone(
            RestockDroneInput(
                drone_id="restock-1",
                owner_id="mech-1",
                tier=1,
                activating_combatant_id="mech-2",
                activating_combatant_position=combatant_pos,
                drone_position=drone_pos,
                action_choice="reload",
                is_primed=True,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.can_activate)


class TestLatchDrone(unittest.TestCase):
    """Tests for Latch Drone ability per PR2 7813-7831."""

    def _make_positions(
        self, shooter_q: int, shooter_r: int, target_q: int, target_r: int
    ) -> tuple[HexPosition, HexPosition]:
        """Create shooter and target positions."""
        shooter_pos = HexPosition(coord=HexCoord(q=shooter_q, r=shooter_r))
        target_pos = HexPosition(coord=HexCoord(q=target_q, r=target_r))
        return shooter_pos, target_pos

    def test_latch_mount_hit(self):
        """Latch drone mount attack hits on successful roll."""
        shooter_pos, target_pos = self._make_positions(5, 5, 5, 8)

        result = resolve_latch_drone(
            LatchDroneInput(
                drone_id="latch-1",
                owner_id="mech-1",
                tier=1,
                mode="mount",
                target_id="mech-2",
                target_position=target_pos,
                shooter_id="mech-1",
                shooter_position=shooter_pos,
                shooter_systems_bonus=3,
                force_roll=12,
                base_attack_dc=8,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.hit)
        self.assertEqual(result.attack_roll, 12)
        self.assertEqual(result.attack_total, 15)
        self.assertEqual(result.hp_healed, 5)

    def test_latch_mount_miss(self):
        """Latch drone mount attack misses on failed roll."""
        shooter_pos, target_pos = self._make_positions(5, 5, 5, 8)

        result = resolve_latch_drone(
            LatchDroneInput(
                drone_id="latch-1",
                owner_id="mech-1",
                tier=1,
                mode="mount",
                target_id="mech-2",
                target_position=target_pos,
                shooter_id="mech-1",
                shooter_position=shooter_pos,
                shooter_systems_bonus=0,
                force_roll=5,
                base_attack_dc=8,
            )
        )

        self.assertTrue(result.success)
        self.assertFalse(result.hit)
        self.assertEqual(result.attack_roll, 5)
        self.assertEqual(result.attack_total, 5)

    def test_latch_mount_tier_scaling(self):
        """Latch drone mount DC scales with tier."""
        shooter_pos, target_pos = self._make_positions(5, 5, 5, 8)

        result_tier1 = resolve_latch_drone(
            LatchDroneInput(
                drone_id="latch-1",
                owner_id="mech-1",
                tier=1,
                mode="mount",
                target_id="mech-2",
                target_position=target_pos,
                shooter_id="mech-1",
                shooter_position=shooter_pos,
                shooter_systems_bonus=0,
                force_roll=10,
                base_attack_dc=8,
            )
        )

        result_tier3 = resolve_latch_drone(
            LatchDroneInput(
                drone_id="latch-1",
                owner_id="mech-1",
                tier=3,
                mode="mount",
                target_id="mech-2",
                target_position=target_pos,
                shooter_id="mech-1",
                shooter_position=shooter_pos,
                shooter_systems_bonus=0,
                force_roll=10,
                base_attack_dc=8,
            )
        )

        self.assertTrue(result_tier1.hit)
        self.assertFalse(result_tier3.hit)

    def test_latch_active_requires_core_power(self):
        """Latch drone active mode requires core power."""
        shooter_pos, target_pos = self._make_positions(5, 5, 5, 8)

        result = resolve_latch_drone(
            LatchDroneInput(
                drone_id="latch-1",
                owner_id="mech-1",
                tier=1,
                mode="active",
                target_id="mech-2",
                target_position=target_pos,
                shooter_id="mech-1",
                shooter_position=shooter_pos,
                shooter_systems_bonus=3,
                has_core_power=False,
            )
        )

        self.assertFalse(result.success)
        self.assertIn("core power", result.reason)

    def test_latch_active_success(self):
        """Latch drone active mode grants buffs and immunities."""
        shooter_pos, target_pos = self._make_positions(5, 5, 5, 8)

        result = resolve_latch_drone(
            LatchDroneInput(
                drone_id="latch-1",
                owner_id="mech-1",
                tier=1,
                mode="active",
                target_id="mech-2",
                target_position=target_pos,
                shooter_id="mech-1",
                shooter_position=shooter_pos,
                shooter_systems_bonus=3,
                has_core_power=True,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.core_power_spent)
        self.assertEqual(result.heat_to_owner, 1)
        self.assertIn("+1 Accuracy", result.buffs_granted[0])
        self.assertIn("immune to impaired", result.immunities_granted[0])

    def test_latch_active_condition_immunities(self):
        """Latch drone active grants all condition immunities."""
        shooter_pos, target_pos = self._make_positions(5, 5, 5, 8)

        result = resolve_latch_drone(
            LatchDroneInput(
                drone_id="latch-1",
                owner_id="mech-1",
                tier=1,
                mode="active",
                target_id="mech-2",
                target_position=target_pos,
                shooter_id="mech-1",
                shooter_position=shooter_pos,
                shooter_systems_bonus=3,
                has_core_power=True,
            )
        )

        self.assertEqual(len(result.immunities_granted), 5)
        self.assertIn("immune to impaired", result.immunities_granted)
        self.assertIn("immune to jammed", result.immunities_granted)
        self.assertIn("immune to slowed", result.immunities_granted)
        self.assertIn("immune to shredded", result.immunities_granted)
        self.assertIn("immune to immobilized", result.immunities_granted)


class TestICEOUTDrone(unittest.TestCase):
    """Tests for ICEOUT Drone ability per PR2 8645-8658."""

    def test_iceout_creates_zone(self):
        """ICEOUT drone creates burst 1 zone."""
        drone_pos = HexPosition(coord=HexCoord(q=5, r=5))

        result = resolve_iceout_drone(
            ICEOUTDroneInput(
                drone_id="iceout-1",
                owner_id="mech-1",
                tier=1,
                drone_position=drone_pos,
                affected_combatant_ids=["mech-2", "mech-3"],
                zone_size=1,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.zone_created)
        self.assertEqual(result.zone_size, 1)
        self.assertEqual(len(result.affected_combatant_ids), 2)
        self.assertEqual(len(result.tech_immunity_granted), 2)

    def test_iceout_move_action(self):
        """ICEOUT drone can move to new position."""
        new_pos = HexPosition(coord=HexCoord(q=10, r=10))

        result = resolve_iceout_drone(
            ICEOUTDroneInput(
                drone_id="iceout-1",
                owner_id="mech-1",
                tier=1,
                drone_position=new_pos,
                affected_combatant_ids=[],
                zone_size=1,
                is_moving=True,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.zone_created)
        self.assertFalse(result.can_move)

    def test_iceout_no_affected_combatants(self):
        """ICEOUT drone creates zone even with no combatants."""
        drone_pos = HexPosition(coord=HexCoord(q=5, r=5))

        result = resolve_iceout_drone(
            ICEOUTDroneInput(
                drone_id="iceout-1",
                owner_id="mech-1",
                tier=1,
                drone_position=drone_pos,
                affected_combatant_ids=[],
                zone_size=1,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.zone_created)
        self.assertEqual(len(result.affected_combatant_ids), 0)
        self.assertEqual(len(result.tech_immunity_granted), 0)


class TestTrackingDrone(unittest.TestCase):
    """Tests for Tracking Drone ability per PR2 8778-8789."""

    def test_tracking_hit_reveals_info(self):
        """Tracking drone hit reveals target information."""
        result = resolve_tracking_drone(
            TrackingDroneInput(
                drone_id="tracking-1",
                owner_id="mech-1",
                tier=1,
                target_id="enemy-1",
                shooter_id="mech-1",
                shooter_systems_bonus=4,
                force_roll=15,
                base_dc=12,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.hit)
        self.assertEqual(result.attack_roll, 15)
        self.assertEqual(result.attack_total, 19)
        self.assertTrue(result.drone_attached)
        self.assertTrue(result.hide_negated)
        self.assertTrue(result.invisibility_ignored)
        self.assertEqual(result.remove_dc, 12)
        self.assertEqual(result.information_revealed["location"], "exact")

    def test_tracking_miss(self):
        """Tracking drone miss does not attach."""
        result = resolve_tracking_drone(
            TrackingDroneInput(
                drone_id="tracking-1",
                owner_id="mech-1",
                tier=1,
                target_id="enemy-1",
                shooter_id="mech-1",
                shooter_systems_bonus=0,
                force_roll=5,
                base_dc=12,
            )
        )

        self.assertTrue(result.success)
        self.assertFalse(result.hit)
        self.assertFalse(result.drone_attached)

    def test_tracking_remove_dc_tier_scaling(self):
        """Tracking drone remove DC scales with tier."""
        result_tier1 = resolve_tracking_drone(
            TrackingDroneInput(
                drone_id="tracking-1",
                owner_id="mech-1",
                tier=1,
                target_id="enemy-1",
                shooter_id="mech-1",
                shooter_systems_bonus=0,
                force_roll=12,
                base_dc=12,
            )
        )

        result_tier3 = resolve_tracking_drone(
            TrackingDroneInput(
                drone_id="tracking-1",
                owner_id="mech-1",
                tier=3,
                target_id="enemy-1",
                shooter_id="mech-1",
                shooter_systems_bonus=0,
                force_roll=12,
                base_dc=12,
            )
        )

        self.assertTrue(result_tier1.hit)
        self.assertFalse(result_tier3.hit)

    def test_tracking_revealed_information(self):
        """Tracking drone reveals all specified information."""
        result = resolve_tracking_drone(
            TrackingDroneInput(
                drone_id="tracking-1",
                owner_id="mech-1",
                tier=1,
                target_id="enemy-1",
                shooter_id="mech-1",
                shooter_systems_bonus=5,
                force_roll=10,
                base_dc=12,
            )
        )

        info = result.information_revealed
        self.assertEqual(info["location"], "exact")
        self.assertEqual(info["hp"], "visible")
        self.assertEqual(info["structure"], "visible")
        self.assertEqual(info["heat"], "visible")
        self.assertEqual(info["speed"], "visible")


class TestHiveDrone(unittest.TestCase):
    """Tests for Hive Drone ability per PR2 9745-9759."""

    def test_hive_creates_zone(self):
        """Hive drone creates burst 2 zone."""
        drone_pos = HexPosition(coord=HexCoord(q=5, r=5))

        result = resolve_hive_drone(
            HiveDroneInput(
                drone_id="hive-1",
                owner_id="mech-1",
                tier=1,
                drone_position=drone_pos,
                enemy_ids=["enemy-1", "enemy-2"],
                ally_ids=["mech-2"],
                zone_size=2,
                base_damage=1,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.zone_created)
        self.assertEqual(result.zone_size, 2)
        self.assertEqual(len(result.allies_covered), 1)
        self.assertEqual(len(result.enemies_damaged), 2)
        self.assertEqual(result.damage_per_target, 1)
        self.assertEqual(result.damage_type, "kinetic")

    def test_hive_damage_tier_scaling(self):
        """Hive drone damage scales with tier."""
        drone_pos = HexPosition(coord=HexCoord(q=5, r=5))

        result_tier1 = resolve_hive_drone(
            HiveDroneInput(
                drone_id="hive-1",
                owner_id="mech-1",
                tier=1,
                drone_position=drone_pos,
                enemy_ids=["enemy-1"],
                ally_ids=[],
                zone_size=2,
                base_damage=1,
            )
        )

        result_tier3 = resolve_hive_drone(
            HiveDroneInput(
                drone_id="hive-1",
                owner_id="mech-1",
                tier=3,
                drone_position=drone_pos,
                enemy_ids=["enemy-1"],
                ally_ids=[],
                zone_size=2,
                base_damage=1,
            )
        )

        self.assertEqual(result_tier1.damage_per_target, 1)
        self.assertEqual(result_tier3.damage_per_target, 5)  # 1 + (3-1)*2 = 5

    def test_hive_soft_cover(self):
        """Hive drone grants soft cover to allies."""
        drone_pos = HexPosition(coord=HexCoord(q=5, r=5))

        result = resolve_hive_drone(
            HiveDroneInput(
                drone_id="hive-1",
                owner_id="mech-1",
                tier=1,
                drone_position=drone_pos,
                enemy_ids=[],
                ally_ids=["mech-2", "mech-3"],
                zone_size=2,
                base_damage=1,
            )
        )

        self.assertEqual(len(result.soft_cover_granted), 2)
        self.assertIn("mech-2", result.soft_cover_granted)
        self.assertIn("mech-3", result.soft_cover_granted)

    def test_hive_move_action(self):
        """Hive drone can move to new position."""
        new_pos = HexPosition(coord=HexCoord(q=10, r=10))

        result = resolve_hive_drone(
            HiveDroneInput(
                drone_id="hive-1",
                owner_id="mech-1",
                tier=1,
                drone_position=new_pos,
                enemy_ids=[],
                ally_ids=[],
                zone_size=2,
                base_damage=1,
                is_move=True,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.zone_created)
        self.assertFalse(result.can_move)

    def test_hive_no_enemies_or_allies(self):
        """Hive drone creates zone even with no combatants."""
        drone_pos = HexPosition(coord=HexCoord(q=5, r=5))

        result = resolve_hive_drone(
            HiveDroneInput(
                drone_id="hive-1",
                owner_id="mech-1",
                tier=1,
                drone_position=drone_pos,
                enemy_ids=[],
                ally_ids=[],
                zone_size=2,
                base_damage=1,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.zone_created)
        self.assertEqual(len(result.enemies_damaged), 0)
        self.assertEqual(len(result.allies_covered), 0)


class TestDroneAbilityDispatch(unittest.TestCase):
    """Tests for the drone ability dispatch function."""

    def test_dispatch_turret(self):
        """Dispatch routes turret ability correctly."""
        turret_pos = HexPosition(coord=HexCoord(q=5, r=5))
        ally_pos = HexPosition(coord=HexCoord(q=5, r=6))

        input_model = TurretDroneInput(
            drone_id="turret-1",
            owner_id="mech-1",
            tier=1,
            ally_attack_hit=True,
            target_id="enemy-1",
            ally_position=ally_pos,
            turret_position=turret_pos,
        )

        result = resolve_drone_ability(input_model)
        self.assertEqual(result.ability_type, "turret")
        self.assertTrue(isinstance(result, TurretDroneResult))

    def test_dispatch_restock(self):
        """Dispatch routes restock ability correctly."""
        drone_pos = HexPosition(coord=HexCoord(q=5, r=5))
        combatant_pos = HexPosition(coord=HexCoord(q=5, r=6))

        input_model = RestockDroneInput(
            drone_id="restock-1",
            owner_id="mech-1",
            tier=1,
            activating_combatant_id="mech-2",
            activating_combatant_position=combatant_pos,
            drone_position=drone_pos,
            action_choice="cool",
            is_primed=True,
        )

        result = resolve_drone_ability(input_model)
        self.assertEqual(result.ability_type, "restock")
        self.assertTrue(isinstance(result, RestockDroneResult))

    def test_dispatch_latch(self):
        """Dispatch routes latch ability correctly."""
        shooter_pos = HexPosition(coord=HexCoord(q=5, r=5))
        target_pos = HexPosition(coord=HexCoord(q=5, r=8))

        input_model = LatchDroneInput(
            drone_id="latch-1",
            owner_id="mech-1",
            tier=1,
            mode="mount",
            target_id="mech-2",
            target_position=target_pos,
            shooter_id="mech-1",
            shooter_position=shooter_pos,
            shooter_systems_bonus=3,
        )

        result = resolve_drone_ability(input_model)
        self.assertEqual(result.ability_type, "latch")
        self.assertTrue(isinstance(result, LatchDroneResult))

    def test_dispatch_iceout(self):
        """Dispatch routes iceout ability correctly."""
        drone_pos = HexPosition(coord=HexCoord(q=5, r=5))

        input_model = ICEOUTDroneInput(
            drone_id="iceout-1",
            owner_id="mech-1",
            tier=1,
            drone_position=drone_pos,
        )

        result = resolve_drone_ability(input_model)
        self.assertEqual(result.ability_type, "iceout")
        self.assertTrue(isinstance(result, ICEOUTDroneResult))

    def test_dispatch_tracking(self):
        """Dispatch routes tracking ability correctly."""
        input_model = TrackingDroneInput(
            drone_id="tracking-1",
            owner_id="mech-1",
            tier=1,
            target_id="enemy-1",
            shooter_id="mech-1",
            shooter_systems_bonus=4,
        )

        result = resolve_drone_ability(input_model)
        self.assertEqual(result.ability_type, "tracking")
        self.assertTrue(isinstance(result, TrackingDroneResult))

    def test_dispatch_hive(self):
        """Dispatch routes hive ability correctly."""
        drone_pos = HexPosition(coord=HexCoord(q=5, r=5))

        input_model = HiveDroneInput(
            drone_id="hive-1",
            owner_id="mech-1",
            tier=1,
            drone_position=drone_pos,
        )

        result = resolve_drone_ability(input_model)
        self.assertEqual(result.ability_type, "hive")
        self.assertTrue(isinstance(result, HiveDroneResult))


if __name__ == "__main__":
    unittest.main()
