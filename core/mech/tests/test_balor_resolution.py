"""Tests for Balor frame effects resolution helpers."""

from __future__ import annotations


from core.mech.combat_resolution import (
    BalorScouringSwarmResult,
    BalorRegenerationResult,
    BalorSelfPerpetuatingResult,
    HellswarmProtocolResult,
    HiveDroneResult,
    SwarmBodyResult,
    resolve_scouring_swarm,
    resolve_balor_regeneration,
    resolve_self_perpetuating,
    activate_hellswarm_protocol,
    deploy_hive_drone,
    resolve_hive_drone_turn_start,
    resolve_hive_drone_zone_entry,
    activate_swarm_body,
    resolve_swarm_body_turn_start,
    resolve_swarm_body_zone_entry,
    end_swarm_body_condition,
    ResolutionSettings,
)
from core.mech.grid import HexCoord


class TestBalorScouringSwarmResult:
    """Tests for BalorScouringSwarmResult model."""

    def test_create_with_all_fields(self):
        """Test creating a result with all fields populated."""
        result = BalorScouringSwarmResult(
            damage_dealt=True,
            damage_per_target=4,
            affected_targets=["target1", "target2"],
            is_core_power_active=True,
        )
        assert result.damage_dealt is True
        assert result.damage_per_target == 4
        assert len(result.affected_targets) == 2
        assert result.is_core_power_active is True

    def test_create_with_no_targets(self):
        """Test creating a result when no targets are affected."""
        result = BalorScouringSwarmResult(
            damage_dealt=False,
            damage_per_target=2,
            affected_targets=[],
            is_core_power_active=False,
        )
        assert result.damage_dealt is False
        assert result.damage_per_target == 2
        assert len(result.affected_targets) == 0

    def test_damage_increases_with_core_power(self):
        """Test that damage increases when core power is active."""
        result_inactive = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=False,
            affected_target_ids=["enemy_1"],
        )
        result_active = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=True,
            affected_target_ids=["enemy_1"],
        )
        assert result_inactive.damage_per_target == 2
        assert result_active.damage_per_target == 4

    def test_core_power_flag_reflects_state(self):
        """Test that is_core_power_active reflects the provided state."""
        result = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=True,
            affected_target_ids=[],
        )
        assert result.is_core_power_active is True


class TestResolveScouringSwarm:
    """Tests for resolve_scouring_swarm function."""

    def test_damage_dealt_with_targets(self):
        """Test that damage is marked as dealt when targets exist."""
        result = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=False,
            affected_target_ids=["enemy_1", "enemy_2"],
        )
        assert result.damage_dealt is True
        assert len(result.affected_targets) == 2

    def test_no_damage_without_targets(self):
        """Test that damage is not dealt when no targets exist."""
        result = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=False,
            affected_target_ids=[],
        )
        assert result.damage_dealt is False
        assert len(result.affected_targets) == 0

    def test_damage_with_core_power_active(self):
        """Test damage calculation with core power active."""
        result = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=True,
            affected_target_ids=["enemy_1"],
        )
        assert result.damage_per_target == 4

    def test_damage_without_core_power(self):
        """Test damage calculation without core power."""
        result = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=False,
            affected_target_ids=["enemy_1"],
        )
        assert result.damage_per_target == 2

    def test_zone_parameters_passed(self):
        """Test that zone parameters are accepted."""
        result = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=False,
            affected_target_ids=[],
            zone_shape="cone",
            zone_size=2,
        )
        assert result is not None


class TestBalorRegenerationResult:
    """Tests for BalorRegenerationResult model."""

    def test_create_with_healing(self):
        """Test creating a result with healing applied."""
        result = BalorRegenerationResult(
            healing_applied=True,
            healing_amount=3,
            was_paused=False,
            pause_reason=None,
        )
        assert result.healing_applied is True
        assert result.healing_amount == 3
        assert result.was_paused is False

    def test_create_with_pause(self):
        """Test creating a result when regeneration is paused."""
        result = BalorRegenerationResult(
            healing_applied=False,
            healing_amount=0,
            was_paused=True,
            pause_reason="overheated",
        )
        assert result.healing_applied is False
        assert result.was_paused is True
        assert result.pause_reason == "overheated"


class TestResolveBalorRegeneration:
    """Tests for resolve_balor_regeneration function."""

    def test_healing_applied_when_no_pause(self):
        """Test that healing is applied when no pause conditions exist."""
        result = resolve_balor_regeneration(
            combatant_id="balor_1",
            max_hp=12,
            is_overheated=False,
            has_structure_damage=False,
            is_core_power_active=False,
            current_hp=8,
        )
        assert result.healing_applied is True
        assert result.healing_amount == 3  # 12 // 4 = 3
        assert result.was_paused is False

    def test_paused_when_overheated(self):
        """Test that regeneration pauses when overheated."""
        result = resolve_balor_regeneration(
            combatant_id="balor_1",
            max_hp=12,
            is_overheated=True,
            has_structure_damage=False,
            is_core_power_active=False,
            current_hp=8,
        )
        assert result.healing_applied is False
        assert result.was_paused is True
        assert result.pause_reason is not None and "overheated" in result.pause_reason

    def test_paused_when_structure_damage(self):
        """Test that regeneration pauses when structure damaged."""
        result = resolve_balor_regeneration(
            combatant_id="balor_1",
            max_hp=12,
            is_overheated=False,
            has_structure_damage=True,
            is_core_power_active=False,
            current_hp=8,
        )
        assert result.healing_applied is False
        assert result.was_paused is True
        assert (
            result.pause_reason is not None
            and "structure_damage" in result.pause_reason
        )

    def test_paused_when_core_power_active(self):
        """Test that regeneration pauses when core power is active."""
        result = resolve_balor_regeneration(
            combatant_id="balor_1",
            max_hp=12,
            is_overheated=False,
            has_structure_damage=False,
            is_core_power_active=True,
            current_hp=8,
        )
        assert result.healing_applied is False
        assert result.was_paused is True
        assert (
            result.pause_reason is not None
            and "core_power_active" in result.pause_reason
        )

    def test_paused_with_multiple_reasons(self):
        """Test pause when multiple conditions are true."""
        result = resolve_balor_regeneration(
            combatant_id="balor_1",
            max_hp=12,
            is_overheated=True,
            has_structure_damage=True,
            is_core_power_active=True,
            current_hp=8,
        )
        assert result.was_paused is True
        assert result.pause_reason is not None
        assert "overheated" in result.pause_reason
        assert "structure_damage" in result.pause_reason
        assert "core_power_active" in result.pause_reason

    def test_healing_capped_at_max_hp(self):
        """Test that healing does not exceed max HP."""
        result = resolve_balor_regeneration(
            combatant_id="balor_1",
            max_hp=12,
            is_overheated=False,
            has_structure_damage=False,
            is_core_power_active=False,
            current_hp=11,
        )
        assert result.healing_amount == 1
        assert result.healing_applied is True

    def test_no_healing_when_at_max_hp(self):
        """Test that no healing is applied when at max HP."""
        result = resolve_balor_regeneration(
            combatant_id="balor_1",
            max_hp=12,
            is_overheated=False,
            has_structure_damage=False,
            is_core_power_active=False,
            current_hp=12,
        )
        assert result.healing_amount == 0
        assert result.healing_applied is False

    def test_healing_amount_calculation(self):
        """Test various max HP values for healing calculation."""
        for max_hp, expected in [(12, 3), (8, 2), (10, 2), (6, 1)]:
            result = resolve_balor_regeneration(
                combatant_id="balor_1",
                max_hp=max_hp,
                is_overheated=False,
                has_structure_damage=False,
                is_core_power_active=False,
                current_hp=0,
            )
            assert result.healing_amount == expected


class TestBalorSelfPerpetuatingResult:
    """Tests for BalorSelfPerpetuatingResult model."""

    def test_create_with_restoration(self):
        """Test creating a result with HP restoration."""
        result = BalorSelfPerpetuatingResult(
            hp_restored=True,
            previous_hp=5,
            new_hp=12,
        )
        assert result.hp_restored is True
        assert result.previous_hp == 5
        assert result.new_hp == 12


class TestResolveSelfPerpetuating:
    """Tests for resolve_self_perpetuating function."""

    def test_full_restoration_on_rest(self):
        """Test that full HP is restored during rest action."""
        result = resolve_self_perpetuating(
            combatant_id="balor_1",
            current_hp=5,
            max_hp=12,
            is_during_rest=True,
        )
        assert result.hp_restored is True
        assert result.previous_hp == 5
        assert result.new_hp == 12

    def test_no_restoration_without_rest(self):
        """Test that no restoration happens outside rest action."""
        result = resolve_self_perpetuating(
            combatant_id="balor_1",
            current_hp=5,
            max_hp=12,
            is_during_rest=False,
        )
        assert result.hp_restored is False
        assert result.previous_hp == 5
        assert result.new_hp == 5


class TestHellswarmProtocolResult:
    """Tests for HellswarmProtocolResult model."""

    def test_create_full_result(self):
        """Test creating a complete protocol result."""
        result = HellswarmProtocolResult(
            cover_granted=True,
            soft_cover_targets=["balor_1", "ally_1"],
            shredded_applied=True,
            structure_avoidance_triggered=True,
            structure_roll=[6],
            avoided_structure_damage=True,
            healing_applied=False,
            healing_amount=None,
        )
        assert result.cover_granted is True
        assert len(result.soft_cover_targets) == 2
        assert result.avoided_structure_damage is True


class TestActivateHellswarmProtocol:
    """Tests for activate_hellswarm_protocol function."""

    def test_cover_granted_to_self_and_allies(self):
        """Test that cover is granted to self and adjacent allies."""
        result = activate_hellswarm_protocol(
            combatant_id="balor_1",
            adjacent_ally_ids=["ally_1", "ally_2"],
            structure_damage_marked=0,
            current_hp=8,
            max_hp=12,
        )
        assert result.cover_granted is True
        assert "balor_1" in result.soft_cover_targets
        assert "ally_1" in result.soft_cover_targets
        assert "ally_2" in result.soft_cover_targets

    def test_shredded_applied(self):
        """Test that shredded status is applied."""
        result = activate_hellswarm_protocol(
            combatant_id="balor_1",
            adjacent_ally_ids=[],
            structure_damage_marked=0,
            current_hp=8,
            max_hp=12,
        )
        assert result.shredded_applied is True

    def test_structure_avoidance_on_structure_damage(self):
        """Test structure avoidance when structure damage is marked."""
        result = activate_hellswarm_protocol(
            combatant_id="balor_1",
            adjacent_ally_ids=[],
            structure_damage_marked=1,
            current_hp=8,
            max_hp=12,
        )
        assert result.structure_avoidance_triggered is True
        assert result.structure_roll is not None
        assert len(result.structure_roll) == 1

    def test_structure_avoidance_success(self):
        """Test successful structure avoidance (roll >= 6)."""
        result = activate_hellswarm_protocol(
            combatant_id="balor_1",
            adjacent_ally_ids=[],
            structure_damage_marked=1,
            current_hp=1,
            max_hp=12,
            settings=ResolutionSettings(forced_rolls=[6]),
        )
        assert result.avoided_structure_damage is True

    def test_structure_avoidance_failure(self):
        """Test failed structure avoidance (roll < 6)."""
        result = activate_hellswarm_protocol(
            combatant_id="balor_1",
            adjacent_ally_ids=[],
            structure_damage_marked=1,
            current_hp=1,
            max_hp=12,
            settings=ResolutionSettings(forced_rolls=[3]),
        )
        assert result.avoided_structure_damage is False

    def test_healing_applied_when_no_structure_damage(self):
        """Test that healing is applied when no structure damage."""
        result = activate_hellswarm_protocol(
            combatant_id="balor_1",
            adjacent_ally_ids=[],
            structure_damage_marked=0,
            current_hp=4,
            max_hp=12,
        )
        assert result.healing_applied is True
        assert result.healing_amount == 6  # half of 12

    def test_healing_capped_at_max_hp(self):
        """Test that healing does not exceed max HP."""
        result = activate_hellswarm_protocol(
            combatant_id="balor_1",
            adjacent_ally_ids=[],
            structure_damage_marked=0,
            current_hp=10,
            max_hp=12,
        )
        assert result.healing_amount == 2


class TestHiveDroneResult:
    """Tests for HiveDroneResult model."""

    def test_create_deployed_result(self):
        """Test creating a result for drone deployment."""
        result = HiveDroneResult(
            drone_deployed=True,
            drone_position=HexCoord(q=5, r=5),
            damage_dealt=False,
            affected_targets=[],
        )
        assert result.drone_deployed is True
        assert result.drone_position == (5, 5)
        assert result.damage_dealt is False

    def test_create_damage_result(self):
        """Test creating a result for drone damage."""
        result = HiveDroneResult(
            drone_deployed=True,
            drone_position=HexCoord(q=5, r=5),
            damage_dealt=True,
            affected_targets=["enemy_1"],
        )
        assert result.damage_dealt is True
        assert "enemy_1" in result.affected_targets


class TestDeployHiveDrone:
    """Tests for deploy_hive_drone function."""

    def test_deployment_success(self):
        """Test successful drone deployment."""
        result = deploy_hive_drone(
            combatant_id="balor_1",
            deploy_position=HexCoord(q=5, r=5),
            enemy_target_ids=["enemy_1"],
        )
        assert result.drone_deployed is True
        assert result.drone_position == (5, 5)
        assert result.damage_dealt is False

    def test_deployment_preserves_targets(self):
        """Test that enemy targets are preserved for later use."""
        result = deploy_hive_drone(
            combatant_id="balor_1",
            deploy_position=HexCoord(q=3, r=7),
            enemy_target_ids=["enemy_1", "enemy_2"],
        )
        assert result is not None


class TestResolveHiveDroneTurnStart:
    """Tests for resolve_hive_drone_turn_start function."""

    def test_damage_to_enemies(self):
        """Test damage to enemies in drone zone at turn start."""
        result = resolve_hive_drone_turn_start(
            drone_position=HexCoord(q=5, r=5),
            enemy_target_ids=["enemy_1", "enemy_2"],
        )
        assert result.damage_dealt is True
        assert len(result.affected_targets) == 2

    def test_no_damage_without_enemies(self):
        """Test no damage when no enemies in zone."""
        result = resolve_hive_drone_turn_start(
            drone_position=HexCoord(q=5, r=5),
            enemy_target_ids=[],
        )
        assert result.damage_dealt is False
        assert len(result.affected_targets) == 0

    def test_zone_size_parameter(self):
        """Test that zone size parameter is accepted."""
        result = resolve_hive_drone_turn_start(
            drone_position=HexCoord(q=5, r=5),
            enemy_target_ids=[],
            zone_size=3,
        )
        assert result is not None


class TestResolveHiveDroneZoneEntry:
    """Tests for resolve_hive_drone_zone_entry function."""

    def test_damage_on_entry(self):
        """Test damage to combatant entering zone."""
        result = resolve_hive_drone_zone_entry(
            entering_combatant_id="enemy_1",
            drone_position=HexCoord(q=5, r=5),
        )
        assert result.damage_dealt is True
        assert "enemy_1" in result.affected_targets

    def test_single_target_affected(self):
        """Test that only the entering combatant is affected."""
        result = resolve_hive_drone_zone_entry(
            entering_combatant_id="enemy_1",
            drone_position=HexCoord(q=5, r=5),
        )
        assert len(result.affected_targets) == 1
        assert result.affected_targets[0] == "enemy_1"


class TestSwarmBodyResult:
    """Tests for SwarmBodyResult model."""

    def test_create_active_zone(self):
        """Test creating a result for active zone."""
        result = SwarmBodyResult(
            zone_active=True,
            condition_met=True,
            save_triggered=True,
            affected_targets=["enemy_1"],
            damage_per_target=3,
            damage_applied_to=["enemy_1"],
            condition_ended=False,
        )
        assert result.zone_active is True
        assert result.condition_met is True
        assert result.save_triggered is True
        assert len(result.damage_applied_to) == 1

    def test_create_inactive_zone(self):
        """Test creating a result for inactive zone (after move)."""
        result = SwarmBodyResult(
            zone_active=False,
            condition_met=False,
            save_triggered=False,
            affected_targets=[],
            damage_per_target=3,
            damage_applied_to=[],
            condition_ended=True,
        )
        assert result.zone_active is False
        assert result.condition_ended is True


class TestActivateSwarmBody:
    """Tests for activate_swarm_body function."""

    def test_zone_active_when_no_movement(self):
        """Test that zone is active when Balor hasn't moved."""
        result = activate_swarm_body(
            combatant_id="balor_1",
            current_hp=8,
            has_moved_this_turn=False,
            enemy_target_ids=["enemy_1"],
        )
        assert result.zone_active is True
        assert result.condition_met is True

    def test_zone_inactive_after_movement(self):
        """Test that zone is inactive after Balor moves."""
        result = activate_swarm_body(
            combatant_id="balor_1",
            current_hp=8,
            has_moved_this_turn=True,
            enemy_target_ids=["enemy_1"],
        )
        assert result.zone_active is False
        assert result.condition_met is False

    def test_damage_per_target_constant(self):
        """Test that damage per target is always 3."""
        result = activate_swarm_body(
            combatant_id="balor_1",
            current_hp=8,
            has_moved_this_turn=False,
            enemy_target_ids=[],
        )
        assert result.damage_per_target == 3


class TestResolveSwarmBodyTurnStart:
    """Tests for resolve_swarm_body_turn_start function."""

    def test_save_triggered_for_enemies(self):
        """Test that save is triggered for enemies in zone."""
        result = resolve_swarm_body_turn_start(
            combatant_id="balor_1",
            has_moved_this_turn=False,
            enemy_target_ids=["enemy_1", "enemy_2"],
        )
        assert result.save_triggered is True
        assert len(result.affected_targets) == 2

    def test_no_save_after_movement(self):
        """Test that no save is triggered after movement."""
        result = resolve_swarm_body_turn_start(
            combatant_id="balor_1",
            has_moved_this_turn=True,
            enemy_target_ids=["enemy_1"],
        )
        assert result.save_triggered is False
        assert result.condition_ended is True

    def test_all_targets_affected(self):
        """Test that all enemies in zone are affected."""
        result = resolve_swarm_body_turn_start(
            combatant_id="balor_1",
            has_moved_this_turn=False,
            enemy_target_ids=["enemy_1", "enemy_2", "enemy_3"],
        )
        assert len(result.affected_targets) == 3
        assert len(result.damage_applied_to) == 3


class TestResolveSwarmBodyZoneEntry:
    """Tests for resolve_swarm_body_zone_entry function."""

    def test_save_on_entry_to_active_zone(self):
        """Test that save is triggered on entry to active zone."""
        result = resolve_swarm_body_zone_entry(
            entering_combatant_id="enemy_1",
            balor_has_moved=False,
        )
        assert result.save_triggered is True
        assert "enemy_1" in result.affected_targets

    def test_no_save_on_entry_to_inactive_zone(self):
        """Test that no save is triggered on entry to inactive zone."""
        result = resolve_swarm_body_zone_entry(
            entering_combatant_id="enemy_1",
            balor_has_moved=True,
        )
        assert result.save_triggered is False
        assert result.zone_active is False

    def test_damage_applied_on_failed_save(self):
        """Test that damage is applied to entering combatant."""
        result = resolve_swarm_body_zone_entry(
            entering_combatant_id="enemy_1",
            balor_has_moved=False,
        )
        assert "enemy_1" in result.damage_applied_to
        assert result.damage_per_target == 3


class TestEndSwarmBodyCondition:
    """Tests for end_swarm_body_condition function."""

    def test_condition_ended(self):
        """Test that condition is marked as ended."""
        result = end_swarm_body_condition(combatant_id="balor_1")
        assert result.condition_ended is True
        assert result.zone_active is False
        assert result.condition_met is False

    def test_no_targets_affected(self):
        """Test that no targets are affected when ending."""
        result = end_swarm_body_condition(combatant_id="balor_1")
        assert len(result.affected_targets) == 0
        assert len(result.damage_applied_to) == 0


class TestBalorResolutionIntegration:
    """Integration tests for Balor resolution helpers."""

    def test_full_hellswarm_activation_sequence(self):
        """Test the complete HELLSWARM protocol activation sequence."""
        result = activate_hellswarm_protocol(
            combatant_id="balor_1",
            adjacent_ally_ids=["ally_1"],
            structure_damage_marked=0,
            current_hp=4,
            max_hp=12,
        )
        assert result.cover_granted
        assert result.shredded_applied
        assert result.healing_applied
        assert result.healing_amount == 6

    def test_full_hive_drone_lifecycle(self):
        """Test the complete Hive Drone deployment and damage sequence."""
        drone_pos = HexCoord(q=5, r=5)
        deploy_result = deploy_hive_drone(
            combatant_id="balor_1",
            deploy_position=drone_pos,
            enemy_target_ids=["enemy_1"],
        )
        turn_start_result = resolve_hive_drone_turn_start(
            drone_position=drone_pos,
            enemy_target_ids=["enemy_1"],
        )
        entry_result = resolve_hive_drone_zone_entry(
            entering_combatant_id="enemy_2",
            drone_position=drone_pos,
        )
        assert deploy_result.drone_deployed
        assert turn_start_result.damage_dealt
        assert entry_result.damage_dealt

    def test_full_swarm_body_lifecycle(self):
        """Test the complete Swarm Body activation and ending sequence."""
        activate_result = activate_swarm_body(
            combatant_id="balor_1",
            current_hp=8,
            has_moved_this_turn=False,
            enemy_target_ids=["enemy_1"],
        )
        turn_start_result = resolve_swarm_body_turn_start(
            combatant_id="balor_1",
            has_moved_this_turn=False,
            enemy_target_ids=["enemy_1"],
        )
        end_result = end_swarm_body_condition(combatant_id="balor_1")
        assert activate_result.zone_active
        assert turn_start_result.save_triggered
        assert end_result.condition_ended

    def test_regeneration_not_stackable_with_healing(self):
        """Test that regeneration and healing don't stack inappropriately."""
        regen_result = resolve_balor_regeneration(
            combatant_id="balor_1",
            max_hp=12,
            is_overheated=False,
            has_structure_damage=False,
            is_core_power_active=False,
            current_hp=8,
        )
        assert regen_result.healing_amount == 3
        assert regen_result.was_paused is False

    def test_scouring_swarm_scales_with_core_power(self):
        """Test that Scouring Swarm damage scales with core power state."""
        inactive = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=False,
            affected_target_ids=["enemy_1"],
        )
        active = resolve_scouring_swarm(
            combatant_id="balor_1",
            is_core_power_active=True,
            affected_target_ids=["enemy_1"],
        )
        assert active.damage_per_target == 2 * inactive.damage_per_target
