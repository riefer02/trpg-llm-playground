"""Tests for combat resolution helpers."""

from core.shared.combat_helpers import (
    AttackPattern,
    validate_attack_geometry,
    CriticalEffect,
    calculate_critical_damage,
    AttackSequenceInput,
    resolve_attack_sequence,
    MovementInput,
    resolve_movement,
    FullActionTurnInput,
    resolve_full_action_turn,
    StatusCheckInput,
    check_status_effects,
    TurretDroneAttackInput,
    resolve_turret_drone_attack,
    LatchDroneInput,
    resolve_latch_drone,
    RestockDroneInput,
    resolve_restock_drone,
    ICEOUTDroneInput,
    resolve_iceout_drone,
    TrackingDroneInput,
    resolve_tracking_drone,
    HiveDroneInput,
    resolve_hive_drone,
)


class TestAttackPattern:
    """Tests for AttackPattern model."""

    def test_single_target_pattern(self):
        pattern = AttackPattern(type="single")
        assert pattern.type == "single"
        assert pattern.size is None
        assert pattern.origin is None

    def test_line_pattern(self):
        pattern = AttackPattern(type="line", size=5, origin=(0, 0))
        assert pattern.type == "line"
        assert pattern.size == 5
        assert pattern.origin == (0, 0)

    def test_cone_pattern(self):
        pattern = AttackPattern(type="cone", size=3)
        assert pattern.type == "cone"
        assert pattern.size == 3

    def test_blast_pattern(self):
        pattern = AttackPattern(type="blast", size=2)
        assert pattern.type == "blast"
        assert pattern.size == 2

    def test_burst_pattern(self):
        pattern = AttackPattern(type="burst", size=1)
        assert pattern.type == "burst"
        assert pattern.size == 1


class TestGeometryValidation:
    """Tests for attack geometry validation."""

    def test_single_target_validation(self):
        pattern = AttackPattern(type="single")
        result = validate_attack_geometry(
            pattern=pattern,
            attacker_position=(0, 0),
            target_positions={"target1": (1, 1)},
        )
        assert result.is_valid is True
        assert "target1" in result.affected_target_ids
        assert result.reason == "Single target: no geometry needed"

    def test_line_pattern_validation(self):
        pattern = AttackPattern(type="line", size=3, origin=(0, 0))
        result = validate_attack_geometry(
            pattern=pattern,
            attacker_position=(0, 0),
            target_positions={"target1": (2, 0)},
        )
        assert result.is_valid is True
        assert len(result.affected_spaces) == 4
        assert "target1" in result.affected_target_ids

    def test_cone_pattern_validation(self):
        pattern = AttackPattern(type="cone", size=2, origin=(0, 0))
        result = validate_attack_geometry(
            pattern=pattern,
            attacker_position=(0, 0),
            target_positions={"target1": (1, 2)},
        )
        assert result.is_valid is True
        assert len(result.affected_spaces) > 0

    def test_blast_pattern_validation(self):
        pattern = AttackPattern(type="blast", size=1, origin=(0, 0))
        result = validate_attack_geometry(
            pattern=pattern,
            attacker_position=(0, 0),
            target_positions={"target1": (0, 1)},
        )
        assert result.is_valid is True
        assert (0, 1) in result.affected_spaces

    def test_burst_pattern_validation(self):
        pattern = AttackPattern(type="burst", size=1, origin=(5, 5))
        result = validate_attack_geometry(
            pattern=pattern,
            attacker_position=(5, 5),
            target_positions={"target1": (5, 5)},
        )
        assert result.is_valid is True
        assert (5, 5) in result.affected_spaces

    def test_pattern_requires_size(self):
        pattern = AttackPattern(type="line")
        result = validate_attack_geometry(
            pattern=pattern,
            attacker_position=(0, 0),
            target_positions={},
        )
        assert result.is_valid is False
        assert "requires size parameter" in result.reason


class TestCriticalDamageCalculation:
    """Tests for critical damage calculation (PR2 3965-3969)."""

    def test_non_critical_damage(self):
        result = calculate_critical_damage(
            base_damage=3,
            bonus_damage=2,
            is_critical=False,
            force_rolls=[4, 5, 6],
        )
        assert result.is_critical is False
        assert result.base_damage == 3
        assert result.bonus_damage == 2
        assert result.rolled_once == [4, 5, 6]
        assert result.total_damage == 17  # 4+5+6+2 = 17

    def test_critical_damage_rolls_twice(self):
        result = calculate_critical_damage(
            base_damage=2,
            bonus_damage=0,
            is_critical=True,
            force_rolls=[3, 4, 5, 6],
        )
        assert result.is_critical is True
        assert len(result.rolled_twice) == 4
        assert len(result.highest_selected) == 2
        assert sum(result.highest_selected) >= sum(result.rolled_twice[:2])

    def test_critical_with_bonus_damage(self):
        result = calculate_critical_damage(
            base_damage=4,
            bonus_damage=2,
            is_critical=True,
            force_rolls=[1, 2, 3, 4, 5, 6, 1, 2],
        )
        assert result.is_critical is True
        assert result.bonus_damage == 2
        assert result.total_damage == sum(result.highest_selected) + 2

    def test_critical_picks_highest(self):
        result = calculate_critical_damage(
            base_damage=3,
            bonus_damage=0,
            is_critical=True,
            force_rolls=[1, 2, 3, 4, 5, 6],
        )
        assert result.is_critical is True
        assert result.highest_selected == [6, 5, 4]
        assert result.total_damage == 15


class TestAttackSequence:
    """Tests for attack sequence resolution."""

    def test_single_target_hit(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1"],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
            force_attack_roll=15,
        )
        result = resolve_attack_sequence(input_data)
        assert result.total_targets == 1
        assert result.targets_hit == 1
        assert result.target_results[0].is_hit is True
        assert result.target_results[0].is_critical is False

    def test_single_target_miss(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1"],
            attack_bonus=5,
            defense_value=15,
            base_damage=4,
            damage_type="kinetic",
            force_attack_roll=9,
        )
        result = resolve_attack_sequence(input_data)
        assert result.total_targets == 1
        assert result.targets_hit == 0
        assert result.target_results[0].is_hit is False

    def test_critical_hit(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1"],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
            force_attack_roll=20,
            force_damage_rolls=[3, 4, 5, 6, 1, 2],
        )
        result = resolve_attack_sequence(input_data)
        assert result.targets_hit == 1
        assert result.targets_critical == 1
        assert result.target_results[0].is_critical is True
        assert result.target_results[0].damage_result is not None
        assert result.target_results[0].damage_result.is_critical is True

    def test_multi_target_attack(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1", "target2"],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
            bonus_damage=4,
            force_attack_roll=15,
        )
        result = resolve_attack_sequence(input_data)
        assert result.total_targets == 2
        assert result.targets_hit == 2
        assert result.total_damage_dealt > 0

    def test_multi_target_bonus_damage_halved(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1", "target2"],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
            bonus_damage=4,
            force_attack_roll=15,
        )
        result = resolve_attack_sequence(input_data)
        for target_result in result.target_results:
            if target_result.damage_result:
                assert target_result.damage_result.bonus_damage == 2

    def test_critical_effects(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1"],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
            critical_effects=[
                CriticalEffect(type="knockback", value=2),
                CriticalEffect(type="shredded"),
            ],
            force_attack_roll=20,
        )
        result = resolve_attack_sequence(input_data)
        assert result.targets_critical == 1
        assert "shredded" in result.conditions_on_targets.get("target1", [])
        assert "target1" in result.positions_changed

    def test_save_on_miss(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1"],
            attack_bonus=5,
            defense_value=20,
            base_damage=4,
            damage_type="kinetic",
            save_target=10,
            save_type="agility",
            save_on_miss=True,
            force_attack_roll=10,
        )
        result = resolve_attack_sequence(input_data)
        assert result.targets_hit == 0
        assert result.target_results[0].save_result is not None

    def test_drone_assisted_attack(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1"],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
            drone_assisted=True,
            drone_bonus=2,
            force_attack_roll=12,
        )
        result = resolve_attack_sequence(input_data)
        assert result.targets_hit == 1


class TestMovement:
    """Tests for movement resolution."""

    def test_basic_movement(self):
        input_data = MovementInput(
            mover_id="mech1",
            start_position=(0, 0),
            end_position=(3, 0),
            speed=4,
        )
        result = resolve_movement(input_data)
        assert result.mover_id == "mech1"
        assert result.spaces_moved == 3
        assert result.path_valid is True
        assert len(result.path_hexes) == 4

    def test_movement_exceeds_speed(self):
        input_data = MovementInput(
            mover_id="mech1",
            start_position=(0, 0),
            end_position=(10, 0),
            speed=4,
        )
        result = resolve_movement(input_data)
        assert result.spaces_moved == 4

    def test_movement_with_enemy_threat(self):
        input_data = MovementInput(
            mover_id="mech1",
            start_position=(0, 0),
            end_position=(5, 0),
            speed=6,
            nearby_enemies=[
                {
                    "id": "enemy1",
                    "position": (3, 0),
                    "threat": 1,
                    "weapons": [{"id": "weapon1", "overwatch_available": True}],
                }
            ],
        )
        result = resolve_movement(input_data)
        assert len(result.overwatch_triggers) > 0
        assert result.overwatch_triggers[0]["enemy_id"] == "enemy1"

    def test_disengage_ignores_reactions(self):
        input_data = MovementInput(
            mover_id="mech1",
            start_position=(0, 0),
            end_position=(3, 0),
            speed=4,
            is_disengaging=True,
            nearby_enemies=[
                {
                    "id": "enemy1",
                    "position": (2, 0),
                    "threat": 1,
                    "weapons": [{"id": "weapon1", "overwatch_available": True}],
                }
            ],
        )
        result = resolve_movement(input_data)
        assert result.disengage_used is True
        assert len(result.reactions_avoided) > 0
        assert len(result.overwatch_triggers) == 0

    def test_terrain_costs(self):
        input_data = MovementInput(
            mover_id="mech1",
            start_position=(0, 0),
            end_position=(4, 0),
            speed=6,
            terrain_map={
                "1,0": "difficult",
                "2,0": "difficult",
            },
        )
        result = resolve_movement(input_data)
        assert result.difficult_terrain_penalty == 2


class TestFullActionTurn:
    """Tests for full action turn resolution."""

    def test_full_action_attack(self):
        attack_input = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1"],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
            force_attack_roll=15,
        )
        input_data = FullActionTurnInput(
            actor_id="mech1",
            position=(0, 0),
            action_choice="full",
            full_action_type="attack",
            attack_input=attack_input,
        )
        result = resolve_full_action_turn(input_data)
        assert result.action_type == "full"
        assert "attack" in result.actions_taken
        assert result.attack_result is not None
        assert result.attack_result.targets_hit == 1

    def test_full_action_move(self):
        movement_input = MovementInput(
            mover_id="mech1",
            start_position=(0, 0),
            end_position=(3, 0),
            speed=4,
        )
        input_data = FullActionTurnInput(
            actor_id="mech1",
            position=(0, 0),
            action_choice="full",
            full_action_type="move",
            movement_input=movement_input,
        )
        result = resolve_full_action_turn(input_data)
        assert result.action_type == "full"
        assert "move" in result.actions_taken
        assert result.position_changed is True
        assert result.movement_result is not None

    def test_two_quick_actions(self):
        input_data = FullActionTurnInput(
            actor_id="mech1",
            position=(0, 0),
            action_choice="two_quick",
            quick_actions=[{"type": "move"}, {"type": "move"}],
            movement_input=MovementInput(
                mover_id="mech1",
                start_position=(0, 0),
                end_position=(2, 0),
                speed=4,
            ),
        )
        result = resolve_full_action_turn(input_data)
        assert result.action_type == "two_quick"
        assert len(result.actions_taken) == 2

    def test_overwatch_triggers_from_attack(self):
        attack_input = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1"],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
            force_attack_roll=15,
            nearby_enemies=[
                {
                    "id": "enemy1",
                    "position": (2, 0),
                    "threat": 1,
                    "weapons": [{"id": "weapon1", "overwatch_available": True}],
                }
            ],
        )
        input_data = FullActionTurnInput(
            actor_id="mech1",
            position=(0, 0),
            action_choice="full",
            full_action_type="attack",
            attack_input=attack_input,
            enemies=[{"id": "enemy1"}],
        )
        result = resolve_full_action_turn(input_data)
        assert result.overwatch_triggers_from_this_actor is not None


class TestStatusInteractions:
    """Tests for status/condition interaction checks."""

    def test_jammed_cannot_take_reactions(self):
        input_data = StatusCheckInput(
            statuses=["jammed"],
            attempted_action="reaction",
        )
        result = check_status_effects(input_data)
        assert result.can_perform is False
        assert result.cannot_take_reactions is True
        assert "JAMMED" in result.reasons[0]

    def test_jammed_only_improvised_attacks(self):
        input_data = StatusCheckInput(
            statuses=["jammed"],
            attempted_action="attack",
        )
        result = check_status_effects(input_data)
        assert result.only_improvised_or_grapple is True

    def test_stunned_cannot_take_actions(self):
        input_data = StatusCheckInput(
            statuses=["stunned"],
            attempted_action="attack",
        )
        result = check_status_effects(input_data)
        assert result.cannot_take_action is True

    def test_immobilized_cannot_move(self):
        input_data = StatusCheckInput(
            statuses=["immobilized"],
            attempted_action="move",
        )
        result = check_status_effects(input_data)
        assert result.cannot_move is True

    def test_immobilized_cannot_stand_up(self):
        input_data = StatusCheckInput(
            statuses=["immobilized"],
            attempted_action="stand_up",
        )
        result = check_status_effects(input_data)
        assert result.cannot_take_action is True

    def test_slowed_movement_cap(self):
        input_data = StatusCheckInput(
            statuses=["slowed"],
            attempted_action="move",
        )
        result = check_status_effects(input_data)
        assert result.movement_speed_cap == 0
        assert result.only_regular_move is True

    def test_engaged_difficulty_modifier(self):
        input_data = StatusCheckInput(
            statuses=["engaged"],
            attempted_action="attack",
        )
        result = check_status_effects(input_data)
        assert result.difficulty_modifier == 1

    def test_hidden_cannot_be_targeted(self):
        input_data = StatusCheckInput(
            statuses=["hidden"],
            attempted_action="attack",
        )
        result = check_status_effects(input_data)
        assert result.cannot_be_targeted is True

    def test_braced_cannot_take_reactions(self):
        input_data = StatusCheckInput(
            statuses=["braced"],
            attempted_action="reaction",
        )
        result = check_status_effects(input_data)
        assert result.cannot_take_reactions is True

    def test_impaired_difficulty_modifier(self):
        input_data = StatusCheckInput(
            statuses=["impaired"],
            attempted_action="attack",
        )
        result = check_status_effects(input_data)
        assert result.difficulty_modifier == 1

    def test_multiple_statuses(self):
        input_data = StatusCheckInput(
            statuses=["engaged", "impaired"],
            attempted_action="attack",
        )
        result = check_status_effects(input_data)
        assert result.difficulty_modifier == 2

    def test_down_cannot_act(self):
        input_data = StatusCheckInput(
            statuses=[
                "stunned"
            ],  # "down" is not a StatusType, using stunned which also prevents actions
            attempted_action="attack",
        )
        result = check_status_effects(input_data)
        assert result.cannot_take_action is True


class TestTurretDrone:
    """Tests for turret drone resolution."""

    def test_turret_drone_attack_hit(self):
        input_data = TurretDroneAttackInput(
            drone_id="turret1",
            owner_id="mech1",
            drone_position=(0, 0),
            ally_attack_hit=True,
            ally_id="mech1",
            ally_position=(5, 0),
            target_id="enemy1",
            target_position=(6, 0),
            drone_tier=1,
        )
        result = resolve_turret_drone_attack(input_data)
        assert result.total_targets == 1

    def test_turret_drone_ally_miss(self):
        input_data = TurretDroneAttackInput(
            drone_id="turret1",
            owner_id="mech1",
            drone_position=(0, 0),
            ally_attack_hit=False,
            ally_id="mech1",
            ally_position=(5, 0),
            target_id="enemy1",
            target_position=(6, 0),
        )
        result = resolve_turret_drone_attack(input_data)
        assert result.total_targets == 0

    def test_turret_drone_out_of_range(self):
        input_data = TurretDroneAttackInput(
            drone_id="turret1",
            owner_id="mech1",
            drone_position=(0, 0),
            ally_attack_hit=True,
            ally_id="mech1",
            ally_position=(15, 0),
            target_id="enemy1",
            target_position=(16, 0),
        )
        result = resolve_turret_drone_attack(input_data)
        assert result.total_targets == 0


class TestLatchDrone:
    """Tests for latch drone resolution."""

    def test_latch_drone_mount_stunned(self):
        input_data = LatchDroneInput(
            drone_id="latch1",
            owner_id="mech1",
            target_mech_id="enemy1",
            mode="mount",
            mount_damage=2,
            target_is_stunned=True,
        )
        result = resolve_latch_drone(input_data)
        assert result.success is True
        assert result.mount_hit is True
        assert "stunned" in result.conditions_cleared

    def test_latch_drone_mount_not_stunned(self):
        input_data = LatchDroneInput(
            drone_id="latch1",
            owner_id="mech1",
            target_mech_id="enemy1",
            mode="mount",
            mount_damage=2,
            target_is_stunned=False,
        )
        result = resolve_latch_drone(input_data)
        assert result.mount_hit is False

    def test_latch_drone_active_buff(self):
        input_data = LatchDroneInput(
            drone_id="latch1",
            owner_id="mech1",
            target_mech_id="enemy1",
            mode="active",
            buff_type="evasion",
            buff_value=2,
        )
        result = resolve_latch_drone(input_data)
        assert result.success is True
        assert result.buff_applied is True
        assert result.buff_details["type"] == "evasion"


class TestRestockDrone:
    """Tests for restock drone resolution."""

    def test_restock_drone_cool(self):
        input_data = RestockDroneInput(
            drone_id="restock1",
            owner_id="mech1",
            activating_combatant_id="mech1",
            activating_combatant_position=(0, 0),
            action_choice="cool",
        )
        result = resolve_restock_drone(input_data)
        assert result.action == "cool"
        assert result.heat_cleared == 6

    def test_restock_drone_reload(self):
        input_data = RestockDroneInput(
            drone_id="restock1",
            owner_id="mech1",
            activating_combatant_id="mech1",
            activating_combatant_position=(0, 0),
            action_choice="reload",
        )
        result = resolve_restock_drone(input_data)
        assert result.action == "reload"
        assert len(result.weapons_reloaded) > 0

    def test_restock_drone_clear_condition(self):
        input_data = RestockDroneInput(
            drone_id="restock1",
            owner_id="mech1",
            activating_combatant_id="mech1",
            activating_combatant_position=(0, 0),
            action_choice="clear_condition",
            condition_to_clear="impaired",
        )
        result = resolve_restock_drone(input_data)
        assert result.action == "clear_condition"
        assert result.condition_cleared == "impaired"


class TestICEOUTDrone:
    """Tests for ICEOUT drone resolution."""

    def test_iceout_drone_activation(self):
        input_data = ICEOUTDroneInput(
            drone_id="iceout1",
            owner_id="mech1",
            drone_position=(5, 5),
            zone_target_ids=["target1", "target2"],
        )
        result = resolve_iceout_drone(input_data)
        assert result.success is True
        assert result.zone_created is True
        assert result.zone_center == (5, 5)
        assert result.zone_radius == 1
        assert result.tech_immunity_granted is True


class TestTrackingDrone:
    """Tests for tracking drone resolution."""

    def test_tracking_drone_hit(self):
        input_data = TrackingDroneInput(
            drone_id="tracking1",
            owner_id="mech1",
            drone_position=(0, 0),
            target_id="enemy1",
            target_position=(5, 0),
            tech_attack_bonus=3,
            target_e_defense=10,
        )
        result = resolve_tracking_drone(input_data)
        assert result.hit is True
        assert result.hide_negated is True
        assert result.invis_negated is True

    def test_tracking_drone_miss(self):
        input_data = TrackingDroneInput(
            drone_id="tracking1",
            owner_id="mech1",
            drone_position=(0, 0),
            target_id="enemy1",
            target_position=(5, 0),
            tech_attack_bonus=-2,
            target_e_defense=15,
        )
        result = resolve_tracking_drone(input_data)
        assert result.hit is False


class TestHiveDrone:
    """Tests for hive drone resolution."""

    def test_hive_drone_activation(self):
        input_data = HiveDroneInput(
            drone_id="hive1",
            owner_id="mech1",
            drone_position=(5, 5),
            zone_target_ids=["target1"],
        )
        result = resolve_hive_drone(input_data)
        assert result.success is True
        assert result.zone_created is True
        assert result.zone_center == (5, 5)
        assert result.zone_radius == 2
        assert result.soft_cover_granted is True


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_targets(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=[],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
        )
        result = resolve_attack_sequence(input_data)
        assert result.total_targets == 0
        assert result.targets_hit == 0
        assert result.total_damage_dealt == 0

    def test_same_position_movement(self):
        input_data = MovementInput(
            mover_id="mech1",
            start_position=(5, 5),
            end_position=(5, 5),
            speed=4,
        )
        result = resolve_movement(input_data)
        assert result.spaces_moved == 0
        assert len(result.path_hexes) == 1

    def test_multiple_critical_effects(self):
        input_data = AttackSequenceInput(
            attacker_id="mech1",
            position=(0, 0),
            target_ids=["target1"],
            attack_bonus=5,
            defense_value=10,
            base_damage=4,
            damage_type="kinetic",
            critical_effects=[
                CriticalEffect(type="knockback", value=2),
                CriticalEffect(type="prone_save"),
                CriticalEffect(type="shredded"),
                CriticalEffect(type="impaired"),
            ],
            force_attack_roll=20,
        )
        result = resolve_attack_sequence(input_data)
        assert result.targets_critical == 1
        conditions = result.conditions_on_targets.get("target1", [])
        assert "shredded" in conditions
        assert "impaired" in conditions

    def test_status_check_with_no_statuses(self):
        input_data = StatusCheckInput(
            statuses=[],
            attempted_action="attack",
        )
        result = check_status_effects(input_data)
        assert result.can_perform is True
        assert result.difficulty_modifier == 0
        assert len(result.reasons) == 0
