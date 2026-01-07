"""Tests for pilot Fight action resolution."""

import pytest
from core.shared.fight import (
    resolve_fight,
    FightInput,
    FightResolutionResult,
    ActionTypeLiteral,
)


class TestResolveFight:
    """Tests for the resolve_fight function."""

    def test_basic_melee_hit(self):
        """Test basic melee attack that hits."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=1,
            ),
            forced_roll=12,
        )

        assert result.hit is True
        assert result.attack_result.roll == 12
        assert result.attack_result.attack_bonus == 3
        assert result.attack_result.total_accuracy == 15
        assert result.action_type == "full"
        assert result.damage_flat == 1
        assert result.damage_on_hit == 1

    def test_basic_melee_miss(self):
        """Test basic melee attack that misses."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=2,
                target_evasion=15,
                damage_flat=1,
            ),
            forced_roll=8,
        )

        assert result.hit is False
        assert result.attack_result.roll == 8
        assert result.attack_result.total_accuracy == 10
        assert result.damage_on_hit is None

    def test_ranged_attack_engaged_penalty(self):
        """Test ranged attack gets +1 difficulty when engaged."""
        result_engaged = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_combat",
                is_ranged=True,
                is_engaged=True,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=2,
            ),
            forced_roll=10,
            forced_difficulty_rolls=[4],
        )

        assert result_engaged.attack_result.difficulty_dice_rolls == [4]
        assert result_engaged.attack_result.net_accuracy == -4
        assert result_engaged.attack_result.total_accuracy == 9

    def test_ranged_attack_cover_penalty(self):
        """Test ranged attack gets +1 difficulty when target has cover."""
        result_cover = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_combat",
                is_ranged=True,
                is_engaged=False,
                target_has_cover=True,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=2,
            ),
            forced_roll=10,
            forced_difficulty_rolls=[3],
        )

        assert result_cover.attack_result.difficulty_dice_rolls == [3]
        assert result_cover.attack_result.net_accuracy == -3
        assert result_cover.attack_result.total_accuracy == 10

    def test_ranged_attack_combined_penalties(self):
        """Test ranged attack gets both engaged and cover penalties."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_combat",
                is_ranged=True,
                is_engaged=True,
                target_has_cover=True,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=2,
            ),
            forced_roll=10,
            forced_difficulty_rolls=[4, 3],
        )

        assert len(result.attack_result.difficulty_dice_rolls) == 2
        assert result.attack_result.net_accuracy == -3
        assert result.attack_result.total_accuracy == 10

    def test_melee_attack_no_engaged_penalty(self):
        """Test melee attack does not get engaged penalty."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=True,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=1,
            ),
            forced_roll=10,
        )

        assert result.attack_result.difficulty_dice_rolls == []
        assert result.attack_result.net_accuracy == 0

    def test_sidearm_quick_action(self):
        """Test sidearm weapon uses quick action."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_sidearm",
                is_ranged=True,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=1,
            ),
            forced_roll=12,
        )

        assert result.action_type == "quick"
        assert result.weapon_id == "signature_weapon_sidearm"

    def test_non_sidearm_full_action(self):
        """Test non-sidearm weapons use full action."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_combat",
                is_ranged=True,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=2,
            ),
            forced_roll=12,
        )

        assert result.action_type == "full"

    def test_critical_hit_doubles_damage(self):
        """Test critical hit doubles damage."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_combat",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=20,
                damage_flat=2,
            ),
            forced_roll=20,
        )

        assert result.is_critical is True
        assert result.hit is True
        assert result.damage_on_hit == 4

    def test_exact_tie_hit_on_10(self):
        """Test ties hit when roll >= 10."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=5,
                target_evasion=15,
                damage_flat=1,
            ),
            forced_roll=10,
        )

        assert result.hit is True
        assert result.attack_result.total_accuracy == 15

    def test_exact_tie_miss_below_10(self):
        """Test ties miss when roll < 10."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=5,
                target_evasion=15,
                damage_flat=1,
            ),
            forced_roll=9,
        )

        assert result.hit is False
        assert result.attack_result.total_accuracy == 14

    def test_unknown_weapon_error(self):
        """Test error when weapon ID is unknown."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="unknown_weapon",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=1,
            ),
            forced_roll=1,
        )

        assert len(result.validation_errors) > 0
        assert "Unknown weapon ID" in result.validation_errors[0]
        assert result.hit is False

    def test_archaic_weapon_cannot_harm_mech(self):
        """Test archaic weapons cannot damage mechs."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="enemy_mech",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=True,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=1,
            ),
            forced_roll=1,
        )

        assert len(result.validation_errors) > 0
        assert "archaic" in result.validation_errors[0].lower()
        assert result.hit is False

    def test_archaic_weapon_can_harm_pilot(self):
        """Test archaic weapons can damage non-mech targets."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="enemy_pilot",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=1,
            ),
            forced_roll=15,
        )

        assert len(result.validation_errors) == 0
        assert result.hit is True

    def test_unarmed_attack(self):
        """Test unarmed (None weapon) attack."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id=None,
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=0,
            ),
            forced_roll=15,
        )

        assert result.weapon_id is None
        assert result.hit is True
        assert result.damage_flat == 0

    def test_accuracy_bonus_applied(self):
        """Test accuracy bonus from systems/talents is applied."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_combat",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=2,
                target_evasion=15,
                damage_flat=2,
                accuracy_bonus=2,
            ),
            forced_roll=8,
            forced_accuracy_rolls=[6],
        )

        assert result.attack_result.accuracy_dice_rolls == [6]
        assert result.attack_result.net_accuracy == 6
        assert result.attack_result.total_accuracy == 16

    def test_edge_case_roll_1(self):
        """Test roll of 1 (minimum)."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=5,
                target_evasion=7,
                damage_flat=1,
            ),
            forced_roll=1,
        )

        assert result.hit is False
        assert result.attack_result.roll == 1
        assert result.attack_result.total_accuracy == 6

    def test_edge_case_roll_10(self):
        """Test roll of 10 (tie threshold)."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=2,
                target_evasion=12,
                damage_flat=1,
            ),
            forced_roll=10,
        )

        assert result.hit is True
        assert result.attack_result.roll == 10

    def test_edge_case_roll_9(self):
        """Test roll of 9 (just below tie threshold)."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=12,
                damage_flat=1,
            ),
            forced_roll=9,
        )

        assert result.hit is False
        assert result.attack_result.roll == 9

    def test_high_defense_target(self):
        """Test attack against very high defense."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_combat",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=5,
                target_evasion=20,
                damage_flat=2,
                accuracy_bonus=2,
            ),
            forced_roll=15,
            forced_accuracy_rolls=[6],
        )

        assert result.hit is True
        assert result.attack_result.total_accuracy == 26

    def test_zero_defense(self):
        """Test attack against zero defense."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=1,
                target_evasion=0,
                damage_flat=1,
            ),
            forced_roll=1,
        )

        assert result.hit is True
        assert result.damage_on_hit == 1

    def test_alloy_composite_light_weapon(self):
        """Test Alloy/Composite Light weapon properties."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="alloy_composite_light",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
            ),
            forced_roll=12,
        )

        assert result.action_type == "full"
        assert result.damage_flat == 1
        assert result.damage_type == "kinetic"

    def test_alloy_composite_heavy_inaccurate(self):
        """Test Alloy/Composite Heavy weapon has inaccurate tag."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="alloy_composite_heavy",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
            ),
            forced_roll=8,
            forced_difficulty_rolls=[4],
        )

        assert len(result.attack_result.difficulty_dice_rolls) == 1


class TestFightInput:
    """Tests for FightInput model."""

    def test_fight_input_defaults(self):
        """Test default values for FightInput."""
        input = FightInput(
            actor_id="pilot_1",
            target_id="target_1",
        )

        assert input.weapon_id is None
        assert input.is_ranged is False
        assert input.is_engaged is False
        assert input.target_has_cover is False
        assert input.target_is_mech is False
        assert input.grit_bonus == 0
        assert input.target_evasion == 10
        assert input.target_e_defense == 10

    def test_fight_input_ranged_with_e_defense(self):
        """Test ranged attack uses e-defense when specified."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_combat",
                is_ranged=True,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                target_e_defense=12,
                damage_flat=2,
            ),
            forced_roll=15,
        )

        assert result.attack_result.target_defense == 12


class TestFightResolutionResult:
    """Tests for FightResolutionResult model."""

    def test_result_contains_attack_result(self):
        """Test that result contains attack resolution."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="archaic_melee",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=1,
            ),
            forced_roll=12,
        )

        assert isinstance(result.attack_result.roll, int)
        assert isinstance(result.attack_result.hit, bool)

    def test_result_damage_fields(self):
        """Test damage-related fields in result."""
        result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_combat",
                is_ranged=False,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
                damage_flat=2,
            ),
            forced_roll=15,
        )

        assert result.damage_flat == 2
        assert result.damage_on_hit == 2
        assert result.damage_type == "kinetic"

    def test_result_action_type_field(self):
        """Test action type field reflects weapon tag."""
        sidearm_result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_sidearm",
                is_ranged=True,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
            ),
            forced_roll=12,
        )

        assert sidearm_result.action_type == "quick"

        heavy_result = resolve_fight(
            input=FightInput(
                actor_id="pilot_1",
                target_id="target_1",
                weapon_id="signature_weapon_heavy",
                is_ranged=True,
                is_engaged=False,
                target_has_cover=False,
                target_is_mech=False,
                grit_bonus=3,
                target_evasion=10,
            ),
            forced_roll=12,
        )

        assert heavy_result.action_type == "full"
