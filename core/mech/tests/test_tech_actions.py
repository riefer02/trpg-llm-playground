"""Tests for tech action resolution helpers."""

from core.mech.tech_actions import (
    resolve_scan,
    resolve_bolster,
    resolve_lock_on,
    resolve_invade,
    ScanResult,
    BolsterResult,
    LockOnResult,
    InvadeResult,
)


class TestResolveScan:
    """Tests for the Scan action resolution."""

    def test_resolve_scan_basic(self):
        """Scan reveals requested information categories."""
        result = resolve_scan(
            actor_id="pilot_1",
            target_id="mech_a",
            scan_options=["stats", "hidden_info", "public_info"],
        )

        assert result.action_id == "scan"
        assert result.actor_id == "pilot_1"
        assert result.target_id == "mech_a"
        assert result.success is True
        assert "stats" in result.revealed_info
        assert "hidden_info" in result.revealed_info
        assert "public_info" in result.revealed_info

    def test_resolve_scan_partial_info(self):
        """Scan can reveal only some information categories."""
        result = resolve_scan(
            actor_id="pilot_1",
            target_id="mech_a",
            scan_options=["stats"],
        )

        assert result.success is True
        assert result.revealed_info == ["stats"]
        assert "hidden_info" not in result.revealed_info

    def test_resolve_scan_no_options(self):
        """Scan with no options reveals nothing."""
        result = resolve_scan(
            actor_id="pilot_1",
            target_id="mech_a",
            scan_options=[],
        )

        assert result.success is True
        assert result.revealed_info == []


class TestResolveBolster:
    """Tests for the Bolster action resolution."""

    def test_resolve_bolster_default(self):
        """Bolster grants +2 accuracy by default."""
        result = resolve_bolster(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
        )

        assert result.action_id == "bolster"
        assert result.actor_id == "pilot_1"
        assert result.target_id == "mech_a"
        assert result.success is True
        assert result.accuracy_bonus == 2
        assert result.duration == "end_of_next_turn"
        assert result.systems_roll is not None
        assert result.check_total is not None

    def test_resolve_bolster_custom_bonus(self):
        """Bolster can grant custom accuracy bonus."""
        result = resolve_bolster(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=8,
            accuracy_bonus=3,
        )

        assert result.accuracy_bonus == 3

    def test_resolve_bolster_high_systems(self):
        """Higher systems score gives better total."""
        result_high = resolve_bolster(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=12,
        )

        result_low = resolve_bolster(
            actor_id="pilot_2",
            target_id="mech_b",
            attacker_systems=6,
        )

        assert result_high.check_total > result_low.check_total

    def test_resolve_bolster_forced_rolls(self):
        """Bolster respects forced roll settings."""
        from core.mech.combat_resolution import ResolutionSettings

        settings = ResolutionSettings(forced_rolls=[6, 6])
        result = resolve_bolster(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
            settings=settings,
        )

        assert result.systems_roll is not None
        assert len(result.systems_roll.rolls) >= 1

    def test_resolve_bolster_duration_end_of_next_turn(self):
        """Bolster duration is end of next turn."""
        result = resolve_bolster(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
        )

        assert result.duration == "end_of_next_turn"


class TestResolveLockOn:
    """Tests for the Lock On action resolution."""

    def test_resolve_lock_on_default(self):
        """Lock On grants +1 accuracy by default."""
        result = resolve_lock_on(
            actor_id="pilot_1",
            target_id="mech_a",
        )

        assert result.action_id == "lock_on"
        assert result.actor_id == "pilot_1"
        assert result.target_id == "mech_a"
        assert result.success is True
        assert result.accuracy_bonus == 1
        assert result.duration == "until_consumed"
        assert result.status_granted == "lock_on"

    def test_resolve_lock_on_custom_bonus(self):
        """Lock On can grant custom accuracy bonus."""
        result = resolve_lock_on(
            actor_id="pilot_1",
            target_id="mech_a",
            accuracy_bonus=2,
        )

        assert result.accuracy_bonus == 2

    def test_resolve_lock_on_always_succeeds(self):
        """Lock On always succeeds - it's not contested."""
        result = resolve_lock_on(
            actor_id="pilot_1",
            target_id="mech_a",
        )

        assert result.success is True

    def test_resolve_lock_on_duration_until_consumed(self):
        """Lock On duration is until consumed by hostile attack."""
        result = resolve_lock_on(
            actor_id="pilot_1",
            target_id="mech_a",
        )

        assert result.duration == "until_consumed"
        assert result.status_granted == "lock_on"


class TestResolveInvade:
    """Tests for the Invade action resolution.

    Invade uses standard Lancer attack resolution: 1d20 + tech_attack vs E-defense.
    Hit when: total > E-defense OR (total == E-defense AND roll >= 10).
    Critical on natural 20 (always hits).
    """

    def test_resolve_invade_hit(self):
        """Invade hits when 1d20 + tech_attack > E-defense."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 15 + tech_attack 5 = 20 vs E-defense 12 -> hit
        settings = ResolutionSettings(forced_roll=15)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=5,
            target_e_defense=12,
            settings=settings,
        )

        assert result.action_id == "invade"
        assert result.actor_id == "pilot_1"
        assert result.target_id == "mech_a"
        assert result.hit is True
        assert result.attack_roll == 15
        assert result.attack_bonus == 5
        assert result.total == 20
        assert result.target_e_defense == 12
        assert result.is_critical is False
        assert result.heat_applied == 2
        assert "impaired" in result.conditions_applied
        assert "slowed" in result.conditions_applied

    def test_resolve_invade_miss(self):
        """Invade misses when 1d20 + tech_attack < E-defense."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 5 + tech_attack 2 = 7 vs E-defense 12 -> miss
        settings = ResolutionSettings(forced_roll=5)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=2,
            target_e_defense=12,
            settings=settings,
        )

        assert result.hit is False
        assert result.attack_roll == 5
        assert result.total == 7
        assert result.heat_applied is None
        assert result.conditions_applied == []

    def test_resolve_invade_critical(self):
        """Natural 20 always hits (critical)."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 20 + tech_attack 0 = 20 vs E-defense 25 -> crit, always hits
        settings = ResolutionSettings(forced_roll=20)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=0,
            target_e_defense=25,
            settings=settings,
        )

        assert result.hit is True
        assert result.is_critical is True
        assert result.attack_roll == 20
        assert result.heat_applied == 2

    def test_resolve_invade_boundary_hit_high_roll(self):
        """Exact match hits when d20 roll >= 10."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 10 + tech_attack 2 = 12 vs E-defense 12, roll >= 10 -> hit
        settings = ResolutionSettings(forced_roll=10)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=2,
            target_e_defense=12,
            settings=settings,
        )

        assert result.hit is True
        assert result.total == 12
        assert result.target_e_defense == 12
        assert result.heat_applied == 2

    def test_resolve_invade_boundary_miss_low_roll(self):
        """Exact match misses when d20 roll < 10."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 5 + tech_attack 7 = 12 vs E-defense 12, roll < 10 -> miss
        settings = ResolutionSettings(forced_roll=5)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=7,
            target_e_defense=12,
            settings=settings,
        )

        assert result.hit is False
        assert result.total == 12
        assert result.target_e_defense == 12

    def test_resolve_invade_custom_heat(self):
        """Invade can deal custom heat amount."""
        from core.mech.combat_resolution import ResolutionSettings

        settings = ResolutionSettings(forced_roll=15)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=5,
            target_e_defense=10,
            heat_on_hit=4,
            settings=settings,
        )

        assert result.hit is True
        assert result.heat_applied == 4

    def test_resolve_invade_custom_conditions(self):
        """Invade can inflict custom conditions."""
        from core.mech.combat_resolution import ResolutionSettings

        settings = ResolutionSettings(forced_roll=15)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=5,
            target_e_defense=10,
            conditions=["stunned"],
            settings=settings,
        )

        assert result.hit is True
        assert result.conditions_applied == ["stunned"]

    def test_resolve_invade_with_accuracy(self):
        """Accuracy dice add to the attack total."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 8 + tech_attack 2 + accuracy (6) = 16 vs E-defense 15 -> hit
        settings = ResolutionSettings(
            forced_roll=8,
            forced_accuracy_rolls=[6],
        )
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=2,
            target_e_defense=15,
            accuracy_bonus=1,
            settings=settings,
        )

        assert result.hit is True
        assert result.attack_roll == 8
        assert result.total == 16
        assert result.accuracy_dice_rolls == [6]

    def test_resolve_invade_with_difficulty(self):
        """Difficulty dice subtract from the attack total."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 15 + tech_attack 5 - difficulty (3) = 17 vs E-defense 18 -> miss
        settings = ResolutionSettings(
            forced_roll=15,
            forced_difficulty_rolls=[3],
        )
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=5,
            target_e_defense=18,
            difficulty_bonus=1,
            settings=settings,
        )

        assert result.hit is False
        assert result.attack_roll == 15
        assert result.total == 17
        assert result.difficulty_dice_rolls == [3]

    def test_resolve_invade_duration_end_of_next_turn(self):
        """Invade conditions last until end of next turn."""
        from core.mech.combat_resolution import ResolutionSettings

        settings = ResolutionSettings(forced_roll=15)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=5,
            target_e_defense=10,
            settings=settings,
        )

        assert result.duration == "end_of_next_turn"

    def test_resolve_invade_high_bonus_hits(self):
        """High tech attack bonus makes hits likely."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 1 + tech_attack 12 = 13 vs E-defense 10 -> hit
        settings = ResolutionSettings(forced_roll=1)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=12,
            target_e_defense=10,
            settings=settings,
        )

        assert result.hit is True
        assert result.total == 13

    def test_resolve_invade_low_bonus_miss(self):
        """Low tech attack bonus against high E-defense misses."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 5 + tech_attack 0 = 5 vs E-defense 15 -> miss
        settings = ResolutionSettings(forced_roll=5)
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=0,
            target_e_defense=15,
            settings=settings,
        )

        assert result.hit is False
        assert result.total == 5


class TestTechActionIntegration:
    """Integration tests for tech action resolution."""

    def test_scan_then_bolster(self):
        """Actor can scan a target then bolster a different target."""
        scan_result = resolve_scan(
            actor_id="pilot_1",
            target_id="mech_a",
            scan_options=["stats"],
        )

        bolster_result = resolve_bolster(
            actor_id="pilot_1",
            target_id="mech_b",
            attacker_systems=10,
        )

        assert scan_result.action_id == "scan"
        assert bolster_result.action_id == "bolster"

    def test_lock_on_then_invade_same_target(self):
        """Actor can lock on then invade the same target."""
        from core.mech.combat_resolution import ResolutionSettings

        lock_on_result = resolve_lock_on(
            actor_id="pilot_1",
            target_id="mech_a",
        )

        # Roll 15 + tech_attack 5 = 20 vs E-defense 8 -> hit
        settings = ResolutionSettings(forced_roll=15)
        invade_result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=5,
            target_e_defense=8,
            settings=settings,
        )

        assert lock_on_result.success is True
        assert invade_result.hit is True

    def test_multiple_invades_tracking(self):
        """Multiple invade attempts can be tracked separately."""
        from core.mech.combat_resolution import ResolutionSettings

        # Roll 15 + tech_attack 5 = 20 vs E-defense 8 -> hit
        settings_hit = ResolutionSettings(forced_roll=15)
        invade_1 = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            tech_attack_bonus=5,
            target_e_defense=8,
            settings=settings_hit,
        )

        # Roll 1 + tech_attack 5 = 6 vs E-defense 12 -> miss
        settings_miss = ResolutionSettings(forced_roll=1)
        invade_2 = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_b",
            tech_attack_bonus=5,
            target_e_defense=12,
            settings=settings_miss,
        )

        assert invade_1.hit is True
        assert invade_2.hit is False

    def test_tech_action_result_types(self):
        """All results have correct types."""
        from core.mech.combat_resolution import ResolutionSettings

        scan = resolve_scan(actor_id="p", target_id="t", scan_options=["stats"])
        bolster = resolve_bolster(actor_id="p", target_id="t", attacker_systems=10)
        lock_on = resolve_lock_on(actor_id="p", target_id="t")
        settings = ResolutionSettings(forced_roll=15)
        invade = resolve_invade(
            actor_id="p", target_id="t", tech_attack_bonus=5, target_e_defense=8,
            settings=settings,
        )

        assert isinstance(scan, ScanResult)
        assert isinstance(bolster, BolsterResult)
        assert isinstance(lock_on, LockOnResult)
        assert isinstance(invade, InvadeResult)

    def test_result_equality(self):
        """Results with same data should be equal."""
        scan_1 = resolve_scan(
            actor_id="pilot_1",
            target_id="mech_a",
            scan_options=["stats"],
        )

        scan_2 = resolve_scan(
            actor_id="pilot_1",
            target_id="mech_a",
            scan_options=["stats"],
        )

        assert scan_1 == scan_2

    def test_bolster_with_different_systems_gives_different_totals(self):
        """Different systems scores give different check totals."""
        from core.mech.combat_resolution import ResolutionSettings

        settings = ResolutionSettings(forced_rolls=[5, 5])

        low_systems = resolve_bolster(
            actor_id="p", target_id="t", attacker_systems=4, settings=settings
        )
        high_systems = resolve_bolster(
            actor_id="p", target_id="t", attacker_systems=12, settings=settings
        )

        assert low_systems.check_total != high_systems.check_total
