"""Tests for tech action resolution helpers."""

import pytest
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
    """Tests for the Invade action resolution."""

    def test_resolve_invade_hit(self):
        """Invade hits when systems >= E-defense."""
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
            target_e_defense=8,
        )

        assert result.action_id == "invade"
        assert result.actor_id == "pilot_1"
        assert result.target_id == "mech_a"
        assert result.hit is True
        assert result.heat_applied == 2
        assert "impaired" in result.conditions_applied
        assert "slowed" in result.conditions_applied
        assert result.systems_roll is not None
        assert result.check_total is not None
        assert result.target_e_defense == 8

    def test_resolve_invade_miss(self):
        """Invade misses when systems < E-defense."""
        from core.mech.combat_resolution import ResolutionSettings

        settings = ResolutionSettings(forced_rolls=[1])
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=6,
            target_e_defense=12,
            settings=settings,
        )

        assert result.hit is False
        assert result.heat_applied is None
        assert result.conditions_applied == []

    def test_resolve_invade_custom_heat(self):
        """Invade can deal custom heat amount."""
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
            target_e_defense=8,
            heat_on_hit=4,
        )

        assert result.hit is True
        assert result.heat_applied == 4

    def test_resolve_invade_custom_conditions(self):
        """Invade can inflict custom conditions."""
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
            target_e_defense=8,
            conditions=["stunned"],
        )

        assert result.hit is True
        assert result.conditions_applied == ["stunned"]

    def test_resolve_invade_boundary_hit(self):
        """Invade hits on exact match (systems == E-defense)."""
        from core.mech.combat_resolution import ResolutionSettings

        settings = ResolutionSettings(forced_rolls=[1])
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=9,
            target_e_defense=10,
            settings=settings,
        )

        assert result.hit is True
        assert result.heat_applied == 2

    def test_resolve_invade_forced_rolls(self):
        """Invade respects forced roll settings."""
        from core.mech.combat_resolution import ResolutionSettings

        settings = ResolutionSettings(forced_rolls=[6])
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
            target_e_defense=20,
            settings=settings,
        )

        assert result.systems_roll is not None

    def test_resolve_invade_duration_end_of_next_turn(self):
        """Invade conditions last until end of next turn."""
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
            target_e_defense=8,
        )

        assert result.duration == "end_of_next_turn"

    def test_resolve_invade_high_systems_always_hits(self):
        """High enough systems score will always hit."""
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=20,
            target_e_defense=8,
        )

        assert result.hit is True

    def test_resolve_invade_low_systems_miss(self):
        """Low systems score will miss high E-defense."""
        from core.mech.combat_resolution import ResolutionSettings

        settings = ResolutionSettings(forced_rolls=[1])
        result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=4,
            target_e_defense=15,
            settings=settings,
        )

        assert result.hit is False


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
        lock_on_result = resolve_lock_on(
            actor_id="pilot_1",
            target_id="mech_a",
        )

        invade_result = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
            target_e_defense=8,
        )

        assert lock_on_result.success is True
        assert invade_result.hit is True

    def test_multiple_invades_tracking(self):
        """Multiple invade attempts can be tracked separately."""
        from core.mech.combat_resolution import ResolutionSettings

        invade_1 = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_a",
            attacker_systems=10,
            target_e_defense=8,
        )

        settings = ResolutionSettings(forced_rolls=[1])
        invade_2 = resolve_invade(
            actor_id="pilot_1",
            target_id="mech_b",
            attacker_systems=10,
            target_e_defense=12,
            settings=settings,
        )

        assert invade_1.hit is True
        assert invade_2.hit is False

    def test_tech_action_result_types(self):
        """All results have correct types."""
        scan = resolve_scan(actor_id="p", target_id="t", scan_options=["stats"])
        bolster = resolve_bolster(actor_id="p", target_id="t", attacker_systems=10)
        lock_on = resolve_lock_on(actor_id="p", target_id="t")
        invade = resolve_invade(
            actor_id="p", target_id="t", attacker_systems=10, target_e_defense=8
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
