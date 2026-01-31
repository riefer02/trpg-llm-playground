"""Tests for Full Tech action resolution."""

import pytest
from core.shared.full_tech import (
    resolve_full_tech,
    apply_full_tech_result,
    FullTechInput,
    FullTechFirstOption,
    FullTechSecondOption,
    ScanTechParams,
    BolsterTechParams,
    LockOnTechParams,
    InvadeTechParams,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
)
from core.mech.tech_actions import InvadeResult, LockOnResult


@pytest.fixture
def test_combatant() -> CombatantState:
    """Create a test combatant for Full Tech tests."""
    return CombatantState(
        id="test_mech",
        name="Test Mech",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=10,
            evasion=10,
            e_defense=10,
            armor=0,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=10,
        ),
        statuses=[],
        conditions=[],
    )


@pytest.fixture
def target_combatant() -> CombatantState:
    """Create a test target for tech actions."""
    return CombatantState(
        id="target_mech",
        name="Target Mech",
        side="hostiles",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=10,
            evasion=10,
            e_defense=8,
            armor=0,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=10,
        ),
        statuses=[],
        conditions=[],
    )


class TestResolveFullTech:
    """Tests for Full Tech resolution (pure logic)."""

    def test_full_tech_scan_and_invade(self):
        """Test Full Tech with Scan + Invade options."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="scan",
                scan_params=ScanTechParams(target_id="target_mech"),
            ),
            second_option=FullTechSecondOption(
                option="invade",
                invade_params=InvadeTechParams(
                    target_id="target_mech",
                    tech_attack_bonus=10,
                    target_e_defense=8,
                ),
            ),
        )
        result = resolve_full_tech(input)

        assert result.is_valid
        assert result.first_option == "scan"
        assert result.second_option == "invade"
        assert result.first_result is not None
        assert result.second_result is not None
        assert result.first_result.target_id == "target_mech"
        assert result.second_result.target_id == "target_mech"

    def test_full_tech_same_option_twice(self):
        """Test Full Tech with Lock On + Lock On (same option twice)."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(target_id="target_mech"),
            ),
            second_option=FullTechSecondOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(target_id="ally_mech"),
            ),
        )
        result = resolve_full_tech(input)

        assert result.is_valid
        assert result.first_option == "lock_on"
        assert result.second_option == "lock_on"
        assert result.first_result.target_id == "target_mech"
        assert result.second_result.target_id == "ally_mech"

    def test_full_tech_bolster_and_lock_on(self):
        """Test Full Tech with Bolster + Lock On options."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="bolster",
                bolster_params=BolsterTechParams(
                    target_id="ally_mech",
                    attacker_systems=10,
                ),
            ),
            second_option=FullTechSecondOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(target_id="target_mech"),
            ),
        )
        result = resolve_full_tech(input)

        assert result.is_valid
        assert result.first_option == "bolster"
        assert result.second_option == "lock_on"
        assert result.first_result is not None
        assert result.second_result is not None

    def test_full_tech_invalid_missing_scan_params(self):
        """Test Full Tech with invalid scan option (missing params)."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(option="scan"),
            second_option=FullTechSecondOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(target_id="target_mech"),
            ),
        )
        result = resolve_full_tech(input)

        assert not result.is_valid
        assert len(result.validation_errors) == 1
        assert "scan_params required" in result.validation_errors[0]

    def test_full_tech_invalid_missing_invade_params(self):
        """Test Full Tech with invalid invade option (missing params)."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="scan",
                scan_params=ScanTechParams(target_id="target_mech"),
            ),
            second_option=FullTechSecondOption(option="invade"),
        )
        result = resolve_full_tech(input)

        assert not result.is_valid
        assert len(result.validation_errors) == 1
        assert "invade_params required" in result.validation_errors[0]

    def test_full_tech_all_four_options(self):
        """Test Full Tech with all different options."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="scan",
                scan_params=ScanTechParams(target_id="target_mech"),
            ),
            second_option=FullTechSecondOption(
                option="invade",
                invade_params=InvadeTechParams(
                    target_id="target_mech",
                    tech_attack_bonus=10,
                    target_e_defense=8,
                ),
            ),
        )
        result = resolve_full_tech(input)

        assert result.is_valid
        assert result.first_option == "scan"
        assert result.second_option == "invade"

    def test_full_tech_invade_hit_and_conditions(self):
        """Test Full Tech invade that hits and applies conditions."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="invade",
                invade_params=InvadeTechParams(
                    target_id="target_mech",
                    tech_attack_bonus=12,
                    target_e_defense=8,
                ),
            ),
            second_option=FullTechSecondOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(target_id="target_mech"),
            ),
        )
        result = resolve_full_tech(input)

        assert result.is_valid
        assert isinstance(result.second_result, LockOnResult)
        assert result.second_result.target_id == "target_mech"

    def test_full_tech_invade_miss(self):
        """Test Full Tech invade that misses."""
        # Force a low roll (5) so the attack misses against e_defense=100
        # Total will be 5 + 5 = 10, which is less than 100
        from core.mech.combat_resolution import ResolutionSettings

        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="invade",
                invade_params=InvadeTechParams(
                    target_id="target_mech",
                    tech_attack_bonus=5,
                    target_e_defense=100,
                ),
            ),
            second_option=FullTechSecondOption(
                option="scan",
                scan_params=ScanTechParams(target_id="target_mech"),
            ),
            settings=ResolutionSettings(forced_roll=5),
        )
        result = resolve_full_tech(input)

        assert result.is_valid
        assert isinstance(result.first_result, InvadeResult)
        assert result.first_result.hit is False


class TestApplyFullTechResult:
    """Tests for applying Full Tech results to combatant state."""

    def test_apply_full_tech_valid(self, test_combatant: CombatantState):
        """Test applying valid Full Tech result."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="scan",
                scan_params=ScanTechParams(target_id="target_mech"),
            ),
            second_option=FullTechSecondOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(target_id="target_mech"),
            ),
        )
        result = resolve_full_tech(input)
        applied = apply_full_tech_result(test_combatant, result)

        assert applied.updated_combatant.id == "test_mech"
        assert applied.first_tech_applied is True
        assert applied.second_tech_applied is True
        assert "target_mech" in applied.targets_affected

    def test_apply_full_tech_invalid(self, test_combatant: CombatantState):
        """Test applying invalid Full Tech result."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(option="scan"),
            second_option=FullTechSecondOption(option="invade"),
        )
        result = resolve_full_tech(input)
        applied = apply_full_tech_result(test_combatant, result)

        assert applied.first_tech_applied is False
        assert applied.second_tech_applied is False
        assert applied.updated_combatant.id == "test_mech"

    def test_apply_full_tech_invade_heat(self, test_combatant: CombatantState):
        """Test Full Tech invade applies heat correctly."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="invade",
                invade_params=InvadeTechParams(
                    target_id="target_mech",
                    tech_attack_bonus=12,
                    target_e_defense=8,
                    heat_on_hit=2,
                ),
            ),
            second_option=FullTechSecondOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(target_id="target_mech"),
            ),
        )
        result = resolve_full_tech(input)
        applied = apply_full_tech_result(test_combatant, result)

        assert applied.heat_dealt == 2
        assert "impaired" in applied.conditions_applied
        assert "slowed" in applied.conditions_applied

    def test_apply_full_tech_conditions_from_invade(
        self, test_combatant: CombatantState
    ):
        """Test Full Tech invade applies conditions correctly."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="invade",
                invade_params=InvadeTechParams(
                    target_id="target_mech",
                    tech_attack_bonus=12,
                    target_e_defense=8,
                ),
            ),
            second_option=FullTechSecondOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(target_id="target_mech"),
            ),
        )
        result = resolve_full_tech(input)
        applied = apply_full_tech_result(test_combatant, result)

        assert "impaired" in applied.conditions_applied
        assert "slowed" in applied.conditions_applied

    def test_apply_full_tech_miss_no_conditions(self, test_combatant: CombatantState):
        """Test Full Tech invade miss doesn't apply conditions."""
        from core.mech.combat_resolution import ResolutionSettings

        # Force a low roll to ensure miss (nat 20 always hits regardless of defense)
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="invade",
                invade_params=InvadeTechParams(
                    target_id="target_mech",
                    tech_attack_bonus=5,
                    target_e_defense=100,
                ),
            ),
            second_option=FullTechSecondOption(
                option="scan",
                scan_params=ScanTechParams(target_id="target_mech"),
            ),
            settings=ResolutionSettings(forced_roll=5),
        )
        result = resolve_full_tech(input)
        applied = apply_full_tech_result(test_combatant, result)

        assert applied.heat_dealt == 0


class TestFullTechInputValidation:
    """Tests for Full Tech input validation."""

    def test_scan_with_all_options(self):
        """Test Scan action with all info options."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="scan",
                scan_params=ScanTechParams(
                    target_id="target_mech",
                    scan_options=["stats", "hidden_info", "public_info"],
                ),
            ),
            second_option=FullTechSecondOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(target_id="target_mech"),
            ),
        )
        result = resolve_full_tech(input)

        assert result.is_valid
        assert result.first_result is not None

    def test_bolster_with_custom_accuracy(self):
        """Test Bolster action with custom accuracy bonus."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="bolster",
                bolster_params=BolsterTechParams(
                    target_id="ally_mech",
                    attacker_systems=10,
                    accuracy_bonus=3,
                ),
            ),
            second_option=FullTechSecondOption(
                option="scan",
                scan_params=ScanTechParams(target_id="target_mech"),
            ),
        )
        result = resolve_full_tech(input)

        assert result.is_valid
        assert result.first_result is not None

    def test_lock_on_with_custom_bonus(self):
        """Test Lock On action with custom accuracy bonus."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(
                    target_id="target_mech",
                    accuracy_bonus=2,
                ),
            ),
            second_option=FullTechSecondOption(
                option="lock_on",
                lock_on_params=LockOnTechParams(
                    target_id="ally_mech",
                    accuracy_bonus=2,
                ),
            ),
        )
        result = resolve_full_tech(input)

        assert result.is_valid

    def test_invade_with_custom_heat(self):
        """Test Invade action with custom heat on hit."""
        input = FullTechInput(
            actor_id="test_mech",
            first_option=FullTechFirstOption(
                option="invade",
                invade_params=InvadeTechParams(
                    target_id="target_mech",
                    tech_attack_bonus=10,
                    target_e_defense=8,
                    heat_on_hit=4,
                ),
            ),
            second_option=FullTechSecondOption(
                option="scan",
                scan_params=ScanTechParams(target_id="target_mech"),
            ),
        )
        result = resolve_full_tech(input)
        applied = apply_full_tech_result(
            CombatantState(
                id="test_mech",
                name="Test Mech",
                side="players",
                kind="mech",
                stats=CombatStats(
                    size="size_2",
                    hp_max=10,
                    evasion=10,
                    e_defense=10,
                    armor=0,
                    speed=4,
                    tech_attack=10,
                ),
                resources=CombatResources(
                    hp_current=10,
                    heat_current=0,
                    heat_cap=10,
                ),
                statuses=[],
                conditions=[],
            ),
            result,
        )

        assert applied.heat_dealt == 4
