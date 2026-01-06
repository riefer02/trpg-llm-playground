"""Tests for Mount, Dismount, and Eject action resolution."""

import pytest
from core.shared.mount import (
    resolve_mount,
    apply_mount_result,
    resolve_dismount,
    apply_dismount_result,
    resolve_eject,
    apply_eject_result,
    MountInput,
    DismountInput,
    EjectInput,
    MountRule,
    DismountRule,
    EjectRule,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
)
from core.mech.grid import HexPosition, HexCoord


@pytest.fixture
def test_pilot() -> CombatantState:
    """Create a test pilot for mount/dismount/eject tests."""
    return CombatantState(
        id="test_pilot",
        name="Test Pilot",
        side="players",
        kind="pilot",
        stats=CombatStats(
            size="size_half",
            hp_max=6,
            evasion=10,
            e_defense=10,
            armor=0,
            speed=4,
        ),
        resources=CombatResources(hp_current=6),
        conditions=["impaired"],
    )


@pytest.fixture
def test_mech() -> CombatantState:
    """Create a test mech for mount/dismount/eject tests."""
    return CombatantState(
        id="test_mech",
        name="Test Mech",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=16,
            evasion=10,
            e_defense=8,
            armor=1,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=16,
            heat_current=0,
            heat_cap=10,
            structure_current=3,
            stress_current=2,
        ),
    )


@pytest.fixture
def enemy_mech() -> CombatantState:
    """Create an enemy mech for jockey tests."""
    return CombatantState(
        id="enemy_mech",
        name="Enemy Mech",
        side="hostiles",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=16,
            evasion=10,
            e_defense=8,
            armor=1,
            speed=4,
            sensor_range=10,
        ),
        resources=CombatResources(
            hp_current=16,
            heat_current=0,
            heat_cap=10,
            structure_current=3,
            stress_current=2,
        ),
    )


@pytest.fixture
def mech_position() -> HexPosition:
    """Create a test mech position."""
    return HexPosition(coord=HexCoord(q=0, r=0))


@pytest.fixture
def adjacent_position() -> HexPosition:
    """Create an adjacent position."""
    return HexPosition(coord=HexCoord(q=1, r=0))


class TestResolveMount:
    """Tests for Mount action resolution."""

    def test_resolve_mount_success(self, test_pilot, test_mech):
        """Test successful mount action."""
        input_data = MountInput(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
        )
        result = resolve_mount(input_data)

        assert result.mount_success is True
        assert result.actor_id == test_pilot.id
        assert result.mech_id == test_mech.id
        assert len(result.validation_errors) == 0

    def test_resolve_mount_self_mount_error(self, test_mech):
        """Test that a mech cannot mount itself."""
        input_data = MountInput(
            actor_id=test_mech.id,
            mech_id=test_mech.id,
        )
        result = resolve_mount(input_data)

        assert result.mount_success is False
        assert len(result.validation_errors) > 0

    def test_resolve_mount_with_custom_rules(self, test_pilot, test_mech):
        """Test mount with custom rules."""
        rules = MountRule(
            requires_adjacent=False,
            allows_allied_mount=False,
        )
        input_data = MountInput(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
            rules=rules,
        )
        result = resolve_mount(input_data)

        assert result.mount_success is True
        assert result.requires_adjacent is False


class TestApplyMountResult:
    """Tests for applying Mount result to combatant state."""

    def test_apply_mount_result_success(self, test_mech):
        """Test applying successful mount result."""
        from core.shared.mount import MountResolutionResult

        result = MountResolutionResult(
            actor_id="test_pilot",
            mech_id="test_mech",
            mount_success=True,
            requires_adjacent=True,
        )

        application = apply_mount_result(test_mech, result)

        assert application.pilot_now_piloting is True
        assert application.updated_mech.id == test_mech.id

    def test_apply_mount_result_failure(self, test_mech):
        """Test applying failed mount result."""
        from core.shared.mount import MountResolutionResult

        result = MountResolutionResult(
            actor_id="test_pilot",
            mech_id="test_mech",
            mount_success=False,
            requires_adjacent=True,
            validation_errors=["Cannot mount self"],
        )

        application = apply_mount_result(test_mech, result)

        assert application.pilot_now_piloting is False


class TestResolveDismount:
    """Tests for Dismount action resolution."""

    def test_resolve_dismount_success_with_space(
        self, test_pilot, test_mech, adjacent_position
    ):
        """Test successful dismount with free adjacent space."""
        input_data = DismountInput(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
        )
        result = resolve_dismount(
            input_data,
            mech_position=None,
            free_adjacent_spaces=[adjacent_position],
        )

        assert result.dismount_success is True
        assert result.pilot_position == adjacent_position
        assert len(result.validation_errors) == 0

    def test_resolve_dismount_failure_no_space(self, test_pilot, test_mech):
        """Test dismount failure when no free adjacent space."""
        input_data = DismountInput(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
        )
        result = resolve_dismount(
            input_data,
            mech_position=None,
            free_adjacent_spaces=[],
        )

        assert result.dismount_success is False
        assert result.pilot_position is None
        assert len(result.validation_errors) > 0

    def test_resolve_dismount_with_custom_rules(self, test_pilot, test_mech):
        """Test dismount with custom rules."""
        rules = DismountRule(
            requires_adjacent_space=False,
            allows_allied_dismount=True,
        )
        input_data = DismountInput(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
            rules=rules,
        )
        result = resolve_dismount(
            input_data,
            mech_position=None,
            free_adjacent_spaces=[],
            rules=rules,
        )

        assert result.dismount_success is True
        assert result.adjacent_space_required is False


class TestApplyDismountResult:
    """Tests for applying Dismount result to combatant state."""

    def test_apply_dismount_result_success(
        self, test_mech, test_pilot, adjacent_position
    ):
        """Test applying successful dismount result."""
        from core.shared.mount import DismountResolutionResult

        result = DismountResolutionResult(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
            dismount_success=True,
            pilot_position=adjacent_position,
            adjacent_space_required=True,
        )

        application = apply_dismount_result(test_mech, test_pilot, result)

        assert application.pilot_no_longer_piloting is True
        assert application.pilot_position == adjacent_position

    def test_apply_dismount_result_failure(self, test_mech, test_pilot):
        """Test applying failed dismount result."""
        from core.shared.mount import DismountResolutionResult

        result = DismountResolutionResult(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
            dismount_success=False,
            pilot_position=None,
            adjacent_space_required=True,
            validation_errors=["No free space"],
        )

        application = apply_dismount_result(test_mech, test_pilot, result)

        assert application.pilot_no_longer_piloting is False


class TestResolveEject:
    """Tests for Eject action resolution."""

    def test_resolve_eject_success(self, test_pilot, test_mech, mech_position):
        """Test successful eject action."""
        input_data = EjectInput(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
            eject_direction=HexCoord(q=1, r=0),
        )
        result = resolve_eject(
            input_data,
            mech_position=mech_position,
            eject_used=False,
        )

        assert result.eject_success is True
        assert result.eject_distance == 6
        assert result.target_position is not None
        assert result.target_position.coord.q == 6
        assert result.target_position.coord.r == 0
        assert result.impaired_applied is True
        assert result.eject_already_used is False
        assert len(result.validation_errors) == 0

    def test_resolve_eject_already_used(self, test_pilot, test_mech):
        """Test eject failure when already used."""
        rules = EjectRule(can_reuse_after_full_repair=False)
        input_data = EjectInput(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
            eject_direction=HexCoord(q=1, r=0),
            rules=rules,
        )
        result = resolve_eject(
            input_data,
            mech_position=None,
            eject_used=True,
            rules=rules,
        )

        assert result.eject_success is False
        assert len(result.validation_errors) > 0

    def test_resolve_eject_no_direction(self, test_pilot, test_mech):
        """Test eject without direction (adjacent space)."""
        input_data = EjectInput(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
            eject_direction=None,
        )
        result = resolve_eject(
            input_data,
            mech_position=None,
            eject_used=False,
        )

        assert result.eject_success is True
        assert result.target_position is None

    def test_resolve_eject_with_custom_rules(self, test_pilot, test_mech):
        """Test eject with custom rules."""
        rules = EjectRule(
            eject_distance=10,
            causes_impaired_until_full_repair=False,
        )
        input_data = EjectInput(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
            eject_direction=HexCoord(q=1, r=0),
            rules=rules,
        )
        result = resolve_eject(
            input_data,
            mech_position=None,
            eject_used=False,
            rules=rules,
        )

        assert result.eject_success is True
        assert result.eject_distance == 10
        assert result.impaired_applied is False


class TestApplyEjectResult:
    """Tests for applying Eject result to combatant state."""

    def test_apply_eject_result_success(self, test_mech):
        """Test applying successful eject result."""
        from core.shared.mount import EjectResolutionResult

        pilot_without_impaired = CombatantState(
            id="test_pilot",
            name="Test Pilot",
            side="players",
            kind="pilot",
            stats=CombatStats(
                size="size_half",
                hp_max=6,
                evasion=10,
                e_defense=10,
                armor=0,
                speed=4,
            ),
            resources=CombatResources(hp_current=6),
        )

        result = EjectResolutionResult(
            actor_id=pilot_without_impaired.id,
            mech_id=test_mech.id,
            eject_success=True,
            eject_distance=6,
            eject_direction=HexCoord(q=1, r=0),
            target_position=HexPosition(coord=HexCoord(q=6, r=0)),
            impaired_applied=True,
            eject_already_used=False,
        )

        application = apply_eject_result(test_mech, pilot_without_impaired, result)

        assert application.pilot_ejected is True
        assert application.impaired_applied is True
        assert application.eject_used_flag_set is True
        assert "impaired" in application.updated_pilot.conditions

    def test_apply_eject_result_no_impaired(self, test_mech):
        """Test eject result without impaired condition."""
        from core.shared.mount import EjectResolutionResult

        pilot = CombatantState(
            id="test_pilot",
            name="Test Pilot",
            side="players",
            kind="pilot",
            stats=CombatStats(
                size="size_half",
                hp_max=6,
                evasion=10,
                e_defense=10,
                armor=0,
                speed=4,
            ),
            resources=CombatResources(hp_current=6),
        )

        result = EjectResolutionResult(
            actor_id=pilot.id,
            mech_id=test_mech.id,
            eject_success=True,
            eject_distance=6,
            eject_direction=HexCoord(q=1, r=0),
            target_position=HexPosition(coord=HexCoord(q=6, r=0)),
            impaired_applied=False,
            eject_already_used=False,
        )

        application = apply_eject_result(test_mech, pilot, result)

        assert application.pilot_ejected is True
        assert application.impaired_applied is False

    def test_apply_eject_result_failure(self, test_mech, test_pilot):
        """Test applying failed eject result."""
        from core.shared.mount import EjectResolutionResult

        result = EjectResolutionResult(
            actor_id=test_pilot.id,
            mech_id=test_mech.id,
            eject_success=False,
            eject_distance=6,
            eject_direction=HexCoord(q=1, r=0),
            target_position=None,
            impaired_applied=False,
            eject_already_used=True,
            validation_errors=["Eject already used"],
        )

        application = apply_eject_result(test_mech, test_pilot, result)

        assert application.pilot_ejected is False
        assert application.impaired_applied is False
        assert application.eject_used_flag_set is False


class TestMountRuleDefaults:
    """Tests for default mount/dismount/eject rules."""

    def test_default_mount_rules(self):
        """Test default MountRule values."""
        from core.shared.mount import DEFAULT_MOUNT_RULES

        assert DEFAULT_MOUNT_RULES.requires_adjacent is True
        assert DEFAULT_MOUNT_RULES.allows_allied_mount is True

    def test_default_dismount_rules(self):
        """Test default DismountRule values."""
        from core.shared.mount import DEFAULT_DISMOUNT_RULES

        assert DEFAULT_DISMOUNT_RULES.places_pilot_adjacent is True
        assert DEFAULT_DISMOUNT_RULES.requires_adjacent_space is True

    def test_default_eject_rules(self):
        """Test default EjectRule values."""
        from core.shared.mount import DEFAULT_EJECT_RULES

        assert DEFAULT_EJECT_RULES.eject_distance == 6
        assert DEFAULT_EJECT_RULES.causes_impaired_until_full_repair is True
        assert DEFAULT_EJECT_RULES.can_reuse_after_full_repair is True
