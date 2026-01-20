"""Tests for pending decisions module."""

import pytest
from core.shared.decisions import (
    PendingDecision,
    DecisionResolution,
    SaveDecisionResult,
    TraumaDecisionResult,
    generate_decision_id,
    create_hull_save_decision,
    create_engineering_save_decision,
    create_engineering_check_decision,
    create_system_trauma_decision,
    check_structure_decisions,
    check_overheat_decisions,
    check_dangerous_terrain_decision,
    resolve_save_decision,
    resolve_trauma_decision,
    get_pending_decisions_for_combatant,
    remove_decision_from_scenario,
    add_decision_to_scenario,
)
from core.shared.structure import StructureResolutionResult, SystemTraumaSelection
from core.shared.heat import OverheatResolutionResult
from core.shared.saves import SaveRequest
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    MechInventory,
    WeaponMountState,
    MechSystemState,
    WeaponState,
)
from core.mech.grid import HexPosition, HexCoord


class TestPendingDecision:
    """Tests for PendingDecision model."""

    def test_hull_save_decision(self):
        """Test creating a hull save decision."""
        decision = PendingDecision(
            decision_id="test_1",
            decision_type="hull_save",
            combatant_id="mech_1",
            trigger_source="structure_cascade",
            trigger_round=3,
            save_type="hull",
            save_target=10,
            save_bonus=2,
        )
        assert decision.decision_type == "hull_save"
        assert decision.save_type == "hull"
        assert decision.save_target == 10
        assert decision.save_bonus == 2
        assert decision.trigger_round == 3

    def test_engineering_save_decision(self):
        """Test creating an engineering save decision."""
        decision = PendingDecision(
            decision_id="test_2",
            decision_type="engineering_save",
            combatant_id="mech_1",
            trigger_source="meltdown",
            trigger_round=4,
            save_type="engineering",
            save_target=10,
            save_bonus=3,
        )
        assert decision.decision_type == "engineering_save"
        assert decision.save_type == "engineering"

    def test_system_trauma_decision(self):
        """Test creating a system trauma decision."""
        decision = PendingDecision(
            decision_id="test_3",
            decision_type="system_trauma",
            combatant_id="mech_1",
            trigger_source="system_trauma",
            trigger_round=2,
            eligible_mounts=[0, 1],
            eligible_systems=["sys_shield", "sys_reactor"],
        )
        assert decision.decision_type == "system_trauma"
        assert decision.eligible_mounts == [0, 1]
        assert decision.eligible_systems == ["sys_shield", "sys_reactor"]

    def test_decision_with_reroll(self):
        """Test decision with reroll available."""
        decision = PendingDecision(
            decision_id="test_4",
            decision_type="hull_save",
            combatant_id="mech_1",
            trigger_source="structure_cascade",
            trigger_round=1,
            save_type="hull",
            save_target=10,
            save_bonus=0,
            reroll_available=True,
            reroll_source="Exemplar talent",
        )
        assert decision.reroll_available is True
        assert decision.reroll_source == "Exemplar talent"


class TestDecisionResolution:
    """Tests for DecisionResolution model."""

    def test_roll_choice(self):
        """Test resolution with roll choice."""
        resolution = DecisionResolution(choice="roll")
        assert resolution.choice == "roll"
        assert resolution.selected_mount_index is None
        assert resolution.selected_system_id is None

    def test_voluntary_fail_choice(self):
        """Test resolution with voluntary fail choice."""
        resolution = DecisionResolution(choice="voluntary_fail")
        assert resolution.choice == "voluntary_fail"

    def test_reroll_choice(self):
        """Test resolution with reroll choice."""
        resolution = DecisionResolution(choice="use_reroll", used_reroll=True)
        assert resolution.choice == "use_reroll"
        assert resolution.used_reroll is True

    def test_trauma_mount_selection(self):
        """Test resolution with mount selection for trauma."""
        resolution = DecisionResolution(
            choice="roll",  # Not used for trauma
            selected_mount_index=1,
        )
        assert resolution.selected_mount_index == 1

    def test_trauma_system_selection(self):
        """Test resolution with system selection for trauma."""
        resolution = DecisionResolution(
            choice="roll",
            selected_system_id="sys_shield",
        )
        assert resolution.selected_system_id == "sys_shield"


class TestGenerateDecisionId:
    """Tests for generate_decision_id function."""

    def test_generates_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = [generate_decision_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_id_format(self):
        """Test that ID has expected format."""
        decision_id = generate_decision_id()
        assert decision_id.startswith("decision_")
        assert len(decision_id) == len("decision_") + 12


class TestCreateDecisionFunctions:
    """Tests for decision creation functions."""

    def test_create_hull_save_decision(self):
        """Test creating a hull save decision."""
        decision = create_hull_save_decision(
            combatant_id="mech_1",
            trigger_round=3,
            save_target=10,
            save_bonus=2,
        )
        assert decision.decision_type == "hull_save"
        assert decision.trigger_source == "structure_cascade"
        assert decision.save_type == "hull"
        assert decision.save_target == 10
        assert decision.save_bonus == 2

    def test_create_engineering_save_decision(self):
        """Test creating an engineering save decision."""
        decision = create_engineering_save_decision(
            combatant_id="mech_1",
            trigger_round=2,
            save_target=10,
            save_bonus=3,
            trigger_source="meltdown",
        )
        assert decision.decision_type == "engineering_save"
        assert decision.trigger_source == "meltdown"
        assert decision.save_type == "engineering"

    def test_create_engineering_check_decision(self):
        """Test creating an engineering check decision."""
        decision = create_engineering_check_decision(
            combatant_id="mech_1",
            trigger_round=4,
            save_target=12,
            save_bonus=1,
            trigger_source="dangerous_terrain:lava",
        )
        assert decision.decision_type == "engineering_check"
        assert decision.trigger_source == "dangerous_terrain:lava"

    def test_create_system_trauma_decision(self):
        """Test creating a system trauma decision."""
        decision = create_system_trauma_decision(
            combatant_id="mech_1",
            trigger_round=1,
            eligible_mounts=[0, 2],
            eligible_systems=["sys_a", "sys_b"],
        )
        assert decision.decision_type == "system_trauma"
        assert decision.eligible_mounts == [0, 2]
        assert decision.eligible_systems == ["sys_a", "sys_b"]


class TestResolveSaveDecision:
    """Tests for resolve_save_decision function."""

    def test_voluntary_fail(self):
        """Test voluntary failure of save."""
        decision = create_hull_save_decision(
            combatant_id="mech_1",
            trigger_round=1,
            save_target=10,
            save_bonus=2,
        )
        resolution = DecisionResolution(choice="voluntary_fail")

        result = resolve_save_decision(decision, resolution)

        assert result.voluntarily_failed is True
        assert result.success is False
        assert result.save_result is None

    def test_successful_roll(self):
        """Test successful save roll."""
        decision = create_hull_save_decision(
            combatant_id="mech_1",
            trigger_round=1,
            save_target=10,
            save_bonus=2,
        )
        resolution = DecisionResolution(choice="roll")

        result = resolve_save_decision(decision, resolution, force_roll=15)

        assert result.voluntarily_failed is False
        assert result.success is True
        assert result.save_result is not None
        save_result = result.save_result
        assert save_result.roll == 15

    def test_failed_roll(self):
        """Test failed save roll."""
        decision = create_hull_save_decision(
            combatant_id="mech_1",
            trigger_round=1,
            save_target=15,
            save_bonus=0,
        )
        resolution = DecisionResolution(choice="roll")

        result = resolve_save_decision(decision, resolution, force_roll=5)

        assert result.success is False
        assert result.save_result is not None
        assert result.save_result.roll == 5

    def test_engineering_save_decision(self):
        """Test engineering save decision resolution."""
        decision = create_engineering_save_decision(
            combatant_id="mech_1",
            trigger_round=2,
            save_target=10,
            save_bonus=3,
        )
        resolution = DecisionResolution(choice="roll")

        result = resolve_save_decision(decision, resolution, force_roll=12)

        assert result.success is True
        assert result.save_result is not None
        assert result.save_result.save_type == "engineering"

    def test_invalid_decision_type(self):
        """Test that invalid decision type raises error."""
        decision = create_system_trauma_decision(
            combatant_id="mech_1",
            trigger_round=1,
            eligible_mounts=[0],
            eligible_systems=[],
        )
        resolution = DecisionResolution(choice="roll")

        with pytest.raises(ValueError, match="Cannot resolve save"):
            resolve_save_decision(decision, resolution)


class TestResolveTraumaDecision:
    """Tests for resolve_trauma_decision function."""

    def test_valid_mount_selection(self):
        """Test valid mount selection."""
        decision = create_system_trauma_decision(
            combatant_id="mech_1",
            trigger_round=1,
            eligible_mounts=[0, 1, 2],
            eligible_systems=["sys_a"],
        )
        resolution = DecisionResolution(choice="roll", selected_mount_index=1)

        result = resolve_trauma_decision(decision, resolution)

        assert result.valid_selection is True
        assert result.selected_target == "mount"
        assert result.mount_index == 1
        assert result.error_message is None

    def test_valid_system_selection(self):
        """Test valid system selection."""
        decision = create_system_trauma_decision(
            combatant_id="mech_1",
            trigger_round=1,
            eligible_mounts=[0],
            eligible_systems=["sys_a", "sys_b", "sys_c"],
        )
        resolution = DecisionResolution(choice="roll", selected_system_id="sys_b")

        result = resolve_trauma_decision(decision, resolution)

        assert result.valid_selection is True
        assert result.selected_target == "system"
        assert result.system_id == "sys_b"

    def test_invalid_mount_selection(self):
        """Test invalid mount selection (not in eligible list)."""
        decision = create_system_trauma_decision(
            combatant_id="mech_1",
            trigger_round=1,
            eligible_mounts=[0, 1],
            eligible_systems=[],
        )
        resolution = DecisionResolution(choice="roll", selected_mount_index=5)

        result = resolve_trauma_decision(decision, resolution)

        assert result.valid_selection is False
        assert result.error_message is not None
        assert "5" in result.error_message

    def test_invalid_system_selection(self):
        """Test invalid system selection (not in eligible list)."""
        decision = create_system_trauma_decision(
            combatant_id="mech_1",
            trigger_round=1,
            eligible_mounts=[],
            eligible_systems=["sys_a", "sys_b"],
        )
        resolution = DecisionResolution(choice="roll", selected_system_id="sys_x")

        result = resolve_trauma_decision(decision, resolution)

        assert result.valid_selection is False
        assert result.error_message is not None
        assert "sys_x" in result.error_message

    def test_no_selection_made(self):
        """Test when no selection is made."""
        decision = create_system_trauma_decision(
            combatant_id="mech_1",
            trigger_round=1,
            eligible_mounts=[0],
            eligible_systems=["sys_a"],
        )
        resolution = DecisionResolution(choice="roll")

        result = resolve_trauma_decision(decision, resolution)

        assert result.valid_selection is False
        assert result.error_message is not None
        assert "No mount or system selected" in result.error_message

    def test_invalid_decision_type(self):
        """Test that invalid decision type raises error."""
        decision = create_hull_save_decision(
            combatant_id="mech_1",
            trigger_round=1,
            save_target=10,
            save_bonus=0,
        )
        resolution = DecisionResolution(choice="roll", selected_mount_index=0)

        with pytest.raises(ValueError, match="Cannot resolve trauma"):
            resolve_trauma_decision(decision, resolution)


def _make_combatant(
    combatant_id: str = "mech_1",
    grit: int = 2,
    engineering_skill: int = 3,
) -> CombatantState:
    """Create a test combatant."""
    return CombatantState(
        id=combatant_id,
        name="Test Mech",
        kind="mech",
        side="players",
        position=HexPosition(coord=HexCoord(q=0, r=0)),
        stats=CombatStats(
            hp_max=10,
            evasion=8,
            e_defense=8,
            speed=4,
            armor=0,
            size="size_1",
            sensor_range=10,
            grit=grit,
            engineering_skill=engineering_skill,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_cap=6,
            structure_current=2,
            stress_current=3,
            heat_current=0,
            repairs_remaining=4,
        ),
    )


def _make_scenario(combatants: list[CombatantState] | None = None) -> MechCombatScenario:
    """Create a test scenario."""
    return MechCombatScenario(
        combatants=combatants or [_make_combatant()],
        rounds=[],
        grapples=[],
        deployables={},
    )


class TestCheckStructureDecisions:
    """Tests for check_structure_decisions function."""

    def test_detects_hull_save_requirement(self):
        """Test detection of hull save from direct hit at 2 structure."""
        combatant = _make_combatant()
        structure_result = StructureResolutionResult(
            outcome="direct_hit",
            dice_rolls=[1],
            lowest_roll=1,
            mech_destroyed=False,
            hull_check_request=SaveRequest(
                save_type="hull",
                save_target=10,
                save_bonus=0,
            ),
        )

        decisions = check_structure_decisions(combatant, structure_result, current_round=2)

        assert len(decisions) == 1
        assert decisions[0].decision_type == "hull_save"
        assert decisions[0].save_target == 10

    def test_detects_system_trauma_requirement(self):
        """Test detection of system trauma selection."""
        combatant = _make_combatant()
        structure_result = StructureResolutionResult(
            outcome="system_trauma",
            dice_rolls=[3],
            lowest_roll=3,
            mech_destroyed=False,
            system_trauma=SystemTraumaSelection(
                trauma_roll=2,
                initial_target="mount",
                resolved_target="mount",
                mount_index=0,
                eligible_mounts=[0, 1],
                eligible_systems=["sys_a"],
            ),
        )

        decisions = check_structure_decisions(combatant, structure_result, current_round=3)

        assert len(decisions) == 1
        assert decisions[0].decision_type == "system_trauma"
        assert decisions[0].eligible_mounts == [0, 1]
        assert decisions[0].eligible_systems == ["sys_a"]

    def test_no_decisions_for_glancing_blow(self):
        """Test that glancing blow doesn't create decisions."""
        combatant = _make_combatant()
        structure_result = StructureResolutionResult(
            outcome="glancing_blow",
            dice_rolls=[6],
            lowest_roll=6,
            mech_destroyed=False,
            statuses_to_apply=["impaired"],
        )

        decisions = check_structure_decisions(combatant, structure_result, current_round=1)

        assert len(decisions) == 0


class TestCheckOverheatDecisions:
    """Tests for check_overheat_decisions function."""

    def test_detects_engineering_save_requirement(self):
        """Test detection of engineering save from meltdown at 2 stress."""
        combatant = _make_combatant()
        overheat_result = OverheatResolutionResult(
            outcome="meltdown",
            dice_rolls=[1],
            lowest_roll=1,
            engineering_check_request=SaveRequest(
                save_type="engineering",
                save_target=10,
                save_bonus=0,
            ),
        )

        decisions = check_overheat_decisions(combatant, overheat_result, current_round=4)

        assert len(decisions) == 1
        assert decisions[0].decision_type == "engineering_save"
        assert decisions[0].save_type == "engineering"
        assert decisions[0].save_target == 10

    def test_no_decisions_for_emergency_shunt(self):
        """Test that emergency shunt doesn't create decisions."""
        combatant = _make_combatant()
        overheat_result = OverheatResolutionResult(
            outcome="emergency_shunt",
            dice_rolls=[6],
            lowest_roll=6,
            statuses_to_apply=["impaired"],
        )

        decisions = check_overheat_decisions(combatant, overheat_result, current_round=2)

        assert len(decisions) == 0


class TestCheckDangerousTerrainDecision:
    """Tests for check_dangerous_terrain_decision function."""

    def test_creates_engineering_check_decision(self):
        """Test creation of engineering check for dangerous terrain."""
        combatant = _make_combatant(engineering_skill=4)

        decision = check_dangerous_terrain_decision(
            combatant=combatant,
            terrain_name="lava",
            check_target=12,
            current_round=3,
        )

        assert decision.decision_type == "engineering_check"
        assert decision.save_type == "engineering"
        assert decision.save_target == 12
        assert decision.save_bonus == 4
        assert "lava" in decision.trigger_source


class TestScenarioHelpers:
    """Tests for scenario helper functions."""

    def test_get_pending_decisions_for_combatant(self):
        """Test getting pending decisions for a specific combatant."""
        decision1 = create_hull_save_decision("mech_1", trigger_round=1)
        decision2 = create_hull_save_decision("mech_2", trigger_round=1)
        decision3 = create_engineering_save_decision("mech_1", trigger_round=2)

        scenario = _make_scenario([
            _make_combatant("mech_1"),
            _make_combatant("mech_2"),
        ])
        scenario = scenario.model_copy(update={
            "pending_decisions": [decision1, decision2, decision3]
        })

        mech1_decisions = get_pending_decisions_for_combatant(scenario, "mech_1")
        mech2_decisions = get_pending_decisions_for_combatant(scenario, "mech_2")

        assert len(mech1_decisions) == 2
        assert len(mech2_decisions) == 1
        assert all(d.combatant_id == "mech_1" for d in mech1_decisions)
        assert all(d.combatant_id == "mech_2" for d in mech2_decisions)

    def test_remove_decision_from_scenario(self):
        """Test removing a decision from scenario."""
        decision1 = create_hull_save_decision("mech_1", trigger_round=1)
        decision2 = create_engineering_save_decision("mech_1", trigger_round=2)

        scenario = _make_scenario()
        scenario = scenario.model_copy(update={
            "pending_decisions": [decision1, decision2]
        })

        updated_scenario = remove_decision_from_scenario(scenario, decision1.decision_id)

        assert len(updated_scenario.pending_decisions) == 1
        assert updated_scenario.pending_decisions[0].decision_id == decision2.decision_id

    def test_add_decision_to_scenario(self):
        """Test adding a decision to scenario."""
        scenario = _make_scenario()
        decision = create_hull_save_decision("mech_1", trigger_round=1)

        updated_scenario = add_decision_to_scenario(scenario, decision)

        assert len(updated_scenario.pending_decisions) == 1
        assert updated_scenario.pending_decisions[0].decision_id == decision.decision_id

    def test_add_multiple_decisions(self):
        """Test adding multiple decisions."""
        scenario = _make_scenario()

        decision1 = create_hull_save_decision("mech_1", trigger_round=1)
        scenario = add_decision_to_scenario(scenario, decision1)

        decision2 = create_system_trauma_decision("mech_1", trigger_round=1, eligible_mounts=[0], eligible_systems=[])
        scenario = add_decision_to_scenario(scenario, decision2)

        assert len(scenario.pending_decisions) == 2


class TestDecisionResults:
    """Tests for decision result models."""

    def test_save_decision_result(self):
        """Test SaveDecisionResult model."""
        result = SaveDecisionResult(
            decision_id="test_1",
            voluntarily_failed=False,
            reroll_used=False,
            success=True,
        )
        assert result.success is True
        assert result.voluntarily_failed is False

    def test_trauma_decision_result(self):
        """Test TraumaDecisionResult model."""
        result = TraumaDecisionResult(
            decision_id="test_1",
            selected_target="mount",
            mount_index=1,
            valid_selection=True,
        )
        assert result.selected_target == "mount"
        assert result.mount_index == 1
        assert result.valid_selection is True
