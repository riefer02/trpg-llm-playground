"""Tests for tag defaults (mine/drone/deployable) and danger zone."""

from __future__ import annotations


from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
    DeployableState,
    DeployableKind,
    MechCombatScenario,
    HexPosition,
    HexCoord,
)
from core.mech.combat_rules import (
    TagDefaultRules,
)
from core.mech.combat_resolution import (
    deploy_object,
    damage_deployable,
    check_mine_trigger,
    get_combatants_in_danger_zone,
    DeploymentResult,
    DeployableDamageResult,
    MineTriggerResult,
)
from core.mech.validation.combat_validation import (
    validate_deployment,
    validate_mine_detection,
    validate_mine_disarm,
)


class TestDeployableKind:
    """Tests for DeployableKind type."""

    def test_deployable_kind_values(self):
        """Verify DeployableKind accepts expected values."""
        assert DeployableKind.__args__ == ("drone", "mine", "deployable", "other")


class TestDeployableState:
    """Tests for DeployableState model."""

    def test_minimal_drone(self):
        """Create minimal drone with defaults."""
        state = DeployableState(
            id="drone-1",
            name="Turret Drone",
            kind="drone",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            size=1,
            hp=10,
            max_hp=10,
        )
        assert state.kind == "drone"
        assert state.evasion == 5  # Default for deployable
        assert state.can_act is False
        assert state.acts_on_owner_turn is True

    def test_drone_with_pr2_defaults(self):
        """Create drone with PR2 default stats."""
        state = DeployableState(
            id="drone-1",
            name="Turret Drone",
            kind="drone",
            owner_id="pilot-1",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            size=1,
            hp=10,
            max_hp=10,
            evasion=10,  # PR2 drone default
            e_defense=10,
            armor=0,
            can_act=True,
            can_move=False,
            acts_on_owner_turn=True,
        )
        assert state.evasion == 10
        assert state.e_defense == 10
        assert state.armor == 0

    def test_mine_with_arming(self):
        """Create mine with arming state."""
        state = DeployableState(
            id="mine-1",
            name="Explosive Mine",
            kind="mine",
            owner_id="pilot-1",
            position=HexPosition(coord=HexCoord(q=5, r=5)),
            size=1,
            hp=5,
            max_hp=5,
            is_armed=False,
            arming_turn=2,
            trigger_on_adjacent_entry=True,
            detection_dc=12,
            disarm_dc=12,
        )
        assert state.is_armed is False
        assert state.arming_turn == 2
        assert state.trigger_on_adjacent_entry is True

    def test_deployable_with_cover(self):
        """Create deployable with cover."""
        state = DeployableState(
            id="cover-1",
            name="Deployable Cover",
            kind="deployable",
            position=HexPosition(coord=HexCoord(q=3, r=3)),
            size=1,
            hp=10,
            max_hp=10,
            cover="hard",
        )
        assert state.cover == "hard"


class TestCombatantStateDangerZone:
    """Tests for CombatantState.in_danger_zone() helper."""

    def _make_combatant(self, heat_current: int, heat_cap: int) -> CombatantState:
        """Helper to create combatant with specific heat."""
        return CombatantState(
            id="test-mech",
            name="Test Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_2",
                hp_max=30,
                evasion=10,
                e_defense=10,
            ),
            resources=CombatResources(
                hp_current=30,
                heat_current=heat_current,
                heat_cap=heat_cap,
            ),
        )

    def test_below_danger_zone(self):
        """Combatant below danger zone threshold."""
        mech = self._make_combatant(heat_current=2, heat_cap=6)
        assert mech.in_danger_zone() is False

    def test_at_danger_zone_threshold_up(self):
        """Combatant at danger zone threshold (rounded up)."""
        mech = self._make_combatant(heat_current=3, heat_cap=6)
        assert mech.in_danger_zone() is True

    def test_above_danger_zone(self):
        """Combatant well above danger zone threshold."""
        mech = self._make_combatant(heat_current=5, heat_cap=6)
        assert mech.in_danger_zone() is True

    def test_danger_zone_odd_capacity(self):
        """Combatant with odd heat capacity (rounds up)."""
        mech = self._make_combatant(heat_current=3, heat_cap=5)
        assert mech.in_danger_zone() is True

    def test_danger_zone_rounding_down(self):
        """Combatant with rounding=down."""
        mech = self._make_combatant(heat_current=2, heat_cap=5)
        # With rounding down, threshold = floor(5 * 0.5) = 2
        # So heat_current=2 puts mech at exactly threshold
        assert mech.in_danger_zone(rounding="down") is True
        assert mech.in_danger_zone(rounding="up") is False

    def test_no_heat_capacity(self):
        """Combatant with no heat capacity."""
        mech = self._make_combatant(heat_current=0, heat_cap=0)
        assert mech.in_danger_zone() is False


class TestTagDefaultRules:
    """Tests for TagDefaultRules model."""

    def test_defaults(self):
        """Verify default values match PR2 rules."""
        rules = TagDefaultRules()
        assert rules.deployable_default_hp_per_size == 10
        assert rules.deployable_default_evasion == 5
        assert rules.deployable_default_armor == 0
        assert rules.drone_default_hp == 10
        assert rules.drone_default_evasion == 10
        assert rules.drone_default_armor == 0
        assert rules.mine_arming_delay_turns == 1
        assert rules.danger_zone_fraction == 0.5

    def test_custom_values(self):
        """Test custom tag default values."""
        rules = TagDefaultRules(
            deployable_default_hp_per_size=15,
            drone_default_evasion=12,
            mine_arming_delay_turns=2,
        )
        assert rules.deployable_default_hp_per_size == 15
        assert rules.drone_default_evasion == 12
        assert rules.mine_arming_delay_turns == 2


class TestDeployObject:
    """Tests for deploy_object resolution helper."""

    def _make_scenario(self) -> MechCombatScenario:
        """Create empty scenario."""
        return MechCombatScenario(
            combatants=[
                CombatantState(
                    id="pilot-1",
                    name="Test Pilot",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                    ),
                    resources=CombatResources(
                        hp_current=30,
                        heat_cap=6,
                    ),
                    position=HexPosition(coord=HexCoord(q=0, r=0)),
                )
            ],
        )

    def test_deploy_drone(self):
        """Deploy a drone."""
        scenario = self._make_scenario()
        new_scenario, result = deploy_object(
            scenario=scenario,
            deployable_id="drone-1",
            name="Turret Drone",
            kind="drone",
            owner_id="pilot-1",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
            size=1,
            hp=10,
            max_hp=10,
            evasion=10,
            can_act=True,
        )
        assert isinstance(result, DeploymentResult)
        assert result.deployable_id == "drone-1"
        assert result.kind == "drone"
        assert result.evasion == 10
        assert "drone-1" in new_scenario.deployables

    def test_deploy_mine(self):
        """Deploy a mine."""
        scenario = self._make_scenario()
        new_scenario, result = deploy_object(
            scenario=scenario,
            deployable_id="mine-1",
            name="Explosive Mine",
            kind="mine",
            owner_id="pilot-1",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
            size=1,
            hp=5,
            max_hp=5,
            is_armed=False,
            arming_turn=2,
        )
        assert result.kind == "mine"
        assert result.is_armed is False
        assert result.arming_turn == 2

    def test_deploy_deployable(self):
        """Deploy a deployable cover."""
        scenario = self._make_scenario()
        new_scenario, result = deploy_object(
            scenario=scenario,
            deployable_id="cover-1",
            name="Deployable Cover",
            kind="deployable",
            owner_id="pilot-1",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
            size=1,
            hp=10,
            max_hp=10,
            cover="hard",
        )
        assert result.kind == "deployable"
        deployed = new_scenario.deployables["cover-1"]
        assert deployed.cover == "hard"


class TestDamageDeployable:
    """Tests for damage_deployable resolution helper."""

    def _make_scenario_with_deployable(
        self, hp: int = 10, armor: int = 0
    ) -> tuple[MechCombatScenario, str]:
        """Create scenario with a deployable."""
        scenario = MechCombatScenario(
            combatants=[],
            deployables={
                "deployable-1": DeployableState(
                    id="deployable-1",
                    name="Test Deployable",
                    kind="deployable",
                    position=HexPosition(coord=HexCoord(q=0, r=0)),
                    size=1,
                    hp=hp,
                    max_hp=10,
                    armor=armor,
                )
            },
        )
        return scenario, "deployable-1"

    def test_damage_deployable(self):
        """Apply damage to deployable."""
        scenario, deployable_id = self._make_scenario_with_deployable(hp=10)
        new_scenario, result = damage_deployable(
            scenario=scenario,
            deployable_id=deployable_id,
            damage=5,
        )
        assert isinstance(result, DeployableDamageResult)
        assert result.damage_dealt == 5
        assert result.hp_before == 10
        assert result.hp_after == 5
        assert result.is_destroyed is False

    def test_destroy_deployable(self):
        """Destroy deployable with sufficient damage."""
        scenario, deployable_id = self._make_scenario_with_deployable(hp=5)
        new_scenario, result = damage_deployable(
            scenario=scenario,
            deployable_id=deployable_id,
            damage=10,
        )
        assert result.is_destroyed is True
        assert result.destroyed is True
        assert new_scenario.deployables[deployable_id].is_destroyed is True

    def test_armor_reduction(self):
        """Armor reduces damage."""
        scenario, deployable_id = self._make_scenario_with_deployable(hp=10, armor=2)
        new_scenario, result = damage_deployable(
            scenario=scenario,
            deployable_id=deployable_id,
            damage=5,
            armor_piercing=0,
        )
        assert result.damage_dealt == 3  # 5 - 2 armor

    def test_armor_piercing(self):
        """Armor piercing ignores armor."""
        scenario, deployable_id = self._make_scenario_with_deployable(hp=10, armor=2)
        new_scenario, result = damage_deployable(
            scenario=scenario,
            deployable_id=deployable_id,
            damage=5,
            armor_piercing=3,  # More than armor
        )
        assert result.damage_dealt == 5  # Full damage


class TestCheckMineTrigger:
    """Tests for check_mine_trigger resolution helper."""

    def _make_scenario_with_mines(self) -> MechCombatScenario:
        """Create scenario with armed and unarmed mines.

        Uses size_1 mech to test mine arming behavior independent of
        size-aware adjacency rules.
        """
        return MechCombatScenario(
            combatants=[
                CombatantState(
                    id="mech-1",
                    name="Test Mech",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_1",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                    ),
                    resources=CombatResources(hp_current=30),
                    position=HexPosition(coord=HexCoord(q=0, r=0)),
                )
            ],
            deployables={
                "armed-mine": DeployableState(
                    id="armed-mine",
                    name="Armed Mine",
                    kind="mine",
                    position=HexPosition(coord=HexCoord(q=1, r=0)),
                    size=1,
                    hp=5,
                    max_hp=5,
                    is_armed=True,
                    trigger_on_adjacent_entry=True,
                ),
                "unarmed-mine": DeployableState(
                    id="unarmed-mine",
                    name="Unarmed Mine",
                    kind="mine",
                    position=HexPosition(coord=HexCoord(q=2, r=0)),
                    size=1,
                    hp=5,
                    max_hp=5,
                    is_armed=False,
                ),
            },
        )

    def test_trigger_armed_mine(self):
        """Moving adjacent to armed mine triggers it."""
        scenario = self._make_scenario_with_mines()
        results = check_mine_trigger(
            scenario=scenario,
            moving_combatant_id="mech-1",
            new_position=HexPosition(coord=HexCoord(q=1, r=1)),  # Adjacent to (1,0)
        )
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, MineTriggerResult)
        assert result.mine_id == "armed-mine"
        assert result.detonated is True

    def test_no_trigger_unarmed_mine(self):
        """Moving adjacent to unarmed mine does not trigger it."""
        scenario = self._make_scenario_with_mines()
        results = check_mine_trigger(
            scenario=scenario,
            moving_combatant_id="mech-1",
            new_position=HexPosition(coord=HexCoord(q=2, r=1)),  # Adjacent to (2,0)
        )
        assert len(results) == 0

    def test_no_trigger_distant_mine(self):
        """Moving far from mine does not trigger it."""
        scenario = self._make_scenario_with_mines()
        results = check_mine_trigger(
            scenario=scenario,
            moving_combatant_id="mech-1",
            new_position=HexPosition(coord=HexCoord(q=5, r=5)),  # Far from (1,0)
        )
        assert len(results) == 0


class TestGetCombatantsInDangerZone:
    """Tests for get_combatants_in_danger_zone resolution helper."""

    def _make_scenario_with_combatants(self) -> MechCombatScenario:
        """Create scenario with combatants at different heat levels."""
        return MechCombatScenario(
            combatants=[
                CombatantState(
                    id="safe-mech",
                    name="Safe Mech",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                    ),
                    resources=CombatResources(
                        hp_current=30,
                        heat_current=1,
                        heat_cap=6,
                    ),
                ),
                CombatantState(
                    id="danger-mech",
                    name="Danger Mech",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                    ),
                    resources=CombatResources(
                        hp_current=30,
                        heat_current=4,
                        heat_cap=6,
                    ),
                ),
            ],
        )

    def test_get_danger_zone_combatants(self):
        """Get combatants in danger zone."""
        scenario = self._make_scenario_with_combatants()
        results = get_combatants_in_danger_zone(scenario)
        assert len(results) == 2
        safe_status = next(r for r in results if r.combatant_id == "safe-mech")
        danger_status = next(r for r in results if r.combatant_id == "danger-mech")
        assert safe_status.in_danger_zone is False
        assert danger_status.in_danger_zone is True
        assert danger_status.danger_zone_threshold == 3


class TestValidateDeployment:
    """Tests for validate_deployment validation helper."""

    def _make_scenario(self) -> MechCombatScenario:
        """Create empty scenario."""
        return MechCombatScenario(
            combatants=[
                CombatantState(
                    id="pilot-1",
                    name="Test Pilot",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                        sensor_range=10,
                    ),
                    resources=CombatResources(hp_current=30),
                    position=HexPosition(coord=HexCoord(q=0, r=0)),
                )
            ],
        )

    def test_valid_adjacent_deployment(self):
        """Valid deployment to adjacent space."""
        scenario = self._make_scenario()
        issues = validate_deployment(
            scenario=scenario,
            deployer_id="pilot-1",
            target_position=HexPosition(coord=HexCoord(q=1, r=0)),
            kind="deployable",
            deploy_range=1,
        )
        assert len(issues) == 0

    def test_out_of_range_deployment(self):
        """Deployment beyond range fails."""
        scenario = self._make_scenario()
        issues = validate_deployment(
            scenario=scenario,
            deployer_id="pilot-1",
            target_position=HexPosition(coord=HexCoord(q=5, r=0)),
            kind="deployable",
            deploy_range=1,
        )
        assert len(issues) == 1
        assert issues[0].code == "deployment_out_of_range"

    def test_deployment_on_occupied_space(self):
        """Deployment on occupied space fails."""
        scenario = MechCombatScenario(
            combatants=[
                CombatantState(
                    id="pilot-1",
                    name="Pilot 1",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                    ),
                    resources=CombatResources(hp_current=30),
                    position=HexPosition(coord=HexCoord(q=1, r=0)),
                ),
                CombatantState(
                    id="pilot-2",
                    name="Pilot 2",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                    ),
                    resources=CombatResources(hp_current=30),
                    position=HexPosition(coord=HexCoord(q=0, r=0)),
                ),
            ],
        )
        issues = validate_deployment(
            scenario=scenario,
            deployer_id="pilot-2",
            target_position=HexPosition(coord=HexCoord(q=1, r=0)),
            kind="deployable",
        )
        assert len(issues) == 1
        assert issues[0].code == "deployment_space_occupied"

    def test_mine_too_close_to_mine(self):
        """Mines cannot be placed adjacent to other mines."""
        scenario = MechCombatScenario(
            combatants=[
                CombatantState(
                    id="pilot-1",
                    name="Test Pilot",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                    ),
                    resources=CombatResources(hp_current=30),
                    position=HexPosition(coord=HexCoord(q=0, r=0)),
                )
            ],
            deployables={
                "existing-mine": DeployableState(
                    id="existing-mine",
                    name="Existing Mine",
                    kind="mine",
                    position=HexPosition(coord=HexCoord(q=1, r=0)),
                    size=1,
                    hp=5,
                    max_hp=5,
                ),
            },
        )
        # Place new mine at (0, 1) - adjacent to pilot at (0, 0) and adjacent to existing mine at (1, 0)
        issues = validate_deployment(
            scenario=scenario,
            deployer_id="pilot-1",
            target_position=HexPosition(coord=HexCoord(q=0, r=1)),
            kind="mine",
        )
        assert len(issues) == 1
        assert issues[0].code == "mine_too_close"


class TestValidateMineDetection:
    """Tests for validate_mine_detection validation helper."""

    def _make_scenario_with_mine(self) -> MechCombatScenario:
        """Create scenario with a mine."""
        return MechCombatScenario(
            combatants=[
                CombatantState(
                    id="pilot-1",
                    name="Test Pilot",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                        sensor_range=10,
                    ),
                    resources=CombatResources(hp_current=30),
                    position=HexPosition(coord=HexCoord(q=0, r=0)),
                )
            ],
            deployables={
                "mine-1": DeployableState(
                    id="mine-1",
                    name="Explosive Mine",
                    kind="mine",
                    position=HexPosition(coord=HexCoord(q=5, r=0)),
                    size=1,
                    hp=5,
                    max_hp=5,
                ),
            },
        )

    def test_valid_detection(self):
        """Valid mine detection within sensor range."""
        scenario = self._make_scenario_with_mine()
        issues = validate_mine_detection(
            scenario=scenario,
            detector_id="pilot-1",
            mine_id="mine-1",
        )
        assert len(issues) == 0

    def test_out_of_sensor_range(self):
        """Detection fails if out of sensor range."""
        scenario = MechCombatScenario(
            combatants=[
                CombatantState(
                    id="pilot-1",
                    name="Test Pilot",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                        sensor_range=3,
                    ),
                    resources=CombatResources(hp_current=30),
                    position=HexPosition(coord=HexCoord(q=0, r=0)),
                )
            ],
            deployables={
                "mine-1": DeployableState(
                    id="mine-1",
                    name="Explosive Mine",
                    kind="mine",
                    position=HexPosition(coord=HexCoord(q=5, r=0)),
                    size=1,
                    hp=5,
                    max_hp=5,
                ),
            },
        )
        issues = validate_mine_detection(
            scenario=scenario,
            detector_id="pilot-1",
            mine_id="mine-1",
        )
        assert len(issues) == 1
        assert issues[0].code == "mine_out_of_sensor_range"

    def test_not_a_mine(self):
        """Detection fails for non-mine deployables."""
        scenario = MechCombatScenario(
            combatants=[
                CombatantState(
                    id="pilot-1",
                    name="Test Pilot",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                        sensor_range=10,
                    ),
                    resources=CombatResources(hp_current=30),
                )
            ],
            deployables={
                "drone-1": DeployableState(
                    id="drone-1",
                    name="Turret Drone",
                    kind="drone",
                    position=HexPosition(coord=HexCoord(q=1, r=0)),
                    size=1,
                    hp=10,
                    max_hp=10,
                ),
            },
        )
        issues = validate_mine_detection(
            scenario=scenario,
            detector_id="pilot-1",
            mine_id="drone-1",
        )
        assert len(issues) == 1
        assert issues[0].code == "not_a_mine"


class TestValidateMineDisarm:
    """Tests for validate_mine_disarm validation helper."""

    def _make_scenario_with_mine(self) -> MechCombatScenario:
        """Create scenario with a mine."""
        return MechCombatScenario(
            combatants=[
                CombatantState(
                    id="pilot-1",
                    name="Test Pilot",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                    ),
                    resources=CombatResources(hp_current=30),
                    position=HexPosition(coord=HexCoord(q=1, r=1)),
                )
            ],
            deployables={
                "mine-1": DeployableState(
                    id="mine-1",
                    name="Explosive Mine",
                    kind="mine",
                    position=HexPosition(coord=HexCoord(q=1, r=0)),
                    size=1,
                    hp=5,
                    max_hp=5,
                ),
            },
        )

    def test_valid_disarm(self):
        """Valid mine disarm when adjacent."""
        scenario = self._make_scenario_with_mine()
        issues = validate_mine_disarm(
            scenario=scenario,
            disarmer_id="pilot-1",
            mine_id="mine-1",
        )
        assert len(issues) == 0

    def test_not_adjacent(self):
        """Disarm fails when not adjacent to mine."""
        scenario = MechCombatScenario(
            combatants=[
                CombatantState(
                    id="pilot-1",
                    name="Test Pilot",
                    side="players",
                    kind="mech",
                    stats=CombatStats(
                        size="size_2",
                        hp_max=30,
                        evasion=10,
                        e_defense=10,
                    ),
                    resources=CombatResources(hp_current=30),
                    position=HexPosition(coord=HexCoord(q=3, r=3)),
                )
            ],
            deployables={
                "mine-1": DeployableState(
                    id="mine-1",
                    name="Explosive Mine",
                    kind="mine",
                    position=HexPosition(coord=HexCoord(q=1, r=0)),
                    size=1,
                    hp=5,
                    max_hp=5,
                ),
            },
        )
        issues = validate_mine_disarm(
            scenario=scenario,
            disarmer_id="pilot-1",
            mine_id="mine-1",
        )
        assert len(issues) == 1
        assert issues[0].code == "not_adjacent_to_mine"


class TestMechCombatScenarioDeployables:
    """Tests for MechCombatScenario.deployables field."""

    def test_empty_deployables(self):
        """Scenario starts with empty deployables dict."""
        scenario = MechCombatScenario()
        assert scenario.deployables == {}

    def test_deployables_field_exists(self):
        """MechCombatScenario has deployables field."""
        assert "deployables" in MechCombatScenario.model_fields
