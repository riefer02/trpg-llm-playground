"""Tests for overwatch trigger detection mechanics.

Tests cover:
- OverwatchOpportunity model
- OverwatchTriggerResult model
- check_overwatch_triggers_at_movement_start() function
- Prevention via Disengage, Hidden, Invisible
- Reaction budget tracking
- Multiple enemies with overlapping threat ranges
"""

from core.shared.overwatch import (
    OverwatchOpportunity,
    OverwatchTriggerResult,
    check_overwatch_triggers_at_movement_start,
    check_overwatch_triggers_for_movement,
)
from core.shared.ids import CombatantId, WeaponId
from core.shared.effects import MechanicalEffect, ReactionTriggerEffect
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    MechInventory,
    WeaponMountState,
    WeaponState,
)
from core.mech.grid import HexPosition, HexCoord


# =============================================================================
# Test Fixtures
# =============================================================================


_DEFAULT_POSITION = object()  # Sentinel for default position


def make_combatant(
    id: str = "mech_1",
    name: str = "Test Mech",
    side: str = "players",
    hp_max: int = 10,
    hp_current: int = 10,
    position: HexPosition | None | object = _DEFAULT_POSITION,
    statuses: list[str] | None = None,
    inventory: MechInventory | None = None,
    per_round_reactions: dict[str, int] | None = None,
    talent_effects: list[MechanicalEffect] | None = None,
) -> CombatantState:
    """Create a test combatant."""
    if position is _DEFAULT_POSITION:
        position = HexPosition(coord=HexCoord(q=0, r=0), elevation=0)
    return CombatantState(
        id=id,
        name=name,
        side=side,
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=hp_max,
            evasion=8,
            e_defense=8,
            armor=0,
            speed=4,
            sensor_range=10,
        ),
        resources=CombatResources(
            hp_current=hp_current,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=position,
        statuses=statuses or [],
        inventory=inventory,
        per_round_reactions=per_round_reactions or {},
        talent_effects=talent_effects or [],
    )


def make_melee_weapon_inventory(weapon_id: str = "charged_blade") -> MechInventory:
    """Create an inventory with a melee weapon."""
    return MechInventory(
        mounts=[
            WeaponMountState(
                mount_index=0,
                weapons=[
                    WeaponState(
                        weapon_id=weapon_id,
                        tags=[],
                        destroyed=False,
                    )
                ],
            )
        ],
        systems=[],
    )


def make_ranged_weapon_inventory(weapon_id: str = "assault_rifle") -> MechInventory:
    """Create an inventory with a ranged weapon (no threat range)."""
    return MechInventory(
        mounts=[
            WeaponMountState(
                mount_index=0,
                weapons=[
                    WeaponState(
                        weapon_id=weapon_id,
                        tags=[],
                        destroyed=False,
                    )
                ],
            )
        ],
        systems=[],
    )


def make_scenario(combatants: list[CombatantState]) -> MechCombatScenario:
    """Create a test scenario."""
    return MechCombatScenario(
        combatants=combatants,
        grapples=[],
        rounds=[],
        terrain=None,
        environment="standard",
        deployables={},
    )


# =============================================================================
# OverwatchOpportunity Model Tests
# =============================================================================


class TestOverwatchOpportunity:
    """Tests for OverwatchOpportunity model."""

    def test_basic_opportunity(self):
        """Test creating a basic overwatch opportunity."""
        opportunity = OverwatchOpportunity(
            reactor_id=CombatantId("enemy_1"),
            weapon_id=WeaponId("charged_blade"),
            weapon_threat=1,
            target_id=CombatantId("player_1"),
            target_position=HexCoord(q=0, r=0),
            can_react=True,
        )
        assert opportunity.reactor_id == "enemy_1"
        assert opportunity.weapon_id == "charged_blade"
        assert opportunity.weapon_threat == 1
        assert opportunity.target_id == "player_1"
        assert opportunity.can_react is True
        assert opportunity.prevention_reason is None

    def test_opportunity_with_prevention(self):
        """Test opportunity with reaction prevention."""
        opportunity = OverwatchOpportunity(
            reactor_id=CombatantId("enemy_1"),
            weapon_id=WeaponId("charged_blade"),
            weapon_threat=1,
            target_id=CombatantId("player_1"),
            target_position=HexCoord(q=0, r=0),
            can_react=False,
            prevention_reason="Overwatch already used this round",
        )
        assert opportunity.can_react is False
        assert opportunity.prevention_reason == "Overwatch already used this round"


# =============================================================================
# OverwatchTriggerResult Model Tests
# =============================================================================


class TestOverwatchTriggerResult:
    """Tests for OverwatchTriggerResult model."""

    def test_empty_result(self):
        """Test creating an empty result."""
        result = OverwatchTriggerResult()
        assert result.opportunities == []
        assert result.reactions_prevented is False
        assert result.prevention_reason is None

    def test_result_with_opportunities(self):
        """Test result with multiple opportunities."""
        opps = [
            OverwatchOpportunity(
                reactor_id=CombatantId("e1"),
                weapon_id=WeaponId("blade1"),
                weapon_threat=1,
                target_id=CombatantId("p1"),
                target_position=HexCoord(q=0, r=0),
                can_react=True,
            ),
            OverwatchOpportunity(
                reactor_id=CombatantId("e2"),
                weapon_id=WeaponId("blade2"),
                weapon_threat=2,
                target_id=CombatantId("p1"),
                target_position=HexCoord(q=0, r=0),
                can_react=True,
            ),
        ]
        result = OverwatchTriggerResult(opportunities=opps)
        assert len(result.opportunities) == 2
        assert result.opportunities[0].reactor_id == "e1"
        assert result.opportunities[1].reactor_id == "e2"

    def test_result_with_global_prevention(self):
        """Test result when all reactions are prevented."""
        result = OverwatchTriggerResult(
            opportunities=[],
            reactions_prevented=True,
            prevention_reason="Disengage prevents overwatch triggers",
        )
        assert result.reactions_prevented is True
        assert "Disengage" in result.prevention_reason


# =============================================================================
# Overwatch Trigger Detection Tests
# =============================================================================


class TestOverwatchTriggerDetection:
    """Tests for check_overwatch_triggers_at_movement_start function."""

    def test_movement_in_threat_range_triggers_opportunity(self):
        """Movement starting in enemy threat range triggers overwatch opportunity."""
        # Player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Enemy mech at (1,0) with melee weapon (threat 1)
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 1
        assert result.opportunities[0].reactor_id == "enemy_1"
        assert result.opportunities[0].weapon_id == "charged_blade"
        assert result.opportunities[0].weapon_threat == 1
        assert result.opportunities[0].target_id == "player_1"
        assert result.opportunities[0].can_react is True
        assert result.reactions_prevented is False

    def test_movement_outside_threat_range_no_trigger(self):
        """Movement outside enemy threat range does not trigger overwatch."""
        # Player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Enemy mech at (3,0) with melee weapon (threat 1) - distance 3 > threat 1
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=3, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 0
        assert result.reactions_prevented is False

    def test_threat_range_3_weapon_triggers_at_distance_2(self):
        """Extended threat weapon (threat 3) triggers at distance 2."""
        # Player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Enemy mech at (2,0) with impact lance (threat 3)
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("ipsn_impact_lance"),
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 1
        assert result.opportunities[0].weapon_threat == 3

    def test_multiple_enemies_multiple_opportunities(self):
        """Multiple enemies can each have overwatch opportunities."""
        # Player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Two enemy mechs adjacent
        enemy1 = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        enemy2 = make_combatant(
            id="enemy_2",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=0, r=1), elevation=0),
            inventory=make_melee_weapon_inventory("heavy_melee_weapon"),
        )
        scenario = make_scenario([player, enemy1, enemy2])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 2
        reactor_ids = {opp.reactor_id for opp in result.opportunities}
        assert "enemy_1" in reactor_ids
        assert "enemy_2" in reactor_ids

    def test_enemy_with_ranged_weapon_triggers_overwatch(self):
        """Ranged weapons still threaten at default 1 for overwatch."""
        # Player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Enemy mech adjacent with ranged weapon only
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_ranged_weapon_inventory("heavy_machine_gun"),  # Ranged only
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 1
        assert result.opportunities[0].weapon_threat == 1

    def test_entering_threat_without_trigger_no_opportunity(self):
        """Entering threat does not trigger overwatch without a trigger effect."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        pistol_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[WeaponState(weapon_id="pistol", tags=[], destroyed=False)],
                )
            ],
            systems=[],
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=4, r=0), elevation=0),
            inventory=pistol_inventory,
        )
        scenario = make_scenario([player, enemy])

        movement_path = [HexPosition(coord=HexCoord(q=1, r=0), elevation=0)]
        result = check_overwatch_triggers_for_movement(
            scenario=scenario,
            mover=player,
            movement_path=movement_path,
        )

        assert len(result.opportunities) == 0

    def test_entering_threat_with_cqb_trigger(self):
        """Semper Vigilo-style triggers fire on entering CQB threat."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        pistol_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[WeaponState(weapon_id="pistol", tags=[], destroyed=False)],
                )
            ],
            systems=[],
        )
        semper_vigilo = MechanicalEffect(
            reaction_triggers=[
                ReactionTriggerEffect(
                    reaction_id="overwatch",
                    trigger_events=["enemy_enters_threat"],
                    condition="cqb_overwatch",
                )
            ]
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=4, r=0), elevation=0),
            inventory=pistol_inventory,
            talent_effects=[semper_vigilo],
        )
        scenario = make_scenario([player, enemy])

        movement_path = [HexPosition(coord=HexCoord(q=1, r=0), elevation=0)]
        result = check_overwatch_triggers_for_movement(
            scenario=scenario,
            mover=player,
            movement_path=movement_path,
        )

        assert len(result.opportunities) == 1
        assert result.opportunities[0].reactor_id == "enemy_1"
        assert result.opportunities[0].weapon_id == "pistol"
        assert result.opportunities[0].weapon_threat == 3

    def test_enemy_already_used_overwatch_cannot_react(self):
        """Enemy who already used overwatch this round cannot react."""
        # Player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Enemy mech adjacent with overwatch already used
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
            per_round_reactions={"overwatch": 1},
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 1
        assert result.opportunities[0].can_react is False
        assert "already used" in result.opportunities[0].prevention_reason

    def test_ally_does_not_trigger_overwatch(self):
        """Allies (same side) do not trigger overwatch against each other."""
        # Two player mechs adjacent
        player1 = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        player2 = make_combatant(
            id="player_2",
            side="players",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        scenario = make_scenario([player1, player2])

        result = check_overwatch_triggers_at_movement_start(scenario, player1)

        assert len(result.opportunities) == 0

    def test_destroyed_weapon_no_trigger(self):
        """Destroyed weapons cannot be used for overwatch."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Enemy with destroyed melee weapon
        destroyed_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[
                        WeaponState(
                            weapon_id="charged_blade",
                            tags=[],
                            destroyed=True,  # Destroyed
                        )
                    ],
                )
            ],
            systems=[],
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=destroyed_inventory,
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 0

    def test_thrown_weapon_no_trigger(self):
        """Thrown weapons (not retrieved) cannot be used for overwatch."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Enemy with thrown melee weapon
        thrown_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[
                        WeaponState(
                            weapon_id="tactical_knife",  # Has thrown tag
                            tags=[],
                            thrown_coord=HexCoord(q=5, r=5),  # Currently thrown
                        )
                    ],
                )
            ],
            systems=[],
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=thrown_inventory,
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 0

    def test_enemy_without_position_no_trigger(self):
        """Enemy without a position cannot trigger overwatch."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=None,  # No position
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 0

    def test_mover_without_position_no_trigger(self):
        """Mover without a position cannot trigger overwatch."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=None,  # No position
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        assert len(result.opportunities) == 0
        assert "no position" in result.prevention_reason.lower()


# =============================================================================
# Overwatch Prevention Tests
# =============================================================================


class TestOverwatchPrevention:
    """Tests for conditions that prevent overwatch triggers."""

    def test_disengage_prevents_overwatch_triggers(self):
        """Disengage action prevents all overwatch triggers."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(
            scenario, player, is_disengaging=True
        )

        assert len(result.opportunities) == 0
        assert result.reactions_prevented is True
        assert "Disengage" in result.prevention_reason

    def test_hidden_prevents_overwatch_triggers(self):
        """Hidden status prevents overwatch triggers."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(
            scenario, player, is_hidden=True
        )

        assert len(result.opportunities) == 0
        assert result.reactions_prevented is True
        assert "Hidden" in result.prevention_reason

    def test_invisible_prevents_overwatch_triggers(self):
        """Invisible status prevents overwatch triggers."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(
            scenario, player, is_invisible=True
        )

        assert len(result.opportunities) == 0
        assert result.reactions_prevented is True
        assert "Invisible" in result.prevention_reason

    def test_stunned_enemy_cannot_react(self):
        """Stunned enemy cannot use overwatch reaction."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
            statuses=["stunned"],
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        # Stunned enemy is skipped entirely
        assert len(result.opportunities) == 0

    def test_shutdown_enemy_cannot_react(self):
        """Shutdown enemy cannot use overwatch reaction."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
            statuses=["shutdown"],
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        # Shutdown enemy is skipped entirely
        assert len(result.opportunities) == 0

    def test_incapacitated_enemy_cannot_react(self):
        """Enemy with 0 HP cannot use overwatch reaction."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            hp_current=0,  # Incapacitated
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=make_melee_weapon_inventory("charged_blade"),
        )
        scenario = make_scenario([player, enemy])

        result = check_overwatch_triggers_at_movement_start(scenario, player)

        # Incapacitated enemy is skipped entirely
        assert len(result.opportunities) == 0
