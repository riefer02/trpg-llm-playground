"""Unit tests for combat state serializer."""

import pytest
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    HexPosition,
    HexCoord,
)
from core.shared.enums import SizeClass
from llm.src.tactician.state_serializer import serialize_combat_state


def test_serialize_empty_scenario():
    """Test serialization of empty combat scenario."""
    scenario = MechCombatScenario()
    result = serialize_combat_state(scenario)

    assert isinstance(result, dict)
    assert result["current_actor"] is None
    assert result["combatants"] == []
    assert result["terrain"] is None
    assert result["deployables"] == []
    assert result["grapples"] == []
    assert result["round_number"] == 1
    assert result["current_turn_index"] == 0
    assert result["available_actions"] == []
    assert "action_economy" in result


def test_serialize_single_combatant():
    """Test serialization with one combatant."""
    combatant = CombatantState(
        id="test_mech",
        name="Test Mech",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=10,
            evasion=10,
            e_defense=10,
            armor=0,
            speed=5,
            sensor_range=10,
            tech_attack=0,
            grit=0,
            engineering_skill=0,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=10,
            structure_current=3,
            stress_current=0,
            repairs_remaining=1,
            burn_marked=0,
        ),
        position=HexPosition(
            coord=HexCoord(q=0, r=0),
        ),
        statuses=[],
        conditions=[],
    )
    scenario = MechCombatScenario(combatants=[combatant])
    result = serialize_combat_state(scenario)

    assert len(result["combatants"]) == 1
    combatant_dict = result["combatants"][0]
    assert combatant_dict["id"] == "test_mech"
    assert combatant_dict["position"] is not None
    coord = combatant_dict["position"]["coord"]
    assert coord["q"] == 0
    assert coord["r"] == 0
    assert combatant_dict["stats"]["hp_max"] == 10
    assert combatant_dict["resources"]["hp_current"] == 10


def test_serialize_with_rounds_and_turns():
    """Test serialization with a round and turn in progress."""
    from core.mech.combat_state import CombatRound, CombatTurn, ActionUse
    from core.shared.enums import ActionType

    combatant = CombatantState(
        id="actor_1",
        name="Actor",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=10,
            evasion=10,
            e_defense=10,
            armor=0,
            speed=5,
            sensor_range=10,
            tech_attack=0,
            grit=0,
            engineering_skill=0,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=10,
            structure_current=3,
            stress_current=0,
            repairs_remaining=1,
            burn_marked=0,
        ),
    )

    # Create a round with one turn that has taken a quick action
    turn = CombatTurn(
        actor_id="actor_1",
        actions=[
            ActionUse(
                action_id="skirmish",
                action_type="quick",
                target_ids=["target_1"],
            )
        ],
    )
    round_data = CombatRound(
        round_index=1,
        turns=[turn],
    )

    scenario = MechCombatScenario(
        combatants=[combatant],
        rounds=[round_data],
    )

    result = serialize_combat_state(scenario)

    assert result["round_number"] == 1
    # current_turn_index should be 0 (last turn index)
    assert result["current_turn_index"] == 0
    # current actor should be actor_1
    assert result["current_actor"] == "actor_1"
    # economy should reflect one quick action used
    economy = result["action_economy"]
    assert economy["quick_actions_remaining"] == 1  # 2 - 1 used
    assert economy["full_actions_remaining"] == 1
