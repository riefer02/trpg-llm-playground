from core.mech.combat_state import (
    ActionUse,
    CombatantState,
    CombatResources,
    CombatRound,
    CombatStats,
    CombatTurn,
    MechCombatScenario,
)
from core.mech.validation.combat_validation import validate_combat_scenario
from core.mech.grid import HexCoord, HexPosition
from core.shared.effects import (
    ActionRestriction,
    MechanicalEffect,
    ModeEffect,
    ReactionTriggerEffect,
)


def _base_combatant(
    combatant_id: str,
    side: str,
    coord: HexCoord,
    active_mode_effects: list[ModeEffect] | None = None,
    reaction_triggers: list[ReactionTriggerEffect] | None = None,
) -> CombatantState:
    return CombatantState(
        id=combatant_id,
        name=combatant_id.title(),
        side=side,
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=10,
            evasion=10,
            e_defense=8,
            armor=0,
            speed=4,
            sensor_range=10,
            tech_attack=0,
        ),
        resources=CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=HexPosition(coord=coord, elevation=0),
        active_mode_effects=active_mode_effects or [],
        reaction_triggers=reaction_triggers or [],
    )


def test_mode_heat_generation_and_overwatch_override() -> None:
    reserve_power_mode = ModeEffect(
        name="Reserve Power Mode",
        activation_action_id="shutdown",
        activation_action_type="quick",
        deactivation_action_id="boot_up",
        deactivation_action_type="full",
        duration="until_cleared",
        effects=MechanicalEffect(
            action_restrictions=[ActionRestriction(disallow_heat_generation=True)]
        ),
    )

    alpha = _base_combatant(
        "alpha",
        "players",
        HexCoord(q=0, r=0),
        active_mode_effects=[reserve_power_mode],
        reaction_triggers=[
            ReactionTriggerEffect(
                reaction_id="overwatch",
                trigger_events=["enemy_enters_threat"],
            )
        ],
    )
    bravo = _base_combatant("bravo", "hostiles", HexCoord(q=2, r=0))

    round_one = CombatRound(
        round_index=1,
        turns=[
            CombatTurn(
                actor_id="alpha",
                move_used=True,
                actions=[
                    ActionUse(
                        action_id="skirmish",
                        action_type="quick",
                        target_id="bravo",
                        attack_type_override="ranged",
                        range_spaces=5,
                        weapon_count=1,
                        uses_superheavy=False,
                        uses_aux_bonus_attack=False,
                        heat_generated=1,
                    ),
                    ActionUse(
                        action_id="overwatch",
                        action_type="reaction",
                        target_id="bravo",
                        attack_type_override="ranged",
                        range_spaces=3,
                        weapon_count=1,
                        uses_superheavy=False,
                        uses_aux_bonus_attack=False,
                        reaction_trigger="enemy_enters_threat",
                    ),
                ],
            )
        ],
    )

    scenario = MechCombatScenario(combatants=[alpha, bravo], rounds=[round_one])
    validation = validate_combat_scenario(scenario)

    error_codes = {
        issue.code for issue in validation.issues if issue.severity == "error"
    }
    warning_codes = {
        issue.code for issue in validation.issues if issue.severity == "warning"
    }

    assert "mode_heat_generation_disallowed" in error_codes
    assert "overwatch_trigger_invalid" not in error_codes
    assert "overwatch_trigger_missing" not in warning_codes
