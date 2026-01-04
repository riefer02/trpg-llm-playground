from core.mech.combat_state import (
    ActionUse,
    AppliedPerTargetEffect,
    CombatantState,
    CombatResources,
    CombatRound,
    CombatStats,
    CombatTurn,
    MechCombatScenario,
)
from core.mech.combat_validation import validate_combat_scenario
from core.mech.grid import HexCoord, HexPosition
from core.shared.effects import PerTargetCounter


def _combatant(
    combatant_id: str,
    side: str,
    coord: HexCoord,
    per_target_counters: dict[str, PerTargetCounter] | None = None,
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
        per_target_counters=per_target_counters or {},
    )


def _scenario(
    actions: list[ActionUse],
    per_target_counters: dict[str, PerTargetCounter] | None = None,
) -> MechCombatScenario:
    alpha = _combatant(
        "alpha",
        "players",
        HexCoord(q=0, r=0),
        per_target_counters=per_target_counters,
    )
    bravo = _combatant("bravo", "hostiles", HexCoord(q=1, r=0))
    round_one = CombatRound(
        round_index=1,
        turns=[CombatTurn(actor_id="alpha", actions=actions)],
    )
    return MechCombatScenario(combatants=[alpha, bravo], rounds=[round_one])


def test_per_target_limit_exceeded_with_template() -> None:
    template_counters = {
        "basilisk_stun": PerTargetCounter(effect_id="basilisk_stun", max_count=1),
    }
    actions = [
        ActionUse(
            action_id="invade",
            action_type="quick",
            target_id="bravo",
            applied_per_target_effects=[
                AppliedPerTargetEffect(
                    effect_id="basilisk_stun",
                    target_id="bravo",
                    source="save_check",
                )
            ],
        ),
        ActionUse(
            action_id="lock_on",
            action_type="quick",
            target_id="bravo",
            applied_per_target_effects=[
                AppliedPerTargetEffect(
                    effect_id="basilisk_stun",
                    target_id="bravo",
                    source="triggered_effect",
                )
            ],
        ),
    ]
    validation = validate_combat_scenario(_scenario(actions, template_counters))
    error_codes = {issue.code for issue in validation.issues if issue.severity == "error"}
    warning_codes = {issue.code for issue in validation.issues if issue.severity == "warning"}

    assert "per_target_limit_exceeded" in error_codes
    assert "per_target_limit_unknown" not in warning_codes


def test_per_target_limit_unknown_without_template() -> None:
    actions = [
        ActionUse(
            action_id="invade",
            action_type="quick",
            target_id="bravo",
            applied_per_target_effects=[
                AppliedPerTargetEffect(
                    effect_id="basilisk_stun",
                    target_id="bravo",
                )
            ],
        )
    ]
    validation = validate_combat_scenario(_scenario(actions))
    error_codes = {issue.code for issue in validation.issues if issue.severity == "error"}
    warning_codes = {issue.code for issue in validation.issues if issue.severity == "warning"}

    assert "per_target_limit_exceeded" not in error_codes
    assert "per_target_limit_unknown" in warning_codes
