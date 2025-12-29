from core.pilot.gear import (
    PilotGearItemDefinition,
    PilotGearRules,
    PilotLoadout,
    validate_pilot_loadout,
)
from core.shared.effects import MechanicalEffect, DicePoolEffect, DicePoolSpendOption


def test_pilot_loadout_effect_validation() -> None:
    invalid_effect = MechanicalEffect(
        dice_pools=[
            DicePoolEffect(
                pool_name="test_pool",
                die_size=6,
                max_dice=1,
                spend_options=[
                    DicePoolSpendOption(
                        name="Overcost",
                        dice_cost=2,
                        effect=MechanicalEffect(),
                    )
                ],
            )
        ]
    )
    definitions = {
        "test_gear": PilotGearItemDefinition(
            id="test_gear",
            name="Test Gear",
            category="gear",
            effects=invalid_effect,
        )
    }
    loadout = PilotLoadout(gear=["test_gear"])
    rules = PilotGearRules(clothing_required=False)
    validation = validate_pilot_loadout(loadout, rules=rules, gear_definitions=definitions)
    error_codes = {issue.code for issue in validation.issues if issue.severity == "error"}
    assert "effect_validation_error" in error_codes
