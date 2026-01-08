"""Combat validation module for mech combat.

Submodules:
    - combat_validation: Main entry point with public API
    - geometry_validation: Cover, LOS, movement helpers
    - action_validation: Action timing, cooldowns, overcharge
    - area_validation: Area attack geometry

Public API:
    - CombatValidationIssue: Validation issue class
    - CombatValidation: Validation result class
    - validate_combat_scenario: Validate full combat scenario
    - validate_deployment: Validate deployable placement
    - validate_mine_detection: Validate mine detection attempt
    - validate_mine_disarm: Validate mine disarm attempt
"""

from core.mech.validation.combat_validation import (
    CombatValidationIssue,
    CombatValidation,
    validate_combat_scenario,
    validate_deployment,
    validate_mine_detection,
    validate_mine_disarm,
)
from core.mech.build_validation import MechBuildIssue

__all__ = [
    "CombatValidationIssue",
    "CombatValidation",
    "validate_combat_scenario",
    "validate_deployment",
    "validate_mine_detection",
    "validate_mine_disarm",
    "MechBuildIssue",
]
