"""Invisibility resolution helpers for Lancer combat.

This module provides type-safe helpers for invisibility mechanics including:
- Detection: Contested checks to detect invisible targets
- Miss Chance: 50% miss chance handling for attacks on invisible targets
- Breaking: Tracking when invisibility is lost
- Heat Patterns: Detection difficulty based on target's heat signature

Invisibility Rules (per PR2 4073-4076):
- All attacks against invisible targets have a 50% chance to miss outright
- Invisibility is checked before rolling the attack
- Invisible characters are detectable by heat patterns and visual artifacts
- Invisible mechs can always hide, even without cover
- Invisibility can be broken by damage, actions, or abilities
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.dice import roll_dice


InvisibilityBreakTrigger = Literal[
    "takes_damage",
    "attacks",
    "cast_tech_action",
    "takes_reaction",
    "moves",
    "end_of_turn",
    "targeted_by_scan",
    "sensor_jammed",
    "area_denial",
]


class InvisibilityBreakCheck(FrozenModel):
    """Result of checking if invisibility should be broken."""

    invisibility_broken: bool
    break_reason: str = ""
    break_trigger: InvisibilityBreakTrigger | None = None


class InvisibilityDetectionAttempt(FrozenModel):
    """Input for detecting an invisible target."""

    detector_systems_bonus: int = 0
    detector_has_sensor_enhancement: bool = False
    detector_is_proximate: bool = False
    detector_proximity_bonus: int = 2
    target_systems_bonus: int = 0
    target_agility_bonus: int = 0
    target_heat: int = 0
    target_heat_capacity: int = 10
    target_has_heat_masking: bool = False
    target_has_stealth_system: bool = False
    target_in_sensor_range: bool = False
    is_pilot_detection: bool = False
    pilot_skill_bonus: int = 0


class InvisibilityDetectionResult(FrozenModel):
    """Result of attempting to detect an invisible target."""

    detection_successful: bool
    detection_roll: int | None = None
    detector_total: int | None = None
    target_roll: int | None = None
    target_total: int | None = None
    heat_signature_bonus: int = 0
    difficulty_modifier: str = ""
    reason: str = ""


class InvisibilityMissChanceResult(FrozenModel):
    """Result of invisibility miss chance check."""

    miss_applies: bool
    miss_roll: int | None = None
    miss_result: Literal["miss", "hit"] | None = None
    reason: str = ""


HeatSignatureLevel = Literal["cold", "low", "moderate", "high", "critical"]


class HeatSignatureResult(FrozenModel):
    """Result of heat signature calculation."""

    signature_level: HeatSignatureLevel
    detection_difficulty_modifier: int
    is_detectable: bool
    description: str


class InvisibilityBreakCondition(FrozenModel):
    """Defines when invisibility breaks for a specific source."""

    source_name: str
    breaks_on_damage: bool = False
    breaks_on_attack: bool = False
    breaks_on_tech_action: bool = False
    breaks_on_reaction: bool = False
    breaks_on_move: bool = False
    breaks_end_of_turn: bool = False
    breaks_on_scan: bool = False
    duration_turns: int | None = None


INVISIBILITY_BREAK_CONDITIONS: dict[str, InvisibilityBreakCondition] = {
    "stealth_hardsuit": InvisibilityBreakCondition(
        source_name="Stealth Hardsuit",
        breaks_on_damage=True,
        breaks_on_attack=True,
        breaks_on_tech_action=True,
        breaks_on_reaction=True,
        breaks_on_move=True,
    ),
    "integrated_cloak": InvisibilityBreakCondition(
        source_name="Integrated Cloak",
        breaks_on_damage=False,
        breaks_on_attack=True,
        breaks_on_tech_action=True,
        breaks_on_reaction=True,
        breaks_on_move=True,
        breaks_end_of_turn=True,
    ),
    "flash_cloak": InvisibilityBreakCondition(
        source_name="Flash Cloak",
        breaks_on_damage=False,
        breaks_on_attack=True,
        breaks_on_tech_action=True,
        breaks_on_reaction=True,
        breaks_on_move=True,
    ),
    "harlequin_cloak": InvisibilityBreakCondition(
        source_name="Harlequin Cloak",
        breaks_on_damage=False,
        breaks_on_attack=False,
        breaks_on_tech_action=False,
        breaks_on_reaction=False,
        breaks_on_move=False,
        breaks_end_of_turn=True,
    ),
    "mirage_invisibility": InvisibilityBreakCondition(
        source_name="Mirage Invisibility",
        breaks_on_damage=False,
        breaks_on_attack=True,
        breaks_on_tech_action=True,
        breaks_on_reaction=True,
        breaks_on_move=True,
    ),
    "spectre_permanent": InvisibilityBreakCondition(
        source_name="Spectre Permanent Invisibility",
        breaks_on_damage=False,
        breaks_on_attack=False,
        breaks_on_tech_action=False,
        breaks_on_reaction=False,
        breaks_on_move=False,
        breaks_end_of_turn=False,
    ),
    "operative_invisibility": InvisibilityBreakCondition(
        source_name="Operative Invisibility",
        breaks_on_damage=False,
        breaks_on_attack=True,
        breaks_on_tech_action=True,
        breaks_on_reaction=True,
        breaks_on_move=True,
        breaks_end_of_turn=True,
    ),
    "scout_invisibility": InvisibilityBreakCondition(
        source_name="Scout Invisibility",
        breaks_on_damage=False,
        breaks_on_attack=True,
        breaks_on_tech_action=True,
        breaks_on_reaction=True,
        breaks_on_move=True,
    ),
    "witch_invisibility": InvisibilityBreakCondition(
        source_name="Witch Invisibility",
        breaks_on_damage=False,
        breaks_on_attack=False,
        breaks_on_tech_action=False,
        breaks_on_reaction=False,
        breaks_on_move=False,
        breaks_end_of_turn=True,
    ),
    "ultra_permanent": InvisibilityBreakCondition(
        source_name="Ultra Permanent Invisibility",
        breaks_on_damage=False,
        breaks_on_attack=False,
        breaks_on_tech_action=False,
        breaks_on_reaction=False,
        breaks_on_move=False,
        breaks_end_of_turn=False,
    ),
}


def detect_invisible_target(
    attempt: InvisibilityDetectionAttempt,
    roll_result: int | None = None,
    target_roll_result: int | None = None,
) -> InvisibilityDetectionResult:
    """Attempt to detect an invisible target.

    Per PR2 4073-4074:
    "An invisible character is detectable by heat patterns and some visual
    artifacts, but extremely hard to target - all attacks of any kind have a
    flat 50% chance to miss outright (roll a dice or flip a coin)"

    Detection is a contested check. For mech detection:
    - Attacker makes a Systems check
    - Invisible target makes a Systems or Agility check (whichever is better)
    - Higher total succeeds

    For pilot detection:
    - Pilot makes a skill check (flat 1d20 vs 10)
    - If successful, target is detected

    Args:
        attempt: InvisibilityDetectionAttempt with all required parameters
        roll_result: Optional fixed roll for testing (1d20 for detector)
        target_roll_result: Optional fixed roll for testing (1d20 for target)

    Returns:
        InvisibilityDetectionResult with detection outcome
    """
    if attempt.is_pilot_detection:
        if roll_result is None:
            roll_result = roll_dice("1d20")
        else:
            roll_result = max(1, min(20, roll_result))

        total = roll_result + attempt.pilot_skill_bonus
        success = total >= 10

        return InvisibilityDetectionResult(
            detection_successful=success,
            detection_roll=roll_result,
            detector_total=total,
            reason=f"Pilot detection: {roll_result}+{attempt.pilot_skill_bonus} vs DC 10",
        )

    heat_result = calculate_heat_signature(
        current_heat=attempt.target_heat,
        heat_capacity=attempt.target_heat_capacity,
        has_heat_masking=attempt.target_has_heat_masking,
    )

    detector_bonus = attempt.detector_systems_bonus
    if attempt.detector_has_sensor_enhancement:
        detector_bonus += 2
    if attempt.detector_is_proximate:
        detector_bonus += attempt.detector_proximity_bonus

    detector_bonus += heat_result.detection_difficulty_modifier

    if roll_result is None:
        detector_roll = roll_dice("1d20")
    else:
        detector_roll = max(1, min(20, roll_result))

    if target_roll_result is None:
        target_roll = roll_dice("1d20")
    else:
        target_roll = max(1, min(20, target_roll_result))

    target_bonus = max(attempt.target_systems_bonus, attempt.target_agility_bonus)

    if attempt.target_has_stealth_system:
        target_bonus += 2

    detector_total = detector_roll + detector_bonus
    target_total = target_roll + target_bonus

    success = detector_total >= target_total

    difficulty_desc = ""
    if heat_result.signature_level == "high":
        difficulty_desc = " (target has high heat signature)"
    elif heat_result.signature_level == "critical":
        difficulty_desc = " (target has critical heat signature)"
    elif heat_result.signature_level == "cold":
        difficulty_desc = " (target is cold, hard to detect)"

    return InvisibilityDetectionResult(
        detection_successful=success,
        detection_roll=detector_roll,
        detector_total=detector_total,
        target_roll=target_roll,
        target_total=target_total,
        heat_signature_bonus=heat_result.detection_difficulty_modifier,
        difficulty_modifier=difficulty_desc,
        reason=f"Detection: {detector_roll}+{detector_bonus} vs {target_roll}+{target_bonus}{difficulty_desc}",
    )


def resolve_invisibility_miss_chance(
    target_is_invisible: bool,
    attacker_ignores_invisibility: bool = False,
    roll_result: int | None = None,
) -> InvisibilityMissChanceResult:
    """Resolve the 50% miss chance for attacks on invisible targets.

    Per PR2 4075:
    "All attacks of any kind have a flat 50% chance to miss outright (roll
    a dice or flip a coin) - checked before rolling"

    This function implements the coin flip/1d2 roll to determine if the
    attack misses due to invisibility.

    Args:
        target_is_invisible: Whether the target has invisibility status
        attacker_ignores_invisibility: Whether attacker's abilities negate invisibility
        roll_result: Optional fixed roll for testing (1d2: 1=miss, 2=hit)

    Returns:
        InvisibilityMissChanceResult with miss chance outcome
    """
    if not target_is_invisible:
        return InvisibilityMissChanceResult(
            miss_applies=False,
            reason="Target is not invisible",
        )

    if attacker_ignores_invisibility:
        return InvisibilityMissChanceResult(
            miss_applies=False,
            reason="Attacker ignores invisibility",
        )

    if roll_result is None:
        roll_result = roll_dice("1d2")
    else:
        roll_result = max(1, min(2, roll_result))

    miss_result: Literal["miss", "hit"] = "miss" if roll_result == 1 else "hit"
    miss_applies = miss_result == "miss"

    if miss_applies:
        return InvisibilityMissChanceResult(
            miss_applies=True,
            miss_roll=roll_result,
            miss_result=miss_result,
            reason=f"Miss chance applies (1d2={roll_result}), attack automatically misses",
        )

    return InvisibilityMissChanceResult(
        miss_applies=False,
        miss_roll=roll_result,
        miss_result=miss_result,
        reason=f"Miss chance does not apply (1d2={roll_result}), attack proceeds",
    )


def check_invisibility_broken(
    invisibility_source: str | None,
    current_conditions: list[str],
    took_damage: bool = False,
    took_attack_action: bool = False,
    took_tech_action: bool = False,
    took_reaction: bool = False,
    took_move_action: bool = False,
    is_end_of_turn: bool = False,
) -> InvisibilityBreakCheck:
    """Check if invisibility should be broken based on actions taken.

    Per various system descriptions:
    - Stealth Hardsuit: breaks if you take damage
    - Integrated Cloak: breaks on move/reaction, end of turn
    - Flash Cloak: breaks on move/reaction
    - Most invisibility: breaks on attack/tech action

    Args:
        invisibility_source: Source of invisibility (key into INVISIBILITY_BREAK_CONDITIONS)
        current_conditions: Current conditions on the character
        took_damage: Whether character took damage this turn
        took_attack_action: Whether character attacked this turn
        took_tech_action: Whether character cast a tech action this turn
        took_reaction: Whether character took a reaction this turn
        took_move_action: Whether character moved this turn
        is_end_of_turn: Whether this is end of turn check

    Returns:
        InvisibilityBreakCheck with break outcome
    """
    if (
        invisibility_source is None
        or invisibility_source not in INVISIBILITY_BREAK_CONDITIONS
    ):
        if "invisible" not in current_conditions:
            return InvisibilityBreakCheck(
                invisibility_broken=False,
                break_reason="Not invisible",
                break_trigger=None,
            )
        return InvisibilityBreakCheck(
            invisibility_broken=False,
            break_reason="Unknown invisibility source, no break triggers fire",
            break_trigger=None,
        )

    break_conditions = INVISIBILITY_BREAK_CONDITIONS[invisibility_source]

    if took_damage and break_conditions.breaks_on_damage:
        return InvisibilityBreakCheck(
            invisibility_broken=True,
            break_reason=f"Took damage, {break_conditions.source_name} invisibility breaks",
            break_trigger="takes_damage",
        )

    if took_attack_action and break_conditions.breaks_on_attack:
        return InvisibilityBreakCheck(
            invisibility_broken=True,
            break_reason=f"Attacked, {break_conditions.source_name} invisibility breaks",
            break_trigger="attacks",
        )

    if took_tech_action and break_conditions.breaks_on_tech_action:
        return InvisibilityBreakCheck(
            invisibility_broken=True,
            break_reason=f"Casted tech action, {break_conditions.source_name} invisibility breaks",
            break_trigger="cast_tech_action",
        )

    if took_reaction and break_conditions.breaks_on_reaction:
        return InvisibilityBreakCheck(
            invisibility_broken=True,
            break_reason=f"Took reaction, {break_conditions.source_name} invisibility breaks",
            break_trigger="takes_reaction",
        )

    if took_move_action and break_conditions.breaks_on_move:
        return InvisibilityBreakCheck(
            invisibility_broken=True,
            break_reason=f"Moved, {break_conditions.source_name} invisibility breaks",
            break_trigger="moves",
        )

    if is_end_of_turn and break_conditions.breaks_end_of_turn:
        return InvisibilityBreakCheck(
            invisibility_broken=True,
            break_reason=f"End of turn, {break_conditions.source_name} invisibility ends",
            break_trigger="end_of_turn",
        )

    return InvisibilityBreakCheck(
        invisibility_broken=False,
        break_reason=f"{break_conditions.source_name} invisibility maintained",
        break_trigger=None,
    )


def calculate_heat_signature(
    current_heat: int,
    heat_capacity: int,
    has_heat_masking: bool = False,
) -> HeatSignatureResult:
    """Calculate the heat signature level for detection purposes.

    Per PR2 4073:
    "An invisible character is detectable by heat patterns"

    Heat signature affects detection difficulty:
    - Low heat: harder to detect (negative modifier)
    - High heat: easier to detect (positive modifier)
    - Heat masking systems provide bonuses

    Args:
        current_heat: Current heat level on the mech
        heat_capacity: Maximum heat capacity
        has_heat_masking: Whether mech has heat masking systems

    Returns:
        HeatSignatureResult with signature level and detection modifier
    """
    heat_ratio = current_heat / heat_capacity if heat_capacity > 0 else 0

    base_bonus = 0
    if has_heat_masking:
        base_bonus = -2

    if heat_ratio <= 0.1:
        signature_level: HeatSignatureLevel = "cold"
        heat_bonus = -2
        description = "Cold mech - minimal heat signature"
    elif heat_ratio <= 0.3:
        signature_level = "low"
        heat_bonus = -1
        description = "Low heat - minor heat signature"
    elif heat_ratio <= 0.6:
        signature_level = "moderate"
        heat_bonus = 0
        description = "Moderate heat - normal detection difficulty"
    elif heat_ratio <= 0.85:
        signature_level = "high"
        heat_bonus = 2
        description = "High heat - clear thermal signature"
    else:
        signature_level = "critical"
        heat_bonus = 4
        description = "Critical heat - very easy to detect"

    total_bonus = base_bonus + heat_bonus

    return HeatSignatureResult(
        signature_level=signature_level,
        detection_difficulty_modifier=total_bonus,
        is_detectable=True,
        description=description,
    )


def can_always_hide_while_invisible(
    is_invisible: bool,
    is_engaged: bool,
) -> tuple[bool, str]:
    """Check if character can hide while invisible.

    Per PR2 4076:
    "Additionally, an invisible mech can always hide, even without cover."

    And PR2 4226:
    "You can't attempt to hide if you're engaged with another character,
    even if you're invisible (they can still see you)."

    Args:
        is_invisible: Whether the character is invisible
        is_engaged: Whether the character is engaged

    Returns:
        Tuple of (can_hide, reason)
    """
    if not is_invisible:
        return False, "Not invisible - cover required for hiding"

    if is_engaged:
        return (
            False,
            "Engaged with another character - cannot hide even while invisible",
        )

    return True, "Invisible and not engaged - can always hide"


def get_invisibility_miss_chance_description() -> str:
    """Get a description of invisibility miss chance rules.

    Returns:
        Human-readable description of invisibility targeting rules
    """
    return (
        "Attacks against invisible targets have a 50% chance to miss outright. "
        "The miss check is made before the attack roll (flip a coin or roll 1d2: "
        "1=miss, 2=hit). If the miss applies, the attack fails automatically with "
        "no damage or heat dealt. Some abilities and systems can ignore invisibility."
    )
