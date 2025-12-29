"""Example pilot builds for schema evaluation."""

from core.pilot.background import Background
from core.pilot.pilot import Pilot
from core.pilot.skill import PilotTrigger, SkillSet, TRIGGER_DEFINITIONS
from core.pilot.talent import EXAMPLE_TALENTS, Talent
from core.pilot.license import License
from core.pilot.core_bonus import CoreBonus
from core.pilot.validation import ProgressionValidation
from core.mech.examples import (
    build_oda_ll0_mech_example,
    build_oda_ll3_mech_example,
)
from core.mech.build import compute_mech_stats
from core.mech.validation import MechBuildValidation, validate_mech_build
from core.mech.compendium import SYSTEM_DEFINITIONS_BY_ID, WEAPON_DEFINITIONS_BY_ID
from core.pilot.gear import PilotLoadout
from core.pilot.mission import (
    DowntimeActionUse,
    DowntimePlan,
    ReserveEntry,
    validate_downtime_plan,
)


def build_example_pilot_ll0() -> Pilot:
    """Build a simple LL0 pilot that matches the progression table."""
    triggers = [
        PilotTrigger(trigger_id=TRIGGER_DEFINITIONS[0].id, rank=2),
        PilotTrigger(trigger_id=TRIGGER_DEFINITIONS[1].id, rank=2),
        PilotTrigger(trigger_id=TRIGGER_DEFINITIONS[2].id, rank=2),
        PilotTrigger(trigger_id=TRIGGER_DEFINITIONS[3].id, rank=2),
    ]
    talents = [
        Talent(talent_id=EXAMPLE_TALENTS[0].id, rank=1),
        Talent(talent_id=EXAMPLE_TALENTS[1].id, rank=1),
        Talent(talent_id=EXAMPLE_TALENTS[2].id, rank=1),
    ]

    return Pilot(
        callsign="EXAMPLE",
        name="",
        background=Background(
            id="background_example",
            name="Example Background",
            triggers=[
                "survive",
                "read_a_situation",
                "spot",
                "take_someone_out",
            ],
        ),
        level=0,
        skills=SkillSet(hull=2),
        triggers=triggers,
        talents=talents,
    )


def evaluate_example_pilot_ll0() -> bool:
    """Return True if the example pilot validates cleanly."""
    pilot = build_example_pilot_ll0()
    return pilot.validate_progression().valid


def build_oda_ll0_pilot() -> Pilot:
    """Build Oda's LL0 pilot from the example."""
    return Pilot(
        callsign="ODA",
        name="",
        background=Background(
            id="background_example",
            name="Example Background",
            triggers=[
                "read_a_situation",
                "spot",
                "take_someone_out",
                "survive",
            ],
        ),
        level=0,
        skills=SkillSet(hull=2, agility=0, systems=0, engineering=0),
        triggers=[
            PilotTrigger(trigger_id="read_a_situation", rank=2),
            PilotTrigger(trigger_id="spot", rank=2),
            PilotTrigger(trigger_id="take_someone_out", rank=2),
            PilotTrigger(trigger_id="survive", rank=2),
        ],
        talents=[
            Talent(talent_id="crack_shot", rank=1),
            Talent(talent_id="infiltrator", rank=1),
            Talent(talent_id="leader", rank=1),
        ],
        pilot_gear=PilotLoadout(
            clothing="flight_suit",
            armor="light_hardsuit",
            weapons=["signature_weapon_combat"],
            gear=["extra_rations", "cooking_gear"],
        ),
    )


def build_oda_ll3_pilot() -> Pilot:
    """Build Oda's LL3 pilot from the example."""
    return Pilot(
        callsign="ODA",
        name="",
        background=Background(
            id="background_example",
            name="Example Background",
            triggers=[
                "read_a_situation",
                "spot",
                "take_someone_out",
                "survive",
            ],
        ),
        level=3,
        skills=SkillSet(hull=5, agility=0, systems=0, engineering=0),
        triggers=[
            PilotTrigger(trigger_id="read_a_situation", rank=6),
            PilotTrigger(trigger_id="spot", rank=2),
            PilotTrigger(trigger_id="take_someone_out", rank=4),
            PilotTrigger(trigger_id="survive", rank=2),
        ],
        talents=[
            Talent(talent_id="crack_shot", rank=2),
            Talent(talent_id="infiltrator", rank=1),
            Talent(talent_id="leader", rank=3),
        ],
        licenses=[
            License(license_id="raleigh", rank=3),
        ],
        core_bonuses=[
            CoreBonus(core_bonus_id="ipsn_reinforced_frame"),
        ],
        pilot_gear=PilotLoadout(
            clothing="flight_suit",
            armor="light_hardsuit",
            weapons=["signature_weapon_combat"],
            gear=["extra_rations", "cooking_gear"],
        ),
    )


def evaluate_oda_ll0_example() -> tuple[ProgressionValidation, MechBuildValidation, list[str]]:
    """Validate Oda's LL0 pilot and mech build, returning mismatches."""
    pilot = build_oda_ll0_pilot()
    pilot_validation = pilot.validate_progression()
    frame, build, skills, grit, effects = build_oda_ll0_mech_example()
    mech_validation = validate_mech_build(
        frame,
        build,
        skills,
        grit,
        licenses=pilot.licenses,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        bonus_effects=effects,
    )
    stats = compute_mech_stats(frame, skills, grit, bonus_effects=effects)
    expected = {
        "hp": 16,
        "evasion": 8,
        "speed": 4,
        "heat_cap": 6,
        "sensor_range": 10,
        "armor": 0,
        "e_defense": 8,
        "size": "size_1",
        "repair_cap": 6,
        "tech_attack": 0,
        "system_points": 6,
    }
    mismatches = _compare_mech_stats(stats.model_dump(), expected)
    return pilot_validation, mech_validation, mismatches


def evaluate_oda_ll3_example() -> tuple[ProgressionValidation, MechBuildValidation, list[str]]:
    """Validate Oda's LL3 pilot and mech build, returning mismatches."""
    pilot = build_oda_ll3_pilot()
    pilot_validation = pilot.validate_progression()
    frame, build, skills, grit, effects = build_oda_ll3_mech_example()
    mech_validation = validate_mech_build(
        frame,
        build,
        skills,
        grit,
        licenses=pilot.licenses,
        weapon_definitions=WEAPON_DEFINITIONS_BY_ID,
        system_definitions=SYSTEM_DEFINITIONS_BY_ID,
        bonus_effects=effects,
    )
    stats = compute_mech_stats(frame, skills, grit, bonus_effects=effects)
    expected = {
        "hp": 27,
        "evasion": 8,
        "speed": 4,
        "heat_cap": 5,
        "sensor_range": 10,
        "armor": 1,
        "e_defense": 7,
        "size": "size_1",
        "repair_cap": 7,
        "tech_attack": -1,
        "system_points": 7,
    }
    mismatches = _compare_mech_stats(stats.model_dump(), expected)
    return pilot_validation, mech_validation, mismatches


def build_example_downtime_plan() -> DowntimePlan:
    """Build a small downtime plan with reserve tracking."""
    return DowntimePlan(
        pilot_id="example_pilot",
        actions=[
            DowntimeActionUse(
                action_id="get_a_hold_of_something",
                outcome="reserve",
                reserve=ReserveEntry(
                    id="reserve_cache",
                    name="Reserve Cache",
                    uses_remaining=1,
                    shared=True,
                    source="downtime_action",
                ),
            ),
            DowntimeActionUse(
                action_id="get_a_clue",
                outcome="info",
                reserve=None,
            ),
        ],
    )


def evaluate_example_downtime_plan() -> bool:
    """Return True if the downtime plan validates cleanly."""
    plan = build_example_downtime_plan()
    return validate_downtime_plan(plan).valid


def _compare_mech_stats(actual: dict[str, object], expected: dict[str, object]) -> list[str]:
    mismatches: list[str] = []
    for key, value in expected.items():
        if actual.get(key) != value:
            mismatches.append(f"{key}: expected {value}, got {actual.get(key)}")
    return mismatches
