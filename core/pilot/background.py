"""Pilot background types for Lancer TTRPG.

Backgrounds define a pilot's history before becoming a Lancer.
They provide starting triggers for pilot skill checks.

Note: This module contains only mechanical definitions (allowed under
the Lancer Third Party License). No copyrighted flavor text.
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel


class BackgroundInvokeRule(FrozenModel):
    """
    Rule for invoking a background during narrative play.

    Invoking a background grants either +1 accuracy or +1 difficulty
    on a pilot skill check, depending on the situation.
    """

    bonus: int = Field(default=1, ge=1, le=1)
    modifier_type: Literal["accuracy", "difficulty", "either"] = "either"
    applies_to: Literal["skill_check"] = "skill_check"


class Background(FrozenModel):
    """
    A pilot's background - their life before becoming a lancer.

    Backgrounds provide 4 triggers that define what situations
    the pilot is particularly skilled at handling based on their
    past experience.

    Note: The 'description' field has been intentionally removed
    to avoid including copyrighted flavor text. Users can provide
    their own descriptions if needed.
    """

    id: str = Field(
        ...,
        description="Unique identifier (e.g., 'background_nhp_specialist')",
    )
    name: str = Field(..., description="Display name")
    triggers: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four trigger IDs from this background",
    )


# Background invoke rule (mechanical only)
BACKGROUND_INVOKE_RULE = BackgroundInvokeRule()


# Pilot backgrounds (mechanical triggers only)
PILOT_BACKGROUNDS: list[Background] = [
    Background(
        id="background_nhp_specialist",
        name="NHP Specialist",
        triggers=["stay_cool", "read_a_situation", "invent_or_create", "investigate"],
    ),
    Background(
        id="background_celebrity",
        name="Celebrity",
        triggers=["charm", "pull_rank", "lead_or_inspire", "threaten"],
    ),
    Background(
        id="background_colonist",
        name="Colonist",
        triggers=["word_on_the_streets", "spot", "survive", "patch"],
    ),
    Background(
        id="background_criminal",
        name="Criminal",
        triggers=[
            "threaten",
            "apply_fists_to_faces",
            "word_on_the_streets",
            "take_control",
        ],
    ),
    Background(
        id="background_hacker",
        name="Hacker",
        triggers=[
            "act_unseen_or_unheard",
            "get_a_hold_of_something",
            "hack_or_fix",
            "invent_or_create",
        ],
    ),
    Background(
        id="background_far_field_team",
        name="Far Field Team",
        triggers=["survive", "investigate", "spot", "charm"],
    ),
    Background(
        id="background_mechanic",
        name="Mechanic",
        triggers=[
            "hack_or_fix",
            "get_somewhere_quickly",
            "get_a_hold_of_something",
            "blow_something_up",
        ],
    ),
    Background(
        id="background_medic",
        name="Medic",
        triggers=["patch", "assault", "read_a_situation", "stay_cool"],
    ),
    Background(
        id="background_mercenary",
        name="Mercenary",
        triggers=[
            "threaten",
            "blow_something_up",
            "take_control",
            "apply_fists_to_faces",
        ],
    ),
    Background(
        id="background_noble",
        name="Noble",
        triggers=["pull_rank", "lead_or_inspire", "read_a_situation", "show_off"],
    ),
    Background(
        id="background_outlaw",
        name="Outlaw",
        triggers=["show_off", "take_someone_out", "charm", "survive"],
    ),
    Background(
        id="background_penal_colonist",
        name="Penal Colonist",
        triggers=[
            "survive",
            "apply_fists_to_faces",
            "word_on_the_streets",
            "spot",
        ],
    ),
    Background(
        id="background_priest",
        name="Priest",
        triggers=["read_a_situation", "stay_cool", "take_control", "lead_or_inspire"],
    ),
    Background(
        id="background_scientist",
        name="Scientist",
        triggers=[
            "investigate",
            "invent_or_create",
            "get_a_hold_of_something",
            "blow_something_up",
        ],
    ),
    Background(
        id="background_soldier",
        name="Soldier",
        triggers=["assault", "blow_something_up", "pull_rank", "take_control"],
    ),
    Background(
        id="background_spaceborn",
        name="Spaceborn",
        triggers=["survive", "hack_or_fix", "get_somewhere_quickly", "stay_cool"],
    ),
    Background(
        id="background_spec_ops",
        name="Spec Ops",
        triggers=["act_unseen_or_unheard", "take_someone_out", "spot", "stay_cool"],
    ),
    Background(
        id="background_super_soldier",
        name="Super Soldier",
        triggers=[
            "apply_fists_to_faces",
            "get_somewhere_quickly",
            "assault",
            "read_a_situation",
        ],
    ),
    Background(
        id="background_starship_pilot",
        name="Starship Pilot",
        triggers=[
            "get_somewhere_quickly",
            "show_off",
            "get_a_hold_of_something",
            "hack_or_fix",
        ],
    ),
    Background(
        id="background_worker",
        name="Worker",
        triggers=[
            "word_on_the_streets",
            "stay_cool",
            "lead_or_inspire",
            "invent_or_create",
        ],
    ),
]


def get_background(background_id: str) -> Background | None:
    """Look up a background by ID."""
    for bg in PILOT_BACKGROUNDS:
        if bg.id == background_id:
            return bg
    return None
