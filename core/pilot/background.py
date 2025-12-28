"""Pilot background types for Lancer TTRPG.

Backgrounds define a pilot's history before becoming a Lancer.
They provide starting triggers for pilot skill checks.

Note: This module contains only mechanical definitions (allowed under
the Lancer Third Party License). No copyrighted flavor text.
"""

from typing import Literal
from pydantic import BaseModel, Field


class BackgroundInvokeRule(BaseModel):
    """
    Rule for invoking a background during narrative play.

    Invoking a background grants either +1 accuracy or +1 difficulty
    on a pilot skill check, depending on the situation.
    """

    bonus: int = Field(default=1, ge=1, le=1)
    modifier_type: Literal["accuracy", "difficulty", "either"] = "either"
    applies_to: Literal["skill_check"] = "skill_check"

    model_config = {"frozen": True}


class Background(BaseModel):
    """
    A pilot's background - their life before becoming a lancer.
    
    Backgrounds provide 4 triggers that define what situations
    the pilot is particularly skilled at handling based on their
    past experience.
    
    Note: The 'description' field has been intentionally removed
    to avoid including copyrighted flavor text. Users can provide
    their own descriptions if needed.
    """
    
    id: str = Field(..., description="Unique identifier (e.g., 'background_1')")
    name: str = Field(..., description="Display name")
    triggers: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four trigger IDs from this background"
    )
    
    model_config = {"frozen": True}


# Background invoke rule (mechanical only)
BACKGROUND_INVOKE_RULE = BackgroundInvokeRule()


# Example backgrounds (generic labels, mechanical triggers only)
EXAMPLE_BACKGROUNDS: list[Background] = [
    Background(
        id="background_1",
        name="Example Background 1",
        triggers=[
            "survive",
            "word_on_the_streets",
            "get_a_hold_of_something",
            "hack_or_fix",
        ],
    ),
    Background(
        id="background_2",
        name="Example Background 2",
        triggers=[
            "apply_fists_to_faces",
            "assault",
            "threaten",
            "survive",
        ],
    ),
    Background(
        id="background_3",
        name="Example Background 3",
        triggers=[
            "hack_or_fix",
            "invent_or_create",
            "get_a_hold_of_something",
            "blow_something_up",
        ],
    ),
    Background(
        id="background_4",
        name="Example Background 4",
        triggers=[
            "investigate",
            "read_a_situation",
            "spot",
            "invent_or_create",
        ],
    ),
    Background(
        id="background_5",
        name="Example Background 5",
        triggers=[
            "get_somewhere_quickly",
            "stay_cool",
            "show_off",
            "act_unseen_or_unheard",
        ],
    ),
    Background(
        id="background_6",
        name="Example Background 6",
        triggers=[
            "act_unseen_or_unheard",
            "get_a_hold_of_something",
            "spot",
            "word_on_the_streets",
        ],
    ),
    Background(
        id="background_7",
        name="Example Background 7",
        triggers=[
            "charm",
            "lead_or_inspire",
            "pull_rank",
            "word_on_the_streets",
        ],
    ),
    Background(
        id="background_8",
        name="Example Background 8",
        triggers=[
            "survive",
            "get_somewhere_quickly",
            "stay_cool",
            "hack_or_fix",
        ],
    ),
]


def get_background(background_id: str) -> Background | None:
    """Look up an example background by ID."""
    for bg in EXAMPLE_BACKGROUNDS:
        if bg.id == background_id:
            return bg
    return None


# Backwards-compatible alias
STANDARD_BACKGROUNDS = EXAMPLE_BACKGROUNDS
