"""Pilot background types for Lancer TTRPG.

Backgrounds define a pilot's history before becoming a Lancer.
They provide starting triggers for skill checks.

Note: This module contains only mechanical definitions (allowed under
the Lancer Third Party License). No copyrighted flavor text.
"""

from pydantic import BaseModel, Field


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
    
    id: str = Field(..., description="Unique identifier (e.g., 'colonist', 'soldier')")
    name: str = Field(..., description="Display name")
    triggers: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four skill triggers from this background"
    )
    
    model_config = {"frozen": True}


# Standard backgrounds from the Lancer core book
# Note: Only IDs, names, and mechanical triggers - no flavor text
STANDARD_BACKGROUNDS: list[Background] = [
    Background(
        id="colonist",
        name="Colonist",
        triggers=[
            "Survive",
            "Word on the Street",
            "Get a Hold of Something",
            "Fix or Patch",
        ],
    ),
    Background(
        id="soldier",
        name="Soldier",
        triggers=[
            "Apply Fists to Faces",
            "Assault",
            "Threaten",
            "Survive",
        ],
    ),
    Background(
        id="mechanic",
        name="Mechanic",
        triggers=[
            "Fix or Patch",
            "Hack or Fix",
            "Invent or Create",
            "Get a Hold of Something",
        ],
    ),
    Background(
        id="scientist",
        name="Scientist",
        triggers=[
            "Investigate",
            "Read a Situation",
            "Spot",
            "Invent or Create",
        ],
    ),
    Background(
        id="pilot",
        name="Pilot",
        triggers=[
            "Get Somewhere Quickly",
            "Stay Cool Under Fire",
            "Perform a Feat of Dexterity",
            "Act Unseen or Unheard",
        ],
    ),
    Background(
        id="criminal",
        name="Criminal",
        triggers=[
            "Act Unseen or Unheard",
            "Get a Hold of Something",
            "Spot",
            "Word on the Street",
        ],
    ),
    Background(
        id="noble",
        name="Noble",
        triggers=[
            "Charm",
            "Lead or Inspire",
            "Pull Rank",
            "Word on the Street",
        ],
    ),
    Background(
        id="spacer",
        name="Spacer",
        triggers=[
            "Survive",
            "Get Somewhere Quickly",
            "Stay Cool Under Fire",
            "Hack or Fix",
        ],
    ),
]


def get_background(background_id: str) -> Background | None:
    """Look up a standard background by ID."""
    for bg in STANDARD_BACKGROUNDS:
        if bg.id == background_id:
            return bg
    return None
