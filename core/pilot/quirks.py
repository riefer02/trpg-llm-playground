"""Quirk system for Lancer flash cloning.

Implements the 20-quirk table per PR2 4828-4960:
- Quirks are narrative hooks with no gameplay effects
- Quirks can be physical or mental in nature
- Quirks always complicate the character's situation
- Optional: Quirks can apply to Down and Out survivors as trauma
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.pilot.clone_state import Quirk, QuirkType


QUIRK_TABLE: list[Quirk] = [
    Quirk(
        roll=1,
        name="Cybernetic Replacement",
        description=(
            "Part (or all) of your body was too damaged to be cloned perfectly and a "
            "significant percentage of your clone's body has been replaced with cybernetics. "
            "These are high quality prostheses, and are not visibly synthetic to a casual "
            "observer. The extent of the damage is unknown to you."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=2,
        name="Visible Cybernetic",
        description=(
            "The process required your clone be fitted with a visible cybernetic augment, "
            "such as an arm, leg, eyes, or the like. It is conspicuous and often attracts "
            "unwanted attention."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=3,
        name="Body Swap",
        description=(
            "By accident or malintent, you have been cloned into a different client's body. "
            "You may discover in time that the body might be that of a notorious, famous, or "
            "infamous individual, with enemies and allies who thought they were dead (or they "
            "might encounter the 'original' person you have been cloned from)."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=4,
        name="Vat-Grown Scar",
        description=(
            "You are cloned or revived with a nasty, disfiguring scar or a hideous appearance "
            "that clearly marks you as vat-grown."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=5,
        name="Administrative Mishap",
        description=(
            "Administrative mishaps lead to complete and drastic change in appearance in your "
            "new body."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=6,
        name="Extra Limb",
        description=(
            "An extra, withered limb grows out of your fascimile's chest shortly after your "
            "cloning. It sometimes moves on its own."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=7,
        name="Conspicuous Barcode",
        description=(
            "A conspicuous barcode is now printed on your facsimile body. The barcode has "
            "meaning to powerful organizations, but you are not privy to its meaning - yet."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=8,
        name="Script Under Skin",
        description=(
            "Under certain light conditions, it is possible to read a script or inscription "
            "printed just under your skin. The script is all over your body and contains a "
            "scientific formula, a map, or other information contested by powerful organizations "
            "or entities."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=9,
        name="Radiation Susceptibility",
        description=(
            "Your new body is incredibly susceptible to solar radiation, viruses and bacteria, "
            "or some other widespread 'normal' environmental phenomenon. To operate outside of "
            "a safe environment, you must wear an environmental suit. The cockpit of your mech "
            "is considered a safe environment, as well as your personal quarters. Other rooms "
            "can be made safe as a downtime action, but require time to be converted and a "
            "decontamination process to enter."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=10,
        name="Non-Human DNA",
        description=(
            "DNA from a non-human or possible xenobiological source was used in your "
            "resuscitation. Your revivers will not tell you the exact details or what effects "
            "it will have on you long term, and treat you more as a science experiment. You now "
            "have a useful, visible (though able to be hidden) cosmetic variation."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=11,
        name="Death Dreams",
        description=(
            "You are stricken with persistent dreams, visions, and images of your death in vivid "
            "detail whenever you try and sleep or rest. You know they are all real, but cannot "
            "reconcile the existential gulf between what your previous 'you' experienced, and "
            "your new subjectivity."
        ),
        quirk_type="mental"
    ),
    Quirk(
        roll=12,
        name="Digital Homunculus",
        description=(
            "You are replaced by a digital 'homunculus', an electronic imprint and reconstruction "
            "of your personality that occupies a subaltern, a kind of robotic shell. While "
            "technically not conscious, it's essentially 'you'."
        ),
        quirk_type="mental"
    ),
    Quirk(
        roll=13,
        name="Shadow Self",
        description=(
            "You are plagued by the constant understanding or belief that the 'real' you is "
            "actually dead, and you are merely a shadow aping a dead person, implanted with the "
            "memories of someone else. You cannot establish the difference between the 'you' "
            "that died and the 'you' that exists now."
        ),
        quirk_type="mental"
    ),
    Quirk(
        roll=14,
        name="Implanted Memories",
        description=(
            "Your clone is implanted with the residual memories of an entirely different and "
            "powerful or influential person. This reveals very dangerous and potentially unwanted "
            "information to you that is contested or sought after by powerful entities."
        ),
        quirk_type="mental"
    ),
    Quirk(
        roll=15,
        name="Tabula Rasa",
        description=(
            "The process goes awry and you are revived tabula rasa. In desperation, the techs "
            "dump a stock personality construction into you. Change your background (adjust your "
            "triggers accordingly, and erase all your invocations)."
        ),
        quirk_type="mental"
    ),
    Quirk(
        roll=16,
        name="Shortened Lifespan",
        description=(
            "The process of revival is not without complications. Your natural lifespan is "
            "dramatically shortened, and you know you will have to undergo another flash-cloning "
            "in the near future."
        ),
        quirk_type="physical"
    ),
    Quirk(
        roll=17,
        name="Mental Contact",
        description=(
            "Something changed you, and you have persistent and intrusive mental contact with "
            "another entity or entities. It could be human or non-human in nature."
        ),
        quirk_type="mental"
    ),
    Quirk(
        roll=18,
        name="Future Flashes",
        description=(
            "You often are struck with searing headaches during which you see brief flashes of "
            "what you are pretty sure is the future. Sometimes it comes to pass, sometimes it "
            "doesn't."
        ),
        quirk_type="mental"
    ),
    Quirk(
        roll=19,
        name="Mental Trigger",
        description=(
            "Knowingly or unknowingly, you are implanted with a mental trigger that when heard "
            "or activated, causes you to go into a receptive state, either following a "
            "pre-programmed course of action (kill, lie, etc) or to listen to and follow exactly "
            "the commands of the person who activated you. These commands must be simple, and "
            "the person who gives them (PC or NPC) is determined by the GM. You might be able to "
            "overcome this effect with time."
        ),
        quirk_type="mental"
    ),
    Quirk(
        roll=20,
        name="Complete Amnesia",
        description=(
            "You are brought back with complete amnesia of the time before you were re-born, "
            "causing a 'tabula rasa' situation in which you must be re-trained and interpellated, "
            "a costly process. Your triggers completely reset. Re-assign them as if you were "
            "level 0 and just leveled up to your current level, and you may re-write some "
            "incidental facts of your backstory."
        ),
        quirk_type="mental"
    ),
]


def get_quirk_by_roll(roll: int) -> Quirk | None:
    """Get a quirk by its 1d20 roll value.

    Args:
        roll: 1d20 roll result (1-20)

    Returns:
        Quirk if found, None if invalid roll
    """
    for quirk in QUIRK_TABLE:
        if quirk.roll == roll:
            return quirk
    return None


def get_quirks_by_type(quirk_type: QuirkType) -> list[Quirk]:
    """Get all quirks of a specific type.

    Args:
        quirk_type: "physical" or "mental"

    Returns:
        List of quirks of the specified type
    """
    return [q for q in QUIRK_TABLE if q.quirk_type == quirk_type]


def get_physical_quirks() -> list[Quirk]:
    """Get all physical quirks."""
    return get_quirks_by_type("physical")


def get_mental_quirks() -> list[Quirk]:
    """Get all mental quirks."""
    return get_quirks_by_type("mental")


def roll_random_quirk() -> Quirk:
    """Roll a random quirk (1d20).

    Returns:
        Random quirk from the table
    """
    import random
    roll = random.randint(1, 20)
    return get_quirk_by_roll(roll)


class QuirkApplicationInput(FrozenModel):
    """Input for applying a quirk to a pilot."""

    pilot_id: str = Field(..., description="Pilot receiving the quirk")
    quirk: Quirk = Field(..., description="Quirk to apply")
    source: Literal["clone", "down_and_out_trauma"] = Field(
        ...,
        description="How the quirk was acquired"
    )
    existing_quirks: list[Quirk] = Field(
        default_factory=list,
        description="Quirks pilot already has"
    )


class QuirkApplicationResult(FrozenModel):
    """Result of applying a quirk to a pilot."""

    applied: bool = Field(..., description="Whether quirk was successfully applied")
    quirk: Quirk = Field(..., description="The quirk that was applied")
    source: Literal["clone", "down_and_out_trauma"] = Field(
        ...,
        description="How the quirk was acquired"
    )
    total_quirks: int = Field(..., description="Total quirks pilot now has")
    narrative_prompts: list[str] = Field(
        default_factory=list,
        description="Story hooks for the GM"
    )
    gameplay_notes: list[str] = Field(
        default_factory=list,
        description="Notes about gameplay implications (should be empty per PR2)"
    )


def apply_quirk(input_data: QuirkApplicationInput) -> QuirkApplicationResult:
    """Apply a quirk to a pilot.

    Per PR2 4819-4822:
    - "The quirk could be physical or mental in nature, but whatever the quirk is,
      it should be a story hook or something narrative in design (it shouldn't have
      any major gameplay effects)."
    - "Quirks are always complicating - though your character might adjust to them
      in time."

    Args:
        input_data: Quirk application input

    Returns:
        Quirk application result with narrative prompts
    """
    narrative_prompts = generate_narrative_prompts(input_data.quirk)

    return QuirkApplicationResult(
        applied=True,
        quirk=input_data.quirk,
        source=input_data.source,
        total_quirks=len(input_data.existing_quirks) + 1,
        narrative_prompts=narrative_prompts,
        gameplay_notes=[
            "Quirk has no gameplay effects per PR2",
            "Quirk is purely a narrative hook"
        ]
    )


def generate_narrative_prompts(quirk: Quirk) -> list[str]:
    """Generate narrative prompts/hooks for a quirk.

    Args:
        quirk: The quirk to generate prompts for

    Returns:
        List of narrative prompts for the GM
    """
    prompts: list[str] = []

    prompts.append(f"QUIRK: {quirk.name} (1d20={quirk.roll})")
    prompts.append(f"Type: {quirk.quirk_type.upper()}")
    prompts.append(f"Description: {quirk.description}")
    prompts.append("")
    prompts.append("Narrative Hooks:")

    if quirk.roll == 1:
        prompts.append("- Who caused the original damage?")
        prompts.append("- What cybernetics were added?")
        prompts.append("- Does the pilot know about the replacements?")
    elif quirk.roll == 2:
        prompts.append("- How does the visible augment affect social interactions?")
        prompts.append("- Who stares and why?")
        prompts.append("- Is the augment useful or just conspicuous?")
    elif quirk.roll == 3:
        prompts.append("- Who was the original body owner?")
        prompts.append("- Are the 'original's enemies now the pilot's enemies?")
        prompts.append("- Will the 'original' ever appear?")
    elif quirk.roll == 4:
        prompts.append("- How does the scar affect the pilot's self-image?")
        prompts.append("- Do people treat the pilot differently?")
        prompts.append("- Can the scar be hidden?")
    elif quirk.roll == 5:
        prompts.append("- What was the administrative error?")
        prompts.append("- Is there a record of the 'real' pilot?")
        prompts.append("- Who has the corrected records?")
    elif quirk.roll == 6:
        prompts.append("- What triggers the extra limb's movement?")
        prompts.append("- Is the limb useful or just disturbing?")
        prompts.append("- Does the limb have its own 'agenda'?")
    elif quirk.roll == 7:
        prompts.append("- Who is tracking the pilot via the barcode?")
        prompts.append("- What organizations know about this code?")
        prompts.append("- Can the barcode be removed?")
    elif quirk.roll == 8:
        prompts.append("- What information is encoded in the script?")
        prompts.append("- Who is looking for this information?")
        prompts.append("- Is the script dangerous to read?")
    elif quirk.roll == 9:
        prompts.append("- What environments are unsafe?")
        prompts.append("- How does the pilot cope with this limitation?")
        prompts.append("- Is there a way to become less susceptible?")
    elif quirk.roll == 10:
        prompts.append("- What is the non-human DNA source?")
        prompts.append("- What experiments were performed?")
        prompts.append("- Are there side effects yet to manifest?")
    elif quirk.roll == 11:
        prompts.append("- How does the pilot cope with death dreams?")
        prompts.append("- Is there any truth to the visions?")
        prompts.append("- Does the pilot fear sleep?")
    elif quirk.roll == 12:
        prompts.append("- Is the pilot 'really' conscious?")
        prompts.append("- Does this uncertainty bother the pilot?")
        prompts.append("- How do others treat a 'digital' person?")
    elif quirk.roll == 13:
        prompts.append("- How does the pilot define 'self'?")
        prompts.append("- Can the pilot accept their new existence?")
        prompts.append("- Is there a way to become 'real'?")
    elif quirk.roll == 14:
        prompts.append("- Whose memories were implanted?")
        prompts.append("- What dangerous information was gained?")
        prompts.append("- Who is searching for these memories?")
    elif quirk.roll == 15:
        prompts.append("- What was the pilot like before?")
        prompts.append("- What needs to be relearned?")
        prompts.append("- Are there fragments of the old personality?")
    elif quirk.roll == 16:
        prompts.append("- How much time remains?")
        prompts.append("- Is there any way to extend the lifespan?")
        prompts.append("- How does the pilot feel about their limited time?")
    elif quirk.roll == 17:
        prompts.append("- Who/what is the contacting entity?")
        prompts.append("- What does the entity want?")
        prompts.append("- Can the contact be severed?")
    elif quirk.roll == 18:
        prompts.append("- Which visions come true?")
        prompts.append("- How reliable are the flashes?")
        prompts.append("- Can the visions be influenced or controlled?")
    elif quirk.roll == 19:
        prompts.append("- What trigger words exist?")
        prompts.append("- Who implanted the trigger?")
        prompts.append("- Can the trigger be removed or resisted?")
    elif quirk.roll == 20:
        prompts.append("- What was the pilot's life before death?")
        prompts.append("- How does the pilot rebuild their identity?")
        prompts.append("- What triggers need to be reassigned?")

    prompts.append("")
    prompts.append("Character Development:")
    prompts.append(f"How does {quirk.name} affect the pilot's relationships, goals, and self-image?")

    return prompts


def get_all_quirks() -> list[Quirk]:
    """Get the complete quirk table.

    Returns:
        All 20 quirks
    """
    return QUIRK_TABLE


def count_quirks() -> int:
    """Get the total number of quirks in the table.

    Returns:
        20 (all quirks from PR2)
    """
    return len(QUIRK_TABLE)
