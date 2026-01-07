"""Downtime actions and reserves system for Lancer TTRPG.

PR2 References:
- Downtime Actions: 3471-3650
- Reserves System: 3261-3400
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel


ReserveType = Literal["narrative", "mech", "tactical"]


NarrativeReserveType = Literal[
    "access",
    "backing",
    "supplies",
    "disguise",
    "diversion",
    "blackmail",
    "reputation",
    "safe_harbor",
    "tracking",
    "knowledge",
]


MechReserveType = Literal[
    "ammo",
    "rented_gear",
    "extra_repairs",
    "core_battery",
    "deployable_shield",
    "redundant_repair",
    "systems_reinforcement",
    "smart_ammo",
    "boosted_servos",
    "jump_jets",
]


TacticalReserveType = Literal[
    "scouting",
    "vehicle",
    "reinforcements",
]


class Reserve(FrozenModel):
    """A reserve that can be earned through downtime actions."""

    id: str = Field(..., description="Unique identifier for this reserve")
    reserve_type: ReserveType = Field(..., description="Category of reserve")
    specific_type: str = Field(
        ..., description="Specific type from Narrative/Mech/Tactical reserves"
    )
    description: str = Field(..., description="Mechanical description of the reserve")
    quantity: int = Field(default=1, ge=1, description="Quantity or magnitude")
    mission_scoped: bool = Field(
        default=True,
        description="Whether this reserve expires after the next mission",
    )


class DowntimeOutcome(FrozenModel):
    """Outcome of a downtime action resolution."""

    tier: Literal["failure", "mixed", "success", "exceptional"] = Field(
        ..., description="Outcome tier based on roll"
    )
    reserves_earned: list[Reserve] = Field(
        default_factory=list, description="Reserves earned from this action"
    )
    consequences: list[str] = Field(
        default_factory=list, description="Narrative consequences that apply"
    )
    state_changes: dict[str, object] = Field(
        default_factory=dict,
        description="State changes (e.g., organization efficiency)",
    )
    notes: str | None = Field(default=None, description="Additional narrative notes")


class DowntimeAction(FrozenModel):
    """Base class for downtime actions.

    All downtime actions use the same resolution pattern:
    - Roll 1d20 + relevant modifiers against target 10 (standard tier)
    - ≤9: Failure tier (worst outcome)
    - 10-19: Mixed/Success tier (moderate outcome)
    - 20+: Exceptional tier (best outcome)
    """

    id: str = Field(..., description="Unique identifier for this action")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Mechanical description")
    skill_context: str = Field(
        default="general",
        description="Skill typically used (hull/agility/systems/engineering/general)",
    )

    def get_outcome(
        self,
        roll_result: int,
        modifiers: int = 0,
        difficulty_modifier: int = 0,
    ) -> DowntimeOutcome:
        """Compute outcome based on roll result.

        Args:
            roll_result: The d20 roll
            modifiers: Accuracy bonuses from triggers, help, etc.
            difficulty_modifier: Difficulty penalty (e.g., +1 for difficult tasks)

        Returns:
            The outcome based on total result
        """
        total = roll_result + modifiers - difficulty_modifier

        if total <= 9:
            return self._failure_outcome(roll_result, modifiers, difficulty_modifier)
        elif total <= 19:
            return self._mixed_outcome(roll_result, modifiers, difficulty_modifier)
        else:
            return self._success_outcome(roll_result, modifiers, difficulty_modifier)

    def _failure_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        """Outcome for total ≤9. Override in subclasses."""
        return DowntimeOutcome(
            tier="failure",
            consequences=["Action failed"],
        )

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        """Outcome for total 10-19. Override in subclasses."""
        return DowntimeOutcome(
            tier="mixed",
            consequences=["Partial success"],
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        """Outcome for total 20+. Override in subclasses."""
        return DowntimeOutcome(
            tier="success",
            consequences=["Action succeeded"],
        )

    def _exceptional_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        """Outcome for total 20+. Override in subclasses for exceptional results."""
        return self._success_outcome(roll_result, modifiers, difficulty_modifier)


class PowerAtACost(DowntimeAction):
    """Power at a Cost - Gain rewards with narrative consequences.

    PR2: "Name what you want. You can always get it, but the GM chooses one or two
    consequences depending on how outlandish the request is."
    """

    def __init__(self, **data):
        data.setdefault("id", "power_at_a_cost")
        data.setdefault("name", "Power at a Cost")
        data.setdefault(
            "description",
            "Gain rewards, opportunities, or additional resources. "
            "The GM chooses consequences based on outlandishness.",
        )
        data.setdefault("skill_context", "general")
        super().__init__(**data)

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="mixed",
            reserves_earned=[
                Reserve(
                    id="power_cost_resource",
                    reserve_type="narrative",
                    specific_type="supplies",
                    description="Gained requested resource",
                    mission_scoped=True,
                )
            ],
            consequences=["GM selects one consequence from the list"],
            notes="Resource gained but with narrative cost",
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="success",
            reserves_earned=[
                Reserve(
                    id="power_cost_resource_success",
                    reserve_type="narrative",
                    specific_type="supplies",
                    description="Gained requested resource",
                    mission_scoped=True,
                )
            ],
            consequences=["GM selects one consequence from the list"],
            notes="Resource gained with manageable cost",
        )


class BuySomeTime(DowntimeAction):
    """Buy Some Time - Stave off reckoning, extend window of opportunity.

    PR2: "Try and stave off some reckoning, extend your window of opportunity,
    or merely buy more time and breathing room."
    """

    def __init__(self, **data):
        data.setdefault("id", "buy_some_time")
        data.setdefault("name", "Buy Some Time")
        data.setdefault(
            "description",
            "Stave off reckoning or extend window of opportunity. "
            "Can be used as reserves for next mission.",
        )
        data.setdefault("skill_context", "systems")
        super().__init__(**data)

    def _failure_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="failure",
            consequences=["Can buy only a little time if drastic measures taken now"],
            notes="Reckoning catches up unless drastic action taken",
        )

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="mixed",
            reserves_earned=[
                Reserve(
                    id="bought_time",
                    reserve_type="tactical",
                    specific_type="scouting",
                    description="Bought time for next mission",
                    mission_scoped=True,
                )
            ],
            consequences=["Situation becomes precarious or desperate"],
            notes="Time bought but situation is desperate. Next failure = ≤9",
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="success",
            reserves_earned=[
                Reserve(
                    id="bought_time_success",
                    reserve_type="tactical",
                    specific_type="scouting",
                    description="Enough time until next mission",
                    mission_scoped=True,
                )
            ],
            consequences=[],
            notes="Enough time bought until next mission",
        )


class GetADamnDrink(DowntimeAction):
    """Get a Damn Drink - Social, carouse, make connections.

    PR2: "Blow off some steam, carouse, and generally get into trouble.
    Make connections, collect gossip, forge a reputation."
    """

    def __init__(self, **data):
        data.setdefault("id", "get_a_damn_drink")
        data.setdefault("name", "Get a Damn Drink")
        data.setdefault(
            "description",
            "Blow off steam, make connections, collect gossip. "
            "Only available where there's a drink (town, station, city).",
        )
        data.setdefault("skill_context", "general")
        super().__init__(**data)

    def _failure_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="failure",
            reserves_earned=[],
            consequences=[],
            notes="Wake up in gutter, choose one: dignity, all possessions, or memory",
        )

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="mixed",
            reserves_earned=[
                Reserve(
                    id="drink_connection",
                    reserve_type="narrative",
                    specific_type="reputation",
                    description="Connection or reputation gained",
                    mission_scoped=True,
                )
            ],
            consequences=[
                "Lose one of: reputation, friend/connection, item/info, opportunity"
            ],
            notes="Gain one reserve, lose another",
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="success",
            reserves_earned=[
                Reserve(
                    id="drink_connection_1",
                    reserve_type="narrative",
                    specific_type="reputation",
                    description="Reputation gained",
                    mission_scoped=True,
                ),
                Reserve(
                    id="drink_connection_2",
                    reserve_type="narrative",
                    specific_type="knowledge",
                    description="Information or gossip gained",
                    mission_scoped=True,
                ),
            ],
            consequences=[],
            notes="Gain two reserves, lose nothing",
        )


class GetCreative(DowntimeAction):
    """Get Creative - Tweak or make something new.

    PR2: "Tweak something or attempt to make something new, either a physical project
    or software. Generally can't be as impactful as mech gear. Can be used as reserves."
    """

    def __init__(self, **data):
        data.setdefault("id", "get_creative")
        data.setdefault("name", "Get Creative")
        data.setdefault(
            "description",
            "Tweak or make something new (physical or software). "
            "Once finished, can be used as reserves.",
        )
        data.setdefault("skill_context", "systems")
        super().__init__(**data)

    def _failure_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="failure",
            consequences=["No progress on project"],
            notes="If already failed same project, next attempt = 10-19",
        )

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="mixed",
            reserves_earned=[
                Reserve(
                    id="creative_progress",
                    reserve_type="narrative",
                    specific_type="supplies",
                    description="Project materials accumulated",
                    mission_scoped=True,
                )
            ],
            consequences=["Can make progress but can't finish"],
            notes="Can finish next downtime if get: materials, knowledge, tools, workspace",
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="success",
            reserves_earned=[
                Reserve(
                    id="creative_project",
                    reserve_type="narrative",
                    specific_type="supplies",
                    description="Project completed",
                    mission_scoped=True,
                )
            ],
            consequences=[],
            notes="Project finished. If complicated, treat as 10-19 but only choose 1.",
        )


class GetFocused(DowntimeAction):
    """Get Focused - Skill improvement and self-improvement.

    PR2: "Focus on increasing your own skills, training, and self-improvement.
    Name one thing to learn/improve. GM gives new trigger at +2."
    """

    def __init__(self, **data):
        data.setdefault("id", "get_focused")
        data.setdefault("name", "Get Focused")
        data.setdefault(
            "description",
            "Increase skills, training, self-improvement. "
            "GM gives new trigger based on skill practiced at +2. Max +6.",
        )
        data.setdefault("skill_context", "general")
        super().__init__(**data)

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="mixed",
            state_changes={"trigger_bonus": 2},
            consequences=["New trigger gained at +2"],
            notes="Trigger added to pilot's skill triggers",
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="success",
            state_changes={"trigger_bonus": 2},
            consequences=["New trigger gained at +2"],
            notes="Trigger added at +2. Can improve by repeating action up to +6.",
        )


class GetOrganized(DowntimeAction):
    """Get Organized - Start, run, or improve an organization.

    PR2: "Start, run, or improve an organization, business, or venture.
    Track efficiency and influence from 0-6."
    """

    def __init__(self, **data):
        data.setdefault("id", "get_organized")
        data.setdefault("name", "Get Organized")
        data.setdefault(
            "description",
            "Start, run, or improve an organization. "
            "Track efficiency and influence 0-6. Can be used as reserves.",
        )
        data.setdefault("skill_context", "general")
        super().__init__(**data)

    def _failure_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="failure",
            state_changes={"efficiency_change": -2, "influence_change": -2},
            consequences=[
                "Organization folds unless: pay debts, prove worthiness, get bailed out, or aggressive move"
            ],
            notes="Organization in danger. Must take action to save it.",
        )

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="mixed",
            state_changes={"efficiency_change": 2, "influence_change": 0},
            consequences=["Organization is stable"],
            notes="Organization gains +2 efficiency OR influence, max 6",
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="success",
            state_changes={"efficiency_change": 2, "influence_change": 2},
            consequences=["Organization is thriving"],
            notes="Organization gains +2 efficiency AND influence, max 6",
        )


class GatherInformation(DowntimeAction):
    """Gather Information - Investigate, research, track, or spy.

    PR2: "Poke your nose around, perhaps where it doesn't belong.
    Investigating, doing research, following up on a mystery."
    """

    def __init__(self, **data):
        data.setdefault("id", "gather_information")
        data.setdefault("name", "Gather Information")
        data.setdefault(
            "description",
            "Investigate, research, follow up on mystery, track target. "
            "Information can be used as reserves.",
        )
        data.setdefault("skill_context", "systems")
        super().__init__(**data)

    def _failure_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="failure",
            reserves_earned=[
                Reserve(
                    id="info_trouble",
                    reserve_type="narrative",
                    specific_type="tracking",
                    description="Information gained but immediately causes trouble",
                    mission_scoped=True,
                )
            ],
            consequences=[],
            notes="Get info but immediately get into trouble",
        )

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="mixed",
            reserves_earned=[
                Reserve(
                    id="info_gathered",
                    reserve_type="narrative",
                    specific_type="knowledge",
                    description="Information gathered",
                    mission_scoped=True,
                )
            ],
            consequences=[
                "Choose one: leave evidence, or dispatch someone/implicate innocent"
            ],
            notes="Found information but with complication",
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="success",
            reserves_earned=[
                Reserve(
                    id="info_clean",
                    reserve_type="narrative",
                    specific_type="knowledge",
                    description="Clean information gathered",
                    mission_scoped=True,
                )
            ],
            consequences=[],
            notes="Information gained cleanly, no complications",
        )


class GetConnected(DowntimeAction):
    """Get Connected - Make connections, call in favors.

    PR2: "Try and make connections, call upon favors, ask for help,
    or drum up support. Can use connection's resources as reserves."
    """

    def __init__(self, **data):
        data.setdefault("id", "get_connected")
        data.setdefault("name", "Get Connected")
        data.setdefault(
            "description",
            "Make connections, call in favors, ask for help. "
            "Need communications or face-to-face. Resources can be reserves.",
        )
        data.setdefault("skill_context", "charm")
        super().__init__(**data)

    def _failure_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="failure",
            consequences=["Must do favor or make good on promise RIGHT NOW for help"],
            notes="If take action immediately, they'll help. Otherwise no help.",
        )

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="mixed",
            reserves_earned=[
                Reserve(
                    id="connected_help",
                    reserve_type="narrative",
                    specific_type="backing",
                    description="Connection's help for mission",
                    mission_scoped=True,
                )
            ],
            consequences=["Must do favor or make good on promise AFTER they help"],
            notes="Help now, owe later. If don't repay, next result = ≤9.",
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="success",
            reserves_earned=[
                Reserve(
                    id="connected_help_no_strings",
                    reserve_type="narrative",
                    specific_type="backing",
                    description="Connection's help with no strings attached",
                    mission_scoped=True,
                )
            ],
            consequences=[],
            notes="Help with no strings. If repeat with same org, result = 10-19.",
        )


class ScroungeAndBarter(DowntimeAction):
    """Scrounge and Barter - Get gear, assets, or physical items.

    PR2: "Get hands on gear or asset by dredging scrapyard, chasing rumors,
    bartering, or force of will. Can take on next mission as reserves."
    """

    def __init__(self, **data):
        data.setdefault("id", "scrounge_and_barter")
        data.setdefault("name", "Scrounge and Barter")
        data.setdefault(
            "description",
            "Get gear or assets for group. Can be pilot gear, vehicle, goods. "
            "Physical item, can be taken on next mission as reserves.",
        )
        data.setdefault("skill_context", "general")
        super().__init__(**data)

    def _failure_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="failure",
            reserves_earned=[
                Reserve(
                    id="scrounge_flawed",
                    reserve_type="mech",
                    specific_type="rented_gear",
                    description="Item acquired but with problems",
                    mission_scoped=True,
                )
            ],
            consequences=[
                "Choose one: stolen from someone looking for it, degraded/malfunctioning, or owner won't give up without force"
            ],
            notes="Get what looking for but with significant problem",
        )

    def _mixed_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="mixed",
            reserves_earned=[
                Reserve(
                    id="scrounge_traded",
                    reserve_type="mech",
                    specific_type="rented_gear",
                    description="Item acquired through trade",
                    mission_scoped=True,
                )
            ],
            consequences=["Choose one: time, dignity, reputation, or health/comfort"],
            notes="Get what looking for by trading something",
        )

    def _success_outcome(
        self,
        roll_result: int,
        modifiers: int,
        difficulty_modifier: int,
    ) -> DowntimeOutcome:
        return DowntimeOutcome(
            tier="success",
            reserves_earned=[
                Reserve(
                    id="scrounge_clean",
                    reserve_type="mech",
                    specific_type="rented_gear",
                    description="Item acquired cleanly",
                    mission_scoped=True,
                )
            ],
            consequences=[],
            notes="Get what looking for, no problems",
        )


def resolve_downtime_action(
    action: DowntimeAction,
    roll_result: int,
    modifiers: int = 0,
    difficulty_modifier: int = 0,
) -> DowntimeOutcome:
    """Resolve a downtime action and return the outcome.

    Args:
        action: The downtime action being taken
        roll_result: The d20 roll result (1-20)
        modifiers: Accuracy bonuses from triggers, help, etc.
        difficulty_modifier: Difficulty penalty (e.g., +1 for difficult)

    Returns:
        The outcome with reserves, consequences, and state changes
    """
    return action.get_outcome(
        roll_result=roll_result,
        modifiers=modifiers,
        difficulty_modifier=difficulty_modifier,
    )


DOWNTIME_ACTIONS: dict[str, type[DowntimeAction]] = {
    "power_at_a_cost": PowerAtACost,
    "buy_some_time": BuySomeTime,
    "get_a_damn_drink": GetADamnDrink,
    "get_creative": GetCreative,
    "get_focused": GetFocused,
    "get_organized": GetOrganized,
    "gather_information": GatherInformation,
    "get_connected": GetConnected,
    "scrounge_and_barter": ScroungeAndBarter,
}


def get_downtime_action(action_id: str) -> DowntimeAction | None:
    """Get a downtime action by ID.

    Args:
        action_id: The action identifier

    Returns:
        The action instance, or None if not found
    """
    action_class = DOWNTIME_ACTIONS.get(action_id)
    if action_class:
        return action_class()
    return None


def list_downtime_actions() -> list[tuple[str, DowntimeAction]]:
    """List all available downtime actions.

    Returns:
        List of (action_id, action_instance) tuples
    """
    return [(aid, cls()) for aid, cls in DOWNTIME_ACTIONS.items()]
