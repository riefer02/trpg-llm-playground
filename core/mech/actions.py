"""Action primitives for mech combat."""

from typing import Literal
from pydantic import BaseModel, Field

from core.shared.enums import ActionType


ActionScope = Literal["mech", "pilot", "both"]
ActionTagType = Literal["attack", "tech", "movement", "reaction", "utility", "defense", "protocol"]


class ActionDefinition(BaseModel):
    """Definition for a combat action."""

    id: str = Field(..., description="Unique action identifier")
    name: str = Field(..., description="Display name")
    action_type: ActionType
    scope: ActionScope = "mech"
    tags: list[ActionTagType] = Field(default_factory=list)

    model_config = {"frozen": True}


BASIC_MECH_ACTIONS: list[ActionDefinition] = [
    ActionDefinition(id="move", name="Move", action_type="move", scope="both", tags=["movement"]),
    ActionDefinition(id="skirmish", name="Skirmish", action_type="quick", tags=["attack"]),
    ActionDefinition(id="barrage", name="Barrage", action_type="full", tags=["attack"]),
    ActionDefinition(id="boost", name="Boost", action_type="quick", scope="both", tags=["movement"]),
    ActionDefinition(id="ram", name="Ram", action_type="quick", tags=["attack"]),
    ActionDefinition(id="grapple", name="Grapple", action_type="quick", tags=["attack"]),
    ActionDefinition(id="quick_tech", name="Quick Tech", action_type="quick", tags=["tech"]),
    ActionDefinition(id="full_tech", name="Full Tech", action_type="full", tags=["tech"]),
    ActionDefinition(id="improvised_attack", name="Improvised Attack", action_type="full", tags=["attack"]),
    ActionDefinition(id="stabilize", name="Stabilize", action_type="full", tags=["utility"]),
    ActionDefinition(id="disengage", name="Disengage", action_type="full", scope="both", tags=["movement"]),
    ActionDefinition(id="hide", name="Hide", action_type="quick", scope="both", tags=["utility"]),
    ActionDefinition(id="search", name="Search", action_type="quick", scope="both", tags=["utility"]),
    ActionDefinition(id="prepare", name="Prepare", action_type="quick", scope="both", tags=["reaction"]),
    ActionDefinition(id="overcharge", name="Overcharge", action_type="free", tags=["utility"]),
    ActionDefinition(id="brace", name="Brace", action_type="reaction", tags=["defense"]),
    ActionDefinition(id="overwatch", name="Overwatch", action_type="reaction", tags=["attack"]),
    ActionDefinition(id="skill_check", name="Skill Check", action_type="full", scope="both", tags=["utility"]),
    ActionDefinition(id="activate", name="Activate", action_type="quick", tags=["utility"]),
    ActionDefinition(id="shutdown", name="Shut Down", action_type="quick", tags=["utility"]),
    ActionDefinition(id="boot_up", name="Boot Up", action_type="full", tags=["utility"]),
    ActionDefinition(id="mount", name="Mount", action_type="full", scope="both", tags=["utility"]),
    ActionDefinition(id="dismount", name="Dismount", action_type="full", scope="both", tags=["utility"]),
    ActionDefinition(id="eject", name="Eject", action_type="quick", scope="both", tags=["utility"]),
    ActionDefinition(id="self_destruct", name="Self Destruct", action_type="quick", tags=["utility"]),
    ActionDefinition(id="fight", name="Fight", action_type="full", scope="pilot", tags=["attack"]),
    ActionDefinition(id="jockey", name="Jockey", action_type="full", scope="pilot", tags=["attack"]),
    ActionDefinition(id="bolster", name="Bolster", action_type="quick", tags=["tech"]),
    ActionDefinition(id="scan", name="Scan", action_type="quick", tags=["tech"]),
    ActionDefinition(id="lock_on", name="Lock On", action_type="quick", tags=["tech"]),
    ActionDefinition(id="invade", name="Invade", action_type="quick", tags=["tech"]),
]


ACTION_DEFINITIONS_BY_ID = {action.id: action for action in BASIC_MECH_ACTIONS}


def get_action_definition(action_id: str) -> ActionDefinition | None:
    """Look up an action definition by ID."""
    return ACTION_DEFINITIONS_BY_ID.get(action_id)
