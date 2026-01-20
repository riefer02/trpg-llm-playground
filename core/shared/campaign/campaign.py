"""Campaign persistence models for Lancer TTRPG.

Provides type-safe models for tracking persistent campaign state across sessions,
including characters, mech assignments, mission history, and session metadata.

Note: Uses dict types for nested structures to avoid circular imports.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from pydantic import AliasChoices, BaseModel, Field, model_validator

MissionLifecyclePhase = Literal["downtime", "brief", "prep", "mission", "debrief"]
LifecycleStatus = Literal["pending", "in_progress", "complete"]
ObjectivePriority = Literal["primary", "secondary", "optional"]
ReservePlanStatus = Literal["planned", "spent", "earned"]
LobbyStatus = Literal["draft", "ready", "launched", "cooldown"]
MissionOutcomeResult = Literal["success", "partial", "failure", "catastrophic"]

MISSION_LIFECYCLE_ORDER: tuple[MissionLifecyclePhase, ...] = (
    "downtime",
    "brief",
    "prep",
    "mission",
    "debrief",
)


class SessionLifecycleCheckpoint(BaseModel):
    """Represents one checkpoint in the Downtime → Brief → Prep → Mission → Debrief loop."""

    phase: MissionLifecyclePhase = Field(..., description="Lifecycle phase name")
    status: LifecycleStatus = Field(
        default="pending", description="Current status of this checkpoint"
    )
    summary: str = Field(default="", description="Narrative summary for this phase")
    gm_notes: str | None = Field(default=None, description="Private GM notes")
    completed_at: datetime | None = Field(
        default=None, description="Timestamp when this checkpoint completed"
    )
    issues: list[str] = Field(
        default_factory=list, description="Any issues raised during this phase"
    )
    reserves_spent: list[dict] = Field(
        default_factory=list,
        description="Reserves consumed in this phase (validated elsewhere)",
    )


class MissionObjectiveBrief(BaseModel):
    """Lightweight mission objective summary for lobby/session prep."""

    id: str = Field(..., description="Local identifier for this objective")
    title: str = Field(..., description="Display title for the objective")
    success_condition: str = Field(
        ..., description="What success looks like for this objective"
    )
    priority: ObjectivePriority = Field(
        default="primary", description="Priority communicated to players"
    )
    related_objective_id: str | None = Field(
        default=None,
        description="Optional reference to core.shared.scenario MissionObjective.id",
    )


class MissionStakesBrief(BaseModel):
    """Summarizes mission stakes per PR2 guidance (lines ~2835-2841)."""

    stakes_type: Literal["personal", "faction", "immediate", "gradual", "custom"]
    summary: str = Field(..., description="Narrative description of the stakes")
    consequences_success: str | None = Field(default=None)
    consequences_failure: str | None = Field(default=None)
    consequences_partial: str | None = Field(default=None)


class ReservePlanEntry(BaseModel):
    """Tracks reserve usage/assignments planned for a mission."""

    reserve_id: str = Field(..., description="Identifier for the reserve asset")
    assigned_character_id: str | None = Field(
        default=None,
        description="Character expected to use this reserve",
        validation_alias=AliasChoices(
            "assigned_character_id",
            "assigned_pilot_id",
        ),
        serialization_alias="assigned_character_id",
    )
    usage_notes: str | None = Field(
        default=None, description="When/how it will be used"
    )
    status: ReservePlanStatus = Field(default="planned")


class MissionPrepPlan(BaseModel):
    """Mission prep metadata surfaced in the lobby UI."""

    mission_name: str = Field(..., description="Operation name shared with table")
    operation_code: str | None = Field(default=None, description="Optional ops code")
    theater: str | None = Field(default=None, description="Where this mission occurs")
    objectives: list[MissionObjectiveBrief] = Field(
        default_factory=list, description="Objectives highlighted for this mission"
    )
    stakes: MissionStakesBrief | None = Field(default=None)
    reserves: list[ReservePlanEntry] = Field(
        default_factory=list, description="Reserve plan for the mission"
    )
    briefing_notes: str = Field(default="", description="GM notes for the briefing")
    support_assets: list[str] = Field(
        default_factory=list, description="Allies/support available"
    )
    threats: list[str] = Field(default_factory=list, description="Known threats")


class CampaignIdentity(BaseModel):
    """Stores onboarding prompts per PR2 tables (Who are we? patrons, etc.)."""

    squad_name: str = Field(default="", description="What the squad calls itself")
    patron: str = Field(default="", description="Primary patron or employer")
    who_we_are: str = Field(default="", description="Elevator pitch for the squad")
    relationships: list[str] = Field(
        default_factory=list, description="Key relationships/prompts"
    )
    themes: list[str] = Field(default_factory=list, description="Tone or themes")
    gm_prompts: list[str] = Field(
        default_factory=list, description="Reference prompts for the GM"
    )


class CampaignLobbyState(BaseModel):
    """Active lobby plan that gates combat launch."""

    mission_plan: MissionPrepPlan = Field(
        ..., description="Mission prep plan surfaced to the table"
    )
    assigned_member_ids: list[str] = Field(
        default_factory=list, description="Member IDs slated for this mission"
    )
    preferred_pilot_count: int = Field(
        default=4, ge=1, le=6, description="Soft cap per PR2 (GM + 3-5 pilots)"
    )
    min_pilot_count: int = Field(
        default=3, ge=1, le=5, description="Minimum pilots before launch is allowed"
    )
    gm_notes: str = Field(default="", description="GM-only prep notes")
    status: LobbyStatus = Field(default="draft", description="Lobby readiness state")
    last_ready_check: datetime | None = Field(default=None)
    combat_session_id: str | None = Field(
        default=None, description="Linked CombatSessionDB id once launched"
    )
    enemy_force_preview: dict | None = Field(
        default=None,
        description="Preview of enemy force composition (EnemyForcePreview as dict)",
    )

    @model_validator(mode="after")
    def validate_seat_limits(self) -> "CampaignLobbyState":
        if self.min_pilot_count > self.preferred_pilot_count:
            raise ValueError("Minimum pilots cannot exceed preferred pilot count")
        return self


class MissionOutcomeReport(BaseModel):
    """Summary of a mission's outcome captured during debrief."""

    outcome: MissionOutcomeResult = Field(
        ..., description="Overall mission result communicated to the table"
    )
    completion_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0-1 scale for partial success tracking",
    )
    debrief_notes: str | None = Field(
        default=None, description="Narrative debrief shared with the table"
    )
    reserves_spent: list[dict] = Field(
        default_factory=list, description="Reserves consumed during the mission"
    )
    reserves_earned: list[dict] = Field(
        default_factory=list, description="Reserves awarded after the mission"
    )
    rewards: list[str] = Field(
        default_factory=list, description="Specific rewards, loot, or consequences"
    )
    recorded_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when this outcome was recorded",
    )


def _default_lifecycle_checkpoints() -> list[SessionLifecycleCheckpoint]:
    """Build the canonical Downtime → Brief → Prep → Mission → Debrief checkpoints."""

    return [
        SessionLifecycleCheckpoint(phase=phase) for phase in MISSION_LIFECYCLE_ORDER
    ]


class CharacterMechAssignment(BaseModel):
    """Links a character to their mech build, persisting across sessions.

    Attributes:
        character_id: Reference to the character (matches Character.id)
        mech_id: Unique identifier for this mech (e.g., "callsign-mech_name")
        mech_name: Human-readable name for the mech
        mech_build: Full mech loadout as serialized dict
        is_active: Whether this is the character's current active mech
    """

    character_id: str = Field(
        ...,
        description="Reference to character.id",
        validation_alias=AliasChoices("character_id", "pilot_id"),
        serialization_alias="character_id",
    )
    mech_id: str = Field(..., description="Unique mech identifier")
    mech_name: str = Field(..., description="Human-readable mech name")
    mech_build: dict = Field(..., description="Full mech loadout as serialized dict")
    is_active: bool = Field(
        default=True, description="Is this the character's active mech"
    )


class ActiveSessionMission(BaseModel):
    """Tracks a mission that is in-progress during a session.

    Attributes:
        mission_state: The current state of the mission as serialized dict
        participating_character_ids: Characters currently engaged in this mission
        started_at: When the mission was started
    """

    mission_state: dict = Field(..., description="Mission state as serialized dict")
    participating_character_ids: list[str] = Field(
        default_factory=list,
        description="IDs of characters in this mission",
        validation_alias=AliasChoices(
            "participating_character_ids",
            "participating_pilot_ids",
        ),
        serialization_alias="participating_character_ids",
    )
    started_at: datetime = Field(
        default_factory=datetime.now, description="When the mission was started"
    )


class CampaignMissionRecord(BaseModel):
    """Records a completed mission for campaign history.

    Attributes:
        mission_id: Reference to the mission definition
        session_id: Which session this mission was completed in
        mission_name: Display name of the mission
        outcome: Mission outcome (success, partial, failure, catastrophic)
        completion_score: 0.0-1.0 progress score
        mission_date: When the mission was completed
        participating_character_ids: Characters who took part
        debrief_notes: Optional notes about the mission
    """

    mission_id: str = Field(..., description="Reference to mission.id")
    session_id: str = Field(..., description="Reference to Session.id")
    mission_name: str = Field(..., description="Mission display name")
    outcome: MissionOutcomeResult = Field(..., description="Mission outcome")
    completion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    mission_date: date = Field(
        default_factory=date.today, description="Completion date"
    )
    participating_character_ids: list[str] = Field(
        default_factory=list,
        description="Characters who took part",
        validation_alias=AliasChoices(
            "participating_character_ids",
            "participating_pilot_ids",
        ),
        serialization_alias="participating_character_ids",
    )
    debrief_notes: str | None = Field(default=None, description="Session notes")
    reserves_spent: list[dict] = Field(
        default_factory=list, description="Reserves consumed during the mission"
    )
    reserves_earned: list[dict] = Field(
        default_factory=list, description="Reserves awarded after the mission"
    )
    rewards: list[str] = Field(
        default_factory=list, description="Loot, intel, or standing changes"
    )


class Session(BaseModel):
    """A single play session within a campaign.

    Attributes:
        id: Unique session identifier
        session_number: Ordinal number (1, 2, 3...)
        session_date: Date of the session
        debrief: Summary of what happened
        mission_plan: Mission prep data highlighted during the session
        lifecycle_checkpoints: Downtime → Brief → Prep → Mission → Debrief tracking
        active_missions: Missions currently in progress
        reserves_earned: Reserves gained from this session as dict list
        downtime_plans: Downtime actions taken as dict list
    """

    id: str = Field(..., description="Unique session identifier")
    session_number: int = Field(..., ge=1, description="Ordinal session number")
    session_date: date = Field(default_factory=date.today, description="Session date")
    debrief: str | None = Field(default=None, description="Session summary")
    mission_plan: MissionPrepPlan | None = Field(
        default=None, description="Mission prep metadata tied to this session"
    )
    mission_outcome: MissionOutcomeReport | None = Field(
        default=None, description="Recorded mission outcome / debrief summary"
    )
    lifecycle_checkpoints: list[SessionLifecycleCheckpoint] = Field(
        default_factory=_default_lifecycle_checkpoints,
        description="Downtime → Brief → Prep → Mission → Debrief checkpoints",
    )
    active_missions: list[ActiveSessionMission] = Field(
        default_factory=list, description="In-progress missions"
    )
    reserves_earned: list[dict] = Field(
        default_factory=list, description="Reserves earned this session"
    )
    downtime_plans: list[dict] = Field(
        default_factory=list, description="Downtime actions taken"
    )

    @model_validator(mode="after")
    def validate_mission_character_ids(self) -> "Session":
        """Ensure all missions reference valid character IDs."""
        all_character_ids: set[str] = set()
        for mission in self.active_missions:
            all_character_ids.update(mission.participating_character_ids)
        return self

    @model_validator(mode="after")
    def validate_lifecycle_phases(self) -> "Session":
        """Ensure lifecycle checkpoints cover the canonical phases."""
        seen: set[MissionLifecyclePhase] = set()
        for checkpoint in self.lifecycle_checkpoints:
            if checkpoint.phase in seen:
                raise ValueError(f"Duplicate lifecycle phase found: {checkpoint.phase}")
            seen.add(checkpoint.phase)
        missing = [phase for phase in MISSION_LIFECYCLE_ORDER if phase not in seen]
        if missing:
            raise ValueError(
                "Session lifecycle is missing phases: " + ", ".join(missing)
            )
        return self


class Campaign(BaseModel):
    """Root persistence object for a Lancer campaign.

    Tracks all campaign state including characters, their mechs, session history,
    and mission completion records.

    Attributes:
        id: Unique campaign identifier
        name: Display name for the campaign
        description: Summary of the campaign premise
        sessions: Ordered list of all sessions (completed and in-progress)
        characters: All characters in the campaign as serialized dicts
        character_mech_links: Links between characters and their mech builds
        mission_history: Records of all completed missions
        campaign_notes: GM notes about the campaign
        identity: Onboarding prompts/patrons for fast reference
        lobby_state: Active lobby data ready to launch into combat
        created_at: When the campaign was created
        modified_at: When the campaign was last modified
    """

    id: str = Field(..., description="Unique campaign identifier")
    name: str = Field(..., description="Campaign display name")
    description: str = Field(default="", description="Campaign premise")
    sessions: list[Session] = Field(
        default_factory=list, description="All sessions in chronological order"
    )
    characters: list[dict] = Field(
        default_factory=list,
        description="All characters as serialized dicts",
        validation_alias=AliasChoices("characters", "pilots"),
        serialization_alias="characters",
    )
    character_mech_links: list[CharacterMechAssignment] = Field(
        default_factory=list,
        description="Character to mech assignments",
        validation_alias=AliasChoices("character_mech_links", "pilot_mech_links"),
        serialization_alias="character_mech_links",
    )
    mission_history: list[CampaignMissionRecord] = Field(
        default_factory=list, description="Completed mission records"
    )
    campaign_notes: str = Field(default="", description="GM campaign notes")
    identity: CampaignIdentity | None = Field(
        default=None, description="Who we are / patron prompts"
    )
    lobby_state: CampaignLobbyState | None = Field(
        default=None, description="Active lobby/mission prep data"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Campaign creation time"
    )
    modified_at: datetime = Field(
        default_factory=datetime.now, description="Last modification time"
    )

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def validate_character_mech_links(self) -> Campaign:
        """Ensure all mech links reference valid characters."""
        character_ids = {c.get("id") for c in self.characters}
        for link in self.character_mech_links:
            if link.character_id not in character_ids:
                raise ValueError(
                    f"Character mech link references unknown character: {link.character_id}"
                )
        return self

    @model_validator(mode="after")
    def validate_mission_character_ids(self) -> Campaign:
        """Ensure all mission records reference valid characters."""
        character_ids = {c.get("id") for c in self.characters}
        for session in self.sessions:
            for mission in session.active_missions:
                for character_id in mission.participating_character_ids:
                    if character_id not in character_ids:
                        raise ValueError(
                            "Session mission references unknown character: "
                            f"{character_id}"
                        )
        for record in self.mission_history:
            for character_id in record.participating_character_ids:
                if character_id not in character_ids:
                    raise ValueError(
                        f"Mission history references unknown character: {character_id}"
                    )
        return self

    @model_validator(mode="after")
    def validate_unique_character_ids(self) -> Campaign:
        """Ensure all characters have unique IDs."""
        character_ids = [c.get("id") for c in self.characters]
        if len(character_ids) != len(set(character_ids)):
            raise ValueError("Duplicate character IDs found in campaign")
        return self

    @model_validator(mode="after")
    def validate_unique_mech_ids(self) -> Campaign:
        """Ensure all mech assignments have unique mech IDs."""
        mech_ids = [link.mech_id for link in self.character_mech_links]
        if len(mech_ids) != len(set(mech_ids)):
            raise ValueError("Duplicate mech IDs found in character mech links")
        return self

    @model_validator(mode="after")
    def validate_session_numbers(self) -> Campaign:
        """Ensure sessions have unique, sequential session numbers."""
        session_numbers = [s.session_number for s in self.sessions]
        if len(session_numbers) != len(set(session_numbers)):
            raise ValueError("Duplicate session numbers found")
        return self

    def get_character(self, character_id: str) -> dict | None:
        """Get a character by ID."""
        for character in self.characters:
            if character.get("id") == character_id:
                return character
        return None

    def get_character_mech_assignment(
        self, character_id: str
    ) -> list[CharacterMechAssignment]:
        """Get all mech assignments for a character."""
        return [
            link for link in self.character_mech_links if link.character_id == character_id
        ]

    def get_active_mech_for_character(
        self, character_id: str
    ) -> CharacterMechAssignment | None:
        """Get the active mech assignment for a character."""
        for link in self.character_mech_links:
            if link.character_id == character_id and link.is_active:
                return link
        return None

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        for session in self.sessions:
            if session.id == session_id:
                return session
        return None

    def get_mission_record(self, mission_id: str) -> list[CampaignMissionRecord]:
        """Get all mission records for a specific mission."""
        return [r for r in self.mission_history if r.mission_id == mission_id]

    def get_character_mission_history(
        self, character_id: str
    ) -> list[CampaignMissionRecord]:
        """Get all mission records for a specific character."""
        return [
            r
            for r in self.mission_history
            if character_id in r.participating_character_ids
        ]

    def get_active_missions_for_character(
        self, character_id: str
    ) -> list[ActiveSessionMission]:
        """Get all in-progress missions for a specific character."""
        active: list[ActiveSessionMission] = []
        for session in self.sessions:
            for mission in session.active_missions:
                if character_id in mission.participating_character_ids:
                    active.append(mission)
        return active

    def character_level(self, character_id: str) -> int | None:
        """Get the license level of a character."""
        character = self.get_character(character_id)
        if character and isinstance(character, dict):
            pilot = character.get("pilot", character)
            if isinstance(pilot, dict):
                return pilot.get("level", 0)
        return None

    def character_is_dead(self, character_id: str) -> bool | None:
        """Check if a character's pilot is dead."""
        character = self.get_character(character_id)
        if character and isinstance(character, dict):
            pilot = character.get("pilot", character)
            if isinstance(pilot, dict):
                return pilot.get("is_dead", False)
        return None
