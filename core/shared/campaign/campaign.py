"""Campaign persistence models for Lancer TTRPG.

Provides type-safe models for tracking persistent campaign state across sessions,
including pilots, mech assignments, mission history, and session metadata.

Note: Uses dict types for nested structures to avoid circular imports.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class PilotMechAssignment(BaseModel):
    """Links a pilot to their mech build, persisting across sessions.

    Attributes:
        pilot_id: Reference to the pilot (matches Pilot.id)
        mech_id: Unique identifier for this mech (e.g., "pilot_callsign-mech_name")
        mech_name: Human-readable name for the mech
        mech_build: Full mech loadout as serialized dict
        is_active: Whether this is the pilot's current active mech
    """

    pilot_id: str = Field(..., description="Reference to pilot.id")
    mech_id: str = Field(..., description="Unique mech identifier")
    mech_name: str = Field(..., description="Human-readable mech name")
    mech_build: dict = Field(..., description="Full mech loadout as serialized dict")
    is_active: bool = Field(default=True, description="Is this the pilot's active mech")


class ActiveSessionMission(BaseModel):
    """Tracks a mission that is in-progress during a session.

    Attributes:
        mission_state: The current state of the mission as serialized dict
        participating_pilot_ids: Pilots currently engaged in this mission
        started_at: When the mission was started
    """

    mission_state: dict = Field(..., description="Mission state as serialized dict")
    participating_pilot_ids: list[str] = Field(
        default_factory=list, description="IDs of pilots in this mission"
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
        participating_pilot_ids: Pilots who took part
        debrief_notes: Optional notes about the mission
    """

    mission_id: str = Field(..., description="Reference to mission.id")
    session_id: str = Field(..., description="Reference to Session.id")
    mission_name: str = Field(..., description="Mission display name")
    outcome: Literal["success", "partial", "failure", "catastrophic"] = Field(
        ..., description="Mission outcome"
    )
    completion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    mission_date: date = Field(
        default_factory=date.today, description="Completion date"
    )
    participating_pilot_ids: list[str] = Field(
        default_factory=list, description="Pilots who took part"
    )
    debrief_notes: str | None = Field(default=None, description="Session notes")


class Session(BaseModel):
    """A single play session within a campaign.

    Attributes:
        id: Unique session identifier
        session_number: Ordinal number (1, 2, 3...)
        session_date: Date of the session
        debrief: Summary of what happened
        active_missions: Missions currently in progress
        reserves_earned: Reserves gained from this session as dict list
        downtime_plans: Downtime actions taken as dict list
    """

    id: str = Field(..., description="Unique session identifier")
    session_number: int = Field(..., ge=1, description="Ordinal session number")
    session_date: date = Field(default_factory=date.today, description="Session date")
    debrief: str | None = Field(default=None, description="Session summary")
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
    def validate_mission_pilot_ids(self) -> Session:
        """Ensure all missions reference valid pilot IDs."""
        all_pilot_ids: set[str] = set()
        for mission in self.active_missions:
            all_pilot_ids.update(mission.participating_pilot_ids)
        return self


class Campaign(BaseModel):
    """Root persistence object for a Lancer campaign.

    Tracks all campaign state including pilots, their mechs, session history,
    and mission completion records.

    Attributes:
        id: Unique campaign identifier
        name: Display name for the campaign
        description: Summary of the campaign premise
        sessions: Ordered list of all sessions (completed and in-progress)
        pilots: All pilots in the campaign as serialized dicts
        pilot_mech_links: Links between pilots and their mech builds
        mission_history: Records of all completed missions
        campaign_notes: GM notes about the campaign
        created_at: When the campaign was created
        modified_at: When the campaign was last modified
    """

    id: str = Field(..., description="Unique campaign identifier")
    name: str = Field(..., description="Campaign display name")
    description: str = Field(default="", description="Campaign premise")
    sessions: list[Session] = Field(
        default_factory=list, description="All sessions in chronological order"
    )
    pilots: list[dict] = Field(
        default_factory=list, description="All pilots as serialized dicts"
    )
    pilot_mech_links: list[PilotMechAssignment] = Field(
        default_factory=list, description="Pilot to mech assignments"
    )
    mission_history: list[CampaignMissionRecord] = Field(
        default_factory=list, description="Completed mission records"
    )
    campaign_notes: str = Field(default="", description="GM campaign notes")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Campaign creation time"
    )
    modified_at: datetime = Field(
        default_factory=datetime.now, description="Last modification time"
    )

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def validate_pilot_mech_links(self) -> Campaign:
        """Ensure all mech links reference valid pilots."""
        pilot_ids = {p.get("id") for p in self.pilots}
        for link in self.pilot_mech_links:
            if link.pilot_id not in pilot_ids:
                raise ValueError(
                    f"Pilot mech link references unknown pilot: {link.pilot_id}"
                )
        return self

    @model_validator(mode="after")
    def validate_mission_pilot_ids(self) -> Campaign:
        """Ensure all mission records reference valid pilots."""
        pilot_ids = {p.get("id") for p in self.pilots}
        for session in self.sessions:
            for mission in session.active_missions:
                for pilot_id in mission.participating_pilot_ids:
                    if pilot_id not in pilot_ids:
                        raise ValueError(
                            f"Session mission references unknown pilot: {pilot_id}"
                        )
        for record in self.mission_history:
            for pilot_id in record.participating_pilot_ids:
                if pilot_id not in pilot_ids:
                    raise ValueError(
                        f"Mission history references unknown pilot: {pilot_id}"
                    )
        return self

    @model_validator(mode="after")
    def validate_unique_pilot_ids(self) -> Campaign:
        """Ensure all pilots have unique IDs."""
        pilot_ids = [p.get("id") for p in self.pilots]
        if len(pilot_ids) != len(set(pilot_ids)):
            raise ValueError("Duplicate pilot IDs found in campaign")
        return self

    @model_validator(mode="after")
    def validate_unique_mech_ids(self) -> Campaign:
        """Ensure all mech assignments have unique mech IDs."""
        mech_ids = [link.mech_id for link in self.pilot_mech_links]
        if len(mech_ids) != len(set(mech_ids)):
            raise ValueError("Duplicate mech IDs found in pilot mech links")
        return self

    @model_validator(mode="after")
    def validate_session_numbers(self) -> Campaign:
        """Ensure sessions have unique, sequential session numbers."""
        session_numbers = [s.session_number for s in self.sessions]
        if len(session_numbers) != len(set(session_numbers)):
            raise ValueError("Duplicate session numbers found")
        return self

    def get_pilot(self, pilot_id: str) -> dict | None:
        """Get a pilot by ID."""
        for pilot in self.pilots:
            if pilot.get("id") == pilot_id:
                return pilot
        return None

    def get_pilot_mech_assignment(self, pilot_id: str) -> list[PilotMechAssignment]:
        """Get all mech assignments for a pilot."""
        return [link for link in self.pilot_mech_links if link.pilot_id == pilot_id]

    def get_active_mech_for_pilot(self, pilot_id: str) -> PilotMechAssignment | None:
        """Get the active mech assignment for a pilot."""
        for link in self.pilot_mech_links:
            if link.pilot_id == pilot_id and link.is_active:
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

    def get_pilot_mission_history(self, pilot_id: str) -> list[CampaignMissionRecord]:
        """Get all mission records for a specific pilot."""
        return [
            r for r in self.mission_history if pilot_id in r.participating_pilot_ids
        ]

    def get_active_missions_for_pilot(
        self, pilot_id: str
    ) -> list[ActiveSessionMission]:
        """Get all in-progress missions for a specific pilot."""
        active: list[ActiveSessionMission] = []
        for session in self.sessions:
            for mission in session.active_missions:
                if pilot_id in mission.participating_pilot_ids:
                    active.append(mission)
        return active

    def pilot_level(self, pilot_id: str) -> int | None:
        """Get the license level of a pilot."""
        pilot = self.get_pilot(pilot_id)
        if pilot:
            return pilot.get("level", 0)
        return None

    def pilot_is_dead(self, pilot_id: str) -> bool | None:
        """Check if a pilot is dead."""
        pilot = self.get_pilot(pilot_id)
        if pilot:
            return pilot.get("is_dead", False)
        return None
