"""Clone state models for Lancer pilot cloning system.

Defines the state tracking for pilot cloning eligibility, clone status,
and quirk assignment per PR2 4789-4960.
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel


class CloneStatus(FrozenModel):
    """Tracks a pilot's cloning eligibility and history."""

    times_cloned: int = Field(
        default=0, ge=0, le=1, description="Number of times cloned (max 1)"
    )
    is_dead: bool = Field(
        default=False, description="Whether the pilot is currently dead"
    )

    @property
    def clone_available(self) -> bool:
        """Returns True if the pilot can still be cloned (hasn't exceeded limit).

        Note: This only checks if the clone limit hasn't been exceeded.
        The pilot's alive/dead state is tracked separately.
        A dead pilot can be cloned to bring them back.
        """
        return self.times_cloned < 1

    @property
    def can_be_revived(self) -> bool:
        """Returns True if the pilot can be revived via cloning."""
        return self.is_dead and self.times_cloned < 1

    def mark_cloned(self) -> "CloneStatus":
        """Returns a new CloneStatus with times_cloned incremented."""
        return CloneStatus(times_cloned=self.times_cloned + 1, is_dead=False)

    def mark_dead(self) -> "CloneStatus":
        """Returns a new CloneStatus marked as dead."""
        return CloneStatus(times_cloned=self.times_cloned, is_dead=True)

    def mark_alive(self) -> "CloneStatus":
        """Returns a new CloneStatus marked as alive (after cloning)."""
        return CloneStatus(times_cloned=self.times_cloned, is_dead=False)


QuirkType = Literal["physical", "mental"]


class Quirk(FrozenModel):
    """A flash clone quirk from the PR2 quirk table (1d20).

    Quirks are narrative hooks with no gameplay effects per PR2 4819-4822.
    """

    roll: int = Field(
        ..., ge=1, le=20, description="1d20 roll that triggers this quirk"
    )
    name: str = Field(..., description="Name of the quirk")
    description: str = Field(
        ..., description="Detailed description of the quirk effect"
    )
    quirk_type: QuirkType = Field(
        ..., description="Whether quirk is physical or mental"
    )


QuirkSource = Literal["clone", "down_and_out_trauma"]


class CloneState(FrozenModel):
    """Complete cloning state for tracking a pilot's clone history and quirks.

    Per PR2 4811-4827:
    - Cloned character can only rejoin after mission completion
    - Flash cloned pilot "rewinds" to session start but keeps LL advancement
    - Cloned character always comes back with a Quirk
    - If cloned a second time, can no longer be played as player character
    - Quirks can optionally apply to Down And Out survivors as lingering trauma
    """

    status: CloneStatus = Field(
        default_factory=CloneStatus, description="Current clone/death status"
    )
    assigned_quirk: Quirk | None = Field(
        default=None, description="Quirk assigned from cloning or trauma"
    )
    quirk_source: QuirkSource | None = Field(
        default=None, description="How the quirk was acquired"
    )
    session_start_hp: int | None = Field(
        default=None, description="HP at session start (for clone rewind)"
    )
    session_start_evasion: int | None = Field(
        default=None, description="Evasion at session start (for clone rewind)"
    )
    clone_applicable: bool = Field(
        default=True,
        description="Whether cloning rules apply to this pilot (GM discretion)",
    )

    @property
    def is_cloned(self) -> bool:
        """Returns True if pilot has been cloned at least once."""
        return self.status.times_cloned > 0

    @property
    def can_be_cloned(self) -> bool:
        """Returns True if pilot can still be cloned (hasn't exceeded limit)."""
        return self.status.clone_available and self.clone_applicable

    @property
    def has_quirk(self) -> bool:
        """Returns True if pilot has an assigned quirk."""
        return self.assigned_quirk is not None

    def with_quirk(self, quirk: Quirk, source: QuirkSource) -> "CloneState":
        """Returns a new CloneState with an assigned quirk."""
        return CloneState(
            status=self.status,
            assigned_quirk=quirk,
            quirk_source=source,
            session_start_hp=self.session_start_hp,
            session_start_evasion=self.session_start_evasion,
            clone_applicable=self.clone_applicable,
        )

    def with_increased_clone_count(self) -> "CloneState":
        """Returns a new CloneState with clone count incremented."""
        return CloneState(
            status=self.status.mark_cloned(),
            assigned_quirk=self.assigned_quirk,
            quirk_source=self.quirk_source,
            session_start_hp=self.session_start_hp,
            session_start_evasion=self.session_start_evasion,
            clone_applicable=self.clone_applicable,
        )

    def with_session_snapshot(self, hp: int, evasion: int) -> "CloneState":
        """Returns a new CloneState with session start snapshot."""
        return CloneState(
            status=self.status,
            assigned_quirk=self.assigned_quirk,
            quirk_source=self.quirk_source,
            session_start_hp=hp,
            session_start_evasion=evasion,
            clone_applicable=self.clone_applicable,
        )
