"""Unified Character model for Lancer TTRPG.

A Character represents a complete player entity: pilot + mech(s).
This model unifies the pilot and mech data, automatically computing
mech stats from pilot skills, grit, and core bonuses.

From the Lancer Core Book (PR2):
"Your pilot and your mech are effectively two components of the same character."
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pydantic import BaseModel, Field, computed_field, model_validator

from core.pilot.pilot import Pilot
from core.pilot.core_bonus import get_core_bonus_definition
from core.mech.build import (
    MechBuild,
    MechDerivedStats,
    compute_mech_stats,
)
from core.mech.frame import MechFrameDefinition
from core.mech.compendium import get_frame_definition
from core.shared.effects import MechanicalEffect
from core.shared.id_helpers import CharacterIdField, MechIdField, FrameIdField
from core.character.damage_state import MechDamageState

if TYPE_CHECKING:
    from core.character.validation import CharacterValidation


class MechConfiguration(BaseModel):
    """A named mech loadout owned by a character.

    Represents a single mech configuration that a pilot can use.
    The actual combat stats are computed based on the pilot's skills,
    grit, and core bonuses when accessed through the Character model.
    """

    id: MechIdField = Field(..., description="Unique mech configuration ID")
    name: str = Field(..., min_length=1, description="Custom name (e.g., 'RAIJIN')")
    frame_id: FrameIdField = Field(..., description="Frame definition ID")
    build: MechBuild = Field(
        default_factory=lambda: MechBuild(frame_id="gms_everest"),
        description="Weapon and system loadout",
    )
    damage_state: MechDamageState | None = Field(
        default=None,
        description="Current damage state (None = full health)",
    )

    model_config = {"validate_assignment": True}

    def get_frame(self) -> MechFrameDefinition | None:
        """Get the frame definition for this mech."""
        return get_frame_definition(self.frame_id)


class Character(BaseModel):
    """A complete Lancer character: pilot + mech(s).

    The Character model unifies pilot progression and mech loadouts,
    automatically computing mech stats based on pilot data.

    Key features:
    - Owns the Pilot model (composition)
    - Owns list of MechConfiguration (multiple mechs allowed)
    - Automatically computes active mech stats from pilot skills/grit
    - Automatically collects core bonus effects
    - Provides holistic validation (pilot + mech + license gating)
    """

    id: CharacterIdField = Field(default="", description="Unique character ID")
    pilot: Pilot = Field(..., description="The pilot (person inside the machine)")
    mechs: list[MechConfiguration] = Field(
        default_factory=list,
        description="Mech configurations owned by this character",
    )
    active_mech_id: MechIdField | None = Field(
        default=None,
        description="ID of the currently active mech (None if no mech selected)",
    )

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def _validate_active_mech_exists(self) -> "Character":
        """Ensure active_mech_id references an existing mech."""
        if self.active_mech_id is not None:
            mech_ids = {m.id for m in self.mechs}
            if self.active_mech_id not in mech_ids:
                raise ValueError(
                    f"active_mech_id '{self.active_mech_id}' not found in mechs"
                )
        return self

    @computed_field
    @property
    def active_mech(self) -> MechConfiguration | None:
        """Get the currently active mech configuration."""
        if self.active_mech_id is None:
            return None
        for mech in self.mechs:
            if mech.id == self.active_mech_id:
                return mech
        return None

    @computed_field
    @property
    def core_bonus_effects(self) -> list[MechanicalEffect]:
        """Collect mechanical effects from all pilot core bonuses."""
        effects: list[MechanicalEffect] = []
        for cb in self.pilot.core_bonuses:
            definition = get_core_bonus_definition(cb.core_bonus_id)
            if definition:
                effects.append(definition.effects)
        return effects

    @computed_field
    @property
    def active_mech_stats(self) -> MechDerivedStats | None:
        """Compute derived stats for the active mech.

        Returns None if no active mech is selected or frame is invalid.
        Stats are computed using:
        - Frame base stats
        - Pilot mech skills (HASE)
        - Pilot grit (1/2 level)
        - Core bonus effects
        """
        mech = self.active_mech
        if mech is None:
            return None

        frame = mech.get_frame()
        if frame is None:
            return None

        return compute_mech_stats(
            frame=frame,
            skills=self.pilot.skills,
            grit=self.pilot.grit,
            bonus_effects=self.core_bonus_effects if self.core_bonus_effects else None,
        )

    def get_mech(self, mech_id: MechIdField) -> MechConfiguration | None:
        """Get a mech configuration by ID."""
        for mech in self.mechs:
            if mech.id == mech_id:
                return mech
        return None

    def add_mech(self, mech: MechConfiguration) -> "Character":
        """Return a new Character with an additional mech configuration.

        If this is the first mech, it becomes the active mech.
        """
        existing_ids = {m.id for m in self.mechs}
        if mech.id in existing_ids:
            raise ValueError(f"Mech ID '{mech.id}' already exists")

        new_mechs = [*self.mechs, mech]
        new_active = self.active_mech_id if self.active_mech_id else mech.id

        return Character(
            id=self.id,
            pilot=self.pilot,
            mechs=new_mechs,
            active_mech_id=new_active,
        )

    def remove_mech(self, mech_id: MechIdField) -> "Character":
        """Return a new Character with a mech configuration removed.

        If the removed mech was active, active_mech_id is cleared.
        """
        new_mechs = [m for m in self.mechs if m.id != mech_id]
        if len(new_mechs) == len(self.mechs):
            raise ValueError(f"Mech ID '{mech_id}' not found")

        new_active = self.active_mech_id if self.active_mech_id != mech_id else None

        return Character(
            id=self.id,
            pilot=self.pilot,
            mechs=new_mechs,
            active_mech_id=new_active,
        )

    def set_active_mech(self, mech_id: MechIdField | None) -> "Character":
        """Return a new Character with a different active mech."""
        if mech_id is not None:
            mech_ids = {m.id for m in self.mechs}
            if mech_id not in mech_ids:
                raise ValueError(f"Mech ID '{mech_id}' not found")

        return Character(
            id=self.id,
            pilot=self.pilot,
            mechs=self.mechs,
            active_mech_id=mech_id,
        )

    def update_mech(self, mech_id: MechIdField, **updates: object) -> "Character":
        """Return a new Character with an updated mech configuration.

        Args:
            mech_id: ID of the mech to update
            **updates: Fields to update on the MechConfiguration

        Returns:
            New Character with the updated mech
        """
        new_mechs: list[MechConfiguration] = []
        found = False

        for mech in self.mechs:
            if mech.id == mech_id:
                found = True
                new_mechs.append(mech.model_copy(update=updates))
            else:
                new_mechs.append(mech)

        if not found:
            raise ValueError(f"Mech ID '{mech_id}' not found")

        return Character(
            id=self.id,
            pilot=self.pilot,
            mechs=new_mechs,
            active_mech_id=self.active_mech_id,
        )

    def update_pilot(self, **updates: object) -> "Character":
        """Return a new Character with an updated pilot.

        Args:
            **updates: Fields to update on the Pilot

        Returns:
            New Character with the updated pilot
        """
        return Character(
            id=self.id,
            pilot=self.pilot.model_copy(update=updates),
            mechs=self.mechs,
            active_mech_id=self.active_mech_id,
        )

    def validate_character(self) -> "CharacterValidation":
        """Validate this character against all game rules.

        Includes:
        - Pilot progression validation
        - Mech build validation for each mech
        - License gating checks
        - LL0-specific rules
        """
        from core.character.validation import validate_character

        return validate_character(self)
