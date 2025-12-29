"""Pilot talent types for Lancer TTRPG.

Talents are special abilities that pilots can learn.
Each talent has 3 ranks, providing increasingly powerful benefits.

Note: This module contains only mechanical definitions (allowed under
the Lancer Third Party License). No copyrighted flavor text.
"""

from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.effects import (
    MechanicalEffect,
    StatModifier,
    DamageModifier,
    AccuracyModifier,
    ActionGrant,
    MovementGrant,
    MovementOverrideEffect,
    StatusGrant,
    Resistance,
    BondmateEffect,
    GrappleEffect,
    TriggeredEffect,
)


class TalentRank(FrozenModel):
    """
    A single rank of a talent (1, 2, or 3).
    
    Each rank provides specific mechanical effects.
    """
    
    rank: int = Field(..., ge=1, le=3)
    name: str = Field(..., description="Name of this rank's ability")
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)
    


class TalentDefinition(FrozenModel):
    """
    A talent definition - the template for a learnable talent.
    
    This is the "type" of talent (e.g., ACE, BONDED, etc.)
    that defines what each rank provides.
    """
    
    id: str = Field(..., description="Unique identifier (e.g., 'ace', 'bonded')")
    name: str = Field(..., description="Display name")
    ranks: list[TalentRank] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="The three ranks of this talent"
    )
    
    
    def get_rank(self, rank: int) -> TalentRank:
        """Get a specific rank (1-3)."""
        if rank < 1 or rank > 3:
            raise ValueError(f"Rank must be 1-3, got {rank}")
        return self.ranks[rank - 1]


class Talent(FrozenModel):
    """
    A talent that a pilot has learned.
    
    This represents a pilot's progress in a specific talent,
    tracking which ranks they have unlocked.
    """
    
    talent_id: str = Field(..., description="ID of the talent definition")
    rank: int = Field(default=1, ge=1, le=3, description="Current rank (1-3)")
    


# Example talent definitions with pure mechanical effects
# Note: No flavor text, only mechanical effects
EXAMPLE_TALENTS: list[TalentDefinition] = [
    TalentDefinition(
        id="ace",
        name="ACE",
        ranks=[
            TalentRank(
                rank=1,
                name="Afterburner",
                effects=MechanicalEffect(
                    triggered_effects=[
                        TriggeredEffect(
                            trigger="on_move",
                            condition="after_boost",
                            uses_per="round",
                            effect=MechanicalEffect(
                                movement_overrides=[
                                    MovementOverrideEffect(
                                        movement_modes=["boost"],
                                        override_type="fly",
                                        must_end_on_surface=True,
                                        duration="end_of_turn",
                                    )
                                ],
                            ),
                        )
                    ],
                ),
            ),
            TalentRank(
                rank=2,
                name="Juke",
                effects=MechanicalEffect(
                    movement_grants=[
                        MovementGrant(spaces=2, movement_type="fly", trigger="on_successful_agility_save"),
                    ],
                ),
            ),
            TalentRank(
                rank=3,
                name="Supersonic",
                effects=MechanicalEffect(
                    triggered_effects=[
                        TriggeredEffect(
                            trigger="on_move",
                            condition="after_move_8_plus",
                            uses_per="round",
                            effect=MechanicalEffect(
                                status_grants=[
                                    StatusGrant(
                                        status="invisible",
                                        target="self",
                                        duration="end_of_turn",
                                    )
                                ],
                            ),
                        )
                    ],
                ),
            ),
        ],
    ),
    TalentDefinition(
        id="bonded",
        name="BONDED",
        ranks=[
            TalentRank(
                rank=1,
                name="Bondmate",
                effects=MechanicalEffect(
                    accuracy_mods=[AccuracyModifier(value=1, applies_to="all", condition="can_see_bondmate")],
                    bondmates=[
                        BondmateEffect(
                            allowed_target_types=["pilot", "npc"],
                            can_change_between_missions=True,
                        )
                    ],
                ),
            ),
            TalentRank(
                rank=2,
                name="Coordinated Attack",
                effects=MechanicalEffect(
                    action_grants=[
                        ActionGrant(
                            action_type="reaction",
                            name="Coordinated Attack",
                            trigger="bondmate_hits_hostile",
                        ),
                    ],
                    accuracy_mods=[AccuracyModifier(value=1, applies_to="all", condition="coordinated_attack")],
                ),
            ),
            TalentRank(
                rank=3,
                name="Together, Unbreakable",
                effects=MechanicalEffect(
                    resistances=[Resistance(damage_type="all", condition="can_see_bondmate")],
                ),
            ),
        ],
    ),
    TalentDefinition(
        id="brutal",
        name="BRUTAL",
        ranks=[
            TalentRank(
                rank=1,
                name="Predator",
                effects=MechanicalEffect(
                    damage_mods=[
                        DamageModifier(
                            flat=1,
                            condition="melee_attack_and_target_prone",
                        )
                    ],
                ),
            ),
            TalentRank(
                rank=2,
                name="Grappling Hook",
                effects=MechanicalEffect(
                    action_grants=[
                        ActionGrant(
                            action_type="quick",
                            name="Grappling Hook",
                            trigger=None,
                        ),
                    ],
                    grapple_effects=[
                        GrappleEffect(
                            range=5,
                            pull_grappler_adjacent=True,
                            pull_target_adjacent=True,
                            condition="choose_pull_self_or_target",
                        )
                    ],
                ),
            ),
            TalentRank(
                rank=3,
                name="Executioner",
                effects=MechanicalEffect(
                    damage_mods=[
                        DamageModifier(
                            flat=2,
                            condition="melee_attack_and_target_below_half_hp",
                        )
                    ],
                ),
            ),
        ],
    ),
]


def get_talent_definition(talent_id: str) -> TalentDefinition | None:
    """Look up a talent definition by ID."""
    for talent in EXAMPLE_TALENTS:
        if talent.id == talent_id:
            return talent
    return None
