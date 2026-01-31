"""Mission configuration models for procedural mission generation.

Provides type-safe primitives for mission configuration that can be used
by both the mission generator (LLM pipeline) and the game frontend.
"""

from typing import Literal, Optional
from pydantic import Field
from core.shared.models import FrozenModel
from core.gm_toolkit.encounter_builder import EnemyForcePreview

# Terrain types matching frontend Mission.terrain
TerrainType = Literal[
    "urban",
    "forest",
    "desert",
    "facility",
    "space station",
]

# SITREP types from core/shared/scenario.py SitrepType
SitrepType = Literal[
    "control",
    "escort",
    "extract",
    "gauntlet",
    "hold_out",
    "recon",
]


class MissionConfig(FrozenModel):
    """Complete mission configuration for procedural generation.

    This model contains all data needed to:
    1. Display mission briefing in frontend
    2. Generate enemy force composition via encounter builder
    3. Configure terrain and SITREP-specific rules
    4. Launch combat scenario

    Attributes:
        id: Unique identifier for the mission
        name: Display name (e.g., "Operation Glass Hammer")
        difficulty: Difficulty rating (1-3 stars)
        sitrep: SITREP type (control, extract, gauntlet, etc.)
        terrain: Terrain type matching narrative theme
        enemy_count: Total number of enemy mechs (for display)
        description: Short mission description (1-2 sentences)
        briefing: 2-3 paragraph mission briefing text
        objectives: List of primary and secondary objectives
        enemy_intel: Text description of enemy composition and threat assessment
        map_preview_url: Optional URL to map preview image (placeholder)
        enemy_force_preview: Optional enemy force composition preview
    """

    id: str = Field(..., description="Unique mission identifier")
    name: str = Field(..., description="Display name")
    difficulty: int = Field(..., ge=1, le=3, description="Difficulty stars (1-3)")
    sitrep: SitrepType = Field(..., description="SITREP type")
    terrain: TerrainType = Field(..., description="Terrain theme")
    enemy_count: int = Field(..., ge=1, description="Total enemy count")
    description: Optional[str] = Field(None, description="Short description")
    briefing: str = Field(..., description="2-3 paragraph mission briefing")
    objectives: list[str] = Field(
        default_factory=list, description="Mission objectives"
    )
    enemy_intel: str = Field(..., description="Enemy composition and threat assessment")
    map_preview_url: Optional[str] = Field(None, description="Map preview image URL")
    enemy_force_preview: Optional[EnemyForcePreview] = Field(
        None, description="Enemy force composition preview (calculated)"
    )
