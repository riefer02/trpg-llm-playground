"""Mission generator for procedural mission creation.

Provides the `generate_missions` function that creates mission configurations
based on pilot license level and desired count.
"""

import random
from typing import Optional
from core.gm_toolkit.mission import MissionConfig, SitrepType, TerrainType
from core.shared.terrain_generation import TerrainConfig, generate_terrain_config
from core.gm_toolkit.encounter_builder import (
    EncounterDifficulty,
    build_enemy_force_preview,
)
from core.npc.compendium import get_templates_by_tier, get_all_special_classes
from core.npc.enums import NPCTier  # noqa: F401
from core.npc.models import NPCTemplate
from core.npc.special_classes import SpecialNPCTemplate
from .difficulty import (
    get_encounter_difficulty,
    get_encounter_difficulty_for_mission,
    get_npc_tier_distribution,
    get_mission_difficulty_stars,
)


def _get_available_npc_templates() -> list[NPCTemplate | SpecialNPCTemplate]:
    """Get a list of NPC templates for enemy force composition.

    Returns tier 1 and 2 templates plus special classes.
    """
    templates: list[NPCTemplate | SpecialNPCTemplate] = []
    for tier in ["tier_1", "tier_2"]:
        templates.extend(get_templates_by_tier(tier))  # type: ignore[arg-type]
    # Add special classes (they have tier field as well)
    templates.extend(get_all_special_classes())
    return templates


def _get_npc_templates_for_pilot_level(
    pilot_level: int,
) -> list[NPCTemplate | SpecialNPCTemplate]:
    """Get NPC templates filtered and sorted for given pilot level.

    Higher pilot levels include higher tier templates and prioritize
    them in the selection order.

    Args:
        pilot_level: Pilot license level (0-12)

    Returns:
        List of NPC templates sorted by tier descending (highest tier first)
    """
    # Determine which tiers to include based on pilot level
    tier_dist = get_npc_tier_distribution(pilot_level)
    included_tiers = [tier for tier, proportion in tier_dist.items() if proportion > 0]

    templates: list[NPCTemplate | SpecialNPCTemplate] = []
    for tier in included_tiers:
        templates.extend(get_templates_by_tier(tier))  # type: ignore[arg-type]

    # Add special classes (they have tier field as well)
    special_classes = get_all_special_classes()
    templates.extend(special_classes)

    # Sort by tier descending (tier_3 first, tier_1 last) to prioritize
    # stronger enemies for higher difficulty
    tier_order = {"tier_3": 3, "tier_2": 2, "tier_1": 1}
    templates.sort(key=lambda t: tier_order.get(t.tier, 0), reverse=True)

    return templates


def _map_difficulty_to_encounter_difficulty(difficulty: int) -> EncounterDifficulty:
    """Map integer difficulty (1-3) to EncounterDifficulty."""
    mapping: dict[int, EncounterDifficulty] = {
        1: "easy",
        2: "standard",
        3: "hard",
    }
    return mapping.get(difficulty, "standard")


# Available SITREP types (from acceptance criteria)
AVAILABLE_SITREPS: list[SitrepType] = [
    "control",
    "extract",  # Called "Extraction" in acceptance criteria
    "gauntlet",
    "hold_out",
    "recon",
]

# Terrain themes that match each SITREP type
SITREP_TERRAIN_MAPPING: dict[SitrepType, TerrainType] = {
    "control": "urban",
    "extract": "facility",
    "gauntlet": "desert",
    "hold_out": "forest",
    "recon": "space station",
}

# Mission name templates by SITREP
MISSION_NAME_TEMPLATES: dict[SitrepType, list[str]] = {
    "control": [
        "Operation Glass Hammer",
        "Sector Control",
        "Urban Pacification",
    ],
    "extract": [
        "Shadow Extraction",
        "VIP Recovery",
        "Asset Retrieval",
    ],
    "gauntlet": [
        "Gauntlet Run",
        "Canyon Assault",
        "Martian Convoy Intercept",
    ],
    "hold_out": [
        "Last Stand",
        "Bunker Defense",
        "Perimeter Hold",
    ],
    "recon": [
        "Silent Watch",
        "Deep Recon",
        "Intel Gathering",
    ],
}

# Briefing templates (simplified for MVP)
BRIEFING_TEMPLATES: dict[SitrepType, str] = {
    "control": (
        "Union intelligence reports hostile corporate forces have occupied a "
        "key location. Your mission is to disrupt their occupation and secure "
        "the central control node. Expect resistance from Striker-class mechs "
        "supported by Artillery units. Civilian infrastructure is at risk—"
        "minimize collateral damage where possible.\n\n"
        "Extraction will be available once the control node is secured. "
        "Keep comms open for updates on enemy reinforcements."
    ),
    "extract": (
        "A high-value Union scientist has been captured by hostile forces. "
        "He is being held in a secure facility. Your mission is to infiltrate, "
        "locate the VIP, and escort him to the extraction point. "
        "The facility is heavily guarded with automated turrets and patrol mechs.\n\n"
        "The VIP's safe extraction takes priority over enemy elimination. "
        "Use stealth and speed to avoid prolonged engagements."
    ),
    "gauntlet": (
        "Intel suggests a large pirate convoy is moving through a dangerous "
        "canyon network carrying stolen Union technology. Your mission is to "
        "intercept and destroy the convoy while surviving waves of enemy forces.\n\n"
        "You will be deployed at the canyon entrance and must fight your way "
        "through multiple waves. The terrain provides limited cover but numerous "
        "choke points. Enemy reinforcements will arrive at timed intervals.\n\n"
        "This is a high-intensity combat scenario. Ammunition and heat management "
        "will be critical. Extraction is only available at the far end."
    ),
    "hold_out": (
        "Hostile forces are advancing on a critical Union outpost. "
        "Your mission is to defend the position until reinforcements arrive.\n\n"
        "The outpost provides defensive cover but is vulnerable to flanking. "
        "Enemy waves will attempt to overwhelm your position with combined arms. "
        "Hold the line for six rounds to achieve mission success."
    ),
    "recon": (
        "A suspected enemy base has been detected in a remote sector. "
        "Your mission is to infiltrate, gather intelligence, and exfiltrate "
        "without being detected.\n\n"
        "Stealth is paramount. Avoid direct engagement where possible. "
        "Use sensor suites to map enemy positions and identify key targets. "
        "Extraction will be available at the rally point once intel is secured."
    ),
}

# Objective templates
OBJECTIVE_TEMPLATES: dict[SitrepType, list[str]] = {
    "control": [
        "Secure the central control node",
        "Neutralize all hostile mechs",
        "Minimize collateral damage (optional)",
    ],
    "extract": [
        "Locate and extract the VIP",
        "Reach extraction point within time limit",
        "Disable security systems (optional)",
    ],
    "gauntlet": [
        "Survive all enemy waves",
        "Destroy the convoy command vehicle",
        "Reach extraction point at canyon exit",
    ],
    "hold_out": [
        "Defend the outpost for six rounds",
        "Prevent enemy from reaching the command center",
        "Maintain at least 50% structural integrity (optional)",
    ],
    "recon": [
        "Gather intelligence from three data points",
        "Avoid detection (no alarms triggered)",
        "Exfiltrate to extraction point",
    ],
}

# Enemy intel templates
ENEMY_INTEL_TEMPLATES: dict[SitrepType, str] = {
    "control": (
        "Enemy force consists of 4 mechs: 2 Strikers (Assault-class frames), "
        "1 Defender (Bulwark frame), and 1 Artillery (Longshot frame). "
        "Threat assessment: Moderate. The Defender will attempt to hold the "
        "control node while Strikers flank. Artillery provides long-range support "
        "from elevated positions."
    ),
    "extract": (
        "6 enemy mechs: 3 Skirmishers (Swift-class frames), 2 Defenders (Guardian frames), "
        "and 1 Commander (Overseer frame). Threat assessment: High. Skirmishers are fast "
        "and will attempt to flank. The Commander provides tactical coordination and can "
        "call reinforcements. Automated turrets cover key corridors."
    ),
    "gauntlet": (
        "8+ enemy mechs in waves: 4 Strikers (varied frames), 2 Artillery (Siege-class), "
        "1 Commander (Tactician frame), and 1 Boss (Heavy-class). Threat assessment: Severe. "
        "Waves are coordinated and will attempt to overwhelm with combined arms. "
        "The Boss mech is heavily armored and requires focused fire."
    ),
    "hold_out": (
        "6 enemy mechs: 2 Strikers, 2 Artillery, 1 Defender, 1 Commander. "
        "Threat assessment: High. Enemy will attack in coordinated waves, focusing "
        "on breaching defensive positions. Artillery will provide suppressing fire "
        "from a distance."
    ),
    "recon": (
        "4 enemy mechs: 2 Scouts (Recon-class frames), 1 Striker, 1 Defender. "
        "Threat assessment: Moderate. Scouts patrol the perimeter and can raise alarms. "
        "Striker and Defender protect the central compound. Avoid prolonged engagement."
    ),
}


def generate_missions(
    pilot_level: int,
    count: int = 3,
    seed: Optional[int] = None,
) -> list[MissionConfig]:
    """Generate mission configurations based on pilot license level.

    Args:
        pilot_level: Pilot's license level (0-12). Affects difficulty.
        count: Number of missions to generate (default 3).
        seed: Optional random seed for reproducible generation.

    Returns:
        List of MissionConfig objects with unique SITREP types.

    Raises:
        ValueError: If count > available SITREP types (max 5).
    """
    if count > len(AVAILABLE_SITREPS):
        raise ValueError(
            f"Cannot generate {count} missions with only {len(AVAILABLE_SITREPS)} "
            "available SITREP types."
        )

    if seed is not None:
        random.seed(seed)

    # Select distinct SITREP types
    selected_sitreps = random.sample(AVAILABLE_SITREPS, k=count)

    # Get NPC templates filtered and sorted for pilot level
    npc_templates = _get_npc_templates_for_pilot_level(pilot_level)

    missions = []
    for idx, sitrep in enumerate(selected_sitreps):
        # Determine difficulty stars (1-3) for display
        difficulty = get_mission_difficulty_stars(pilot_level, idx)

        # Get encounter difficulty (trivial → extreme) based on pilot level
        # and mission index (later missions are slightly harder)
        encounter_difficulty = get_encounter_difficulty_for_mission(pilot_level, idx)

        # Build enemy force preview using filtered NPC templates
        enemy_force_preview = build_enemy_force_preview(
            difficulty=encounter_difficulty,
            sitrep_type=sitrep,
            player_count=1,  # Single-player
            avg_license_level=pilot_level,
            npc_templates=npc_templates,
        )

        # Total enemy count from preview
        enemy_count = (
            enemy_force_preview.initial_count + enemy_force_preview.reserve_count
        )

        # Terrain based on SITREP mapping
        terrain = SITREP_TERRAIN_MAPPING[sitrep]

        # Select random name template
        name = random.choice(MISSION_NAME_TEMPLATES[sitrep])

        mission = MissionConfig(
            id=f"mission_{idx + 1}_{sitrep}",
            name=name,
            difficulty=difficulty,
            sitrep=sitrep,
            terrain=terrain,
            enemy_count=enemy_count,
            description=_generate_description(sitrep, name),
            briefing=BRIEFING_TEMPLATES[sitrep],
            objectives=OBJECTIVE_TEMPLATES[sitrep],
            enemy_intel=ENEMY_INTEL_TEMPLATES[sitrep],
            map_preview_url=_generate_map_preview_url(terrain),
            enemy_force_preview=enemy_force_preview,
        )
        missions.append(mission)

    return missions


def _calculate_enemy_count(difficulty: int, sitrep: SitrepType) -> int:
    """Calculate enemy count based on difficulty and SITREP type."""
    base_counts = {
        "control": 4,
        "extract": 6,
        "gauntlet": 8,
        "hold_out": 6,
        "recon": 4,
    }
    base = base_counts.get(sitrep, 4)
    # Scale with difficulty (1-3)
    return base + (difficulty - 1) * 2


def _generate_description(sitrep: SitrepType, name: str) -> str:
    """Generate a short mission description."""
    descriptions = {
        "control": "Secure the central plaza from hostile forces.",
        "extract": "Infiltrate the research facility and extract the VIP.",
        "gauntlet": "Survive waves of enemies while traversing the canyon.",
        "hold_out": "Defend the outpost against waves of attackers.",
        "recon": "Gather intelligence on enemy movements without detection.",
    }
    return descriptions.get(sitrep, f"Complete the {name} mission.")


def _generate_map_preview_url(terrain: TerrainType) -> str:
    """Generate a placeholder map preview URL based on terrain."""
    terrain_slug = terrain.replace(" ", "+")
    return f"https://placehold.co/600x400/1e293b/94a3b8?text={terrain_slug}+Map"


def generate_terrain(sitrep: str, theme: str) -> TerrainConfig:
    """Generate terrain configuration for given SITREP type and theme.

    This is the public API function required by the PRD acceptance criteria.

    Args:
        sitrep: SITREP type (e.g., "control", "extract")
        theme: Terrain theme (e.g., "urban", "forest", "desert", "facility", "space station")

    Returns:
        TerrainConfig with generated terrain map and metadata
    """
    return generate_terrain_config(sitrep_type=sitrep, theme=theme)
