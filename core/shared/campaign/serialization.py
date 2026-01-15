"""Campaign serialization utilities for Lancer TTRPG.

Provides functions for saving and loading campaign state to/from JSON,
and validating campaign data integrity.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from core.shared.campaign.campaign import Campaign


class CampaignValidationError(ValueError):
    """Raised when campaign validation fails."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


def save_campaign(
    campaign: Campaign,
    path: str,
    *,
    indent: int = 2,
) -> None:
    """Save a campaign to a JSON file.

    Args:
        campaign: The campaign to save
        path: File path to write to
        indent: JSON indentation level (default 2)
    """
    data = campaign_to_dict(campaign)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_campaign_to_string(campaign: Campaign, *, indent: int = 2) -> str:
    """Serialize a campaign to a JSON string.

    Args:
        campaign: The campaign to serialize
        indent: JSON indentation level (default 2)

    Returns:
        JSON string representation of the campaign
    """
    data = campaign_to_dict(campaign)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def load_campaign(path: str) -> Campaign:
    """Load a campaign from a JSON file.

    Args:
        path: File path to read from

    Returns:
        The loaded Campaign

    Raises:
        CampaignValidationError: If validation fails
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict_to_campaign(data)


def load_campaign_from_string(data: str) -> Campaign:
    """Parse a campaign from a JSON string.

    Args:
        data: JSON string to parse

    Returns:
        The loaded Campaign

    Raises:
        CampaignValidationError: If validation fails
        json.JSONDecodeError: If the string contains invalid JSON
    """
    parsed = json.loads(data)
    return dict_to_campaign(parsed)


def campaign_to_dict(campaign: Campaign) -> dict[str, Any]:
    """Convert a Campaign to a JSON-serializable dictionary.

    Args:
        campaign: The campaign to convert

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    return campaign.model_dump(mode="json")


def dict_to_campaign(data: dict[str, Any]) -> Campaign:
    """Convert a dictionary to a Campaign.

    Args:
        data: Dictionary to convert

    Returns:
        The constructed Campaign

    Raises:
        CampaignValidationError: If validation fails
    """
    validate_campaign_dict(data)
    return Campaign.model_validate(data)


def validate_campaign(data: dict[str, Any]) -> list[str]:
    """Validate campaign data and return a list of errors.

    Args:
        data: Campaign dictionary to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        errors.append("Campaign data must be a dictionary")
        return errors

    if "id" not in data:
        errors.append("Campaign must have an 'id' field")
    if "name" not in data:
        errors.append("Campaign must have a 'name' field")

    characters_key = None
    if "characters" in data:
        characters_key = "characters"
    elif "pilots" in data:
        characters_key = "pilots"

    if characters_key and not isinstance(data[characters_key], list):
        errors.append(f"'{characters_key}' must be a list")
    elif characters_key:
        for i, character in enumerate(data[characters_key]):
            if not isinstance(character, dict):
                errors.append(f"Character {i} must be a dictionary")
            elif "id" not in character:
                errors.append(f"Character {i} is missing 'id' field")

    links_key = None
    if "character_mech_links" in data:
        links_key = "character_mech_links"
    elif "pilot_mech_links" in data:
        links_key = "pilot_mech_links"

    if links_key and not isinstance(data[links_key], list):
        errors.append(f"'{links_key}' must be a list")
    elif links_key:
        characters = data.get(characters_key or "characters", [])
        character_ids = {
            c.get("id") for c in characters if isinstance(c, dict)
        }
        for i, link in enumerate(data[links_key]):
            if not isinstance(link, dict):
                errors.append(f"Character mech link {i} must be a dictionary")
                continue
            link_character_id = link.get("character_id") or link.get("pilot_id")
            if link_character_id is None:
                errors.append(
                    f"Character mech link {i} is missing 'character_id' field"
                )
            elif link_character_id not in character_ids:
                errors.append(
                    "Character mech link "
                    f"{i} references unknown character: {link_character_id}"
                )

    if "sessions" in data and not isinstance(data["sessions"], list):
        errors.append("'sessions' must be a list")
    elif "sessions" in data:
        for i, session in enumerate(data["sessions"]):
            if not isinstance(session, dict):
                errors.append(f"Session {i} must be a dictionary")
            elif "id" not in session:
                errors.append(f"Session {i} is missing 'id' field")
            elif "session_number" not in session:
                errors.append(f"Session {i} is missing 'session_number' field")

    if "mission_history" in data and not isinstance(data["mission_history"], list):
        errors.append("'mission_history' must be a list")

    return errors


def validate_campaign_dict(data: dict[str, Any]) -> None:
    """Validate and construct a Campaign from a dictionary.

    Args:
        data: Campaign dictionary to validate

    Raises:
        CampaignValidationError: If validation fails
    """
    errors = validate_campaign(data)
    if errors:
        raise CampaignValidationError("Campaign validation failed", errors)


def validate_campaign_synchronous(
    campaign: Campaign,
) -> list[str]:
    """Validate an existing Campaign object.

    Args:
        campaign: The campaign to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors: list[str] = []

    character_ids = {c["id"] for c in campaign.characters}

    for link in campaign.character_mech_links:
        if link.character_id not in character_ids:
            errors.append(
                f"Character mech link references unknown character: {link.character_id}"
            )

    for session in campaign.sessions:
        for mission in session.active_missions:
            for character_id in mission.participating_character_ids:
                if character_id not in character_ids:
                    errors.append(
                        f"Session mission references unknown character: {character_id}"
                    )

    for record in campaign.mission_history:
        for character_id in record.participating_character_ids:
            if character_id not in character_ids:
                errors.append(
                    f"Mission history references unknown character: {character_id}"
                )

    return errors


def is_campaign_valid(campaign: Campaign) -> bool:
    """Check if a campaign is valid.

    Args:
        campaign: The campaign to check

    Returns:
        True if the campaign is valid, False otherwise
    """
    return len(validate_campaign_synchronous(campaign)) == 0


def get_campaign_summary(campaign: Campaign) -> dict[str, Any]:
    """Generate a summary of a campaign for display.

    Args:
        campaign: The campaign to summarize

    Returns:
        Dictionary with campaign summary information
    """
    total_missions = len(campaign.mission_history)
    successful_missions = sum(
        1 for r in campaign.mission_history if r.outcome == "success"
    )
    partial_missions = sum(
        1 for r in campaign.mission_history if r.outcome == "partial"
    )
    failed_missions = sum(
        1 for r in campaign.mission_history if r.outcome in ("failure", "catastrophic")
    )

    avg_completion = (
        sum(r.completion_score for r in campaign.mission_history) / total_missions
        if total_missions > 0
        else 0.0
    )

    last_record = campaign.mission_history[-1] if campaign.mission_history else None
    last_outcome = last_record.outcome if last_record else None
    last_mission_name = last_record.mission_name if last_record else None
    last_mission_date = (
        last_record.mission_date.isoformat() if last_record else None
    )

    def _pilot_data(character: dict) -> dict:
        if not isinstance(character, dict):
            return {}
        pilot = character.get("pilot", character)
        return pilot if isinstance(pilot, dict) else {}

    living_characters = sum(
        1
        for character in campaign.characters
        if not _pilot_data(character).get("is_dead", False)
    )
    cloned_characters = 0
    for character in campaign.characters:
        pilot_data = _pilot_data(character)
        clone_state = pilot_data.get("clone_state") or {}
        status = clone_state.get("status") or {}
        if status.get("times_cloned", 0) > 0:
            cloned_characters += 1

    return {
        "id": campaign.id,
        "name": campaign.name,
        "session_count": len(campaign.sessions),
        "character_count": len(campaign.characters),
        "living_characters": living_characters,
        "cloned_characters": cloned_characters,
        "total_missions": total_missions,
        "successful_missions": successful_missions,
        "partial_missions": partial_missions,
        "failed_missions": failed_missions,
        "average_completion": round(avg_completion, 2),
        "last_outcome": last_outcome,
        "last_mission_name": last_mission_name,
        "last_mission_date": last_mission_date,
        "mech_assignments": len(campaign.character_mech_links),
        "created_at": campaign.created_at.isoformat(),
        "modified_at": campaign.modified_at.isoformat(),
    }
