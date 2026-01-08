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

    if "pilots" in data and not isinstance(data["pilots"], list):
        errors.append("'pilots' must be a list")
    elif "pilots" in data:
        for i, pilot in enumerate(data["pilots"]):
            if not isinstance(pilot, dict):
                errors.append(f"Pilot {i} must be a dictionary")
            elif "id" not in pilot:
                errors.append(f"Pilot {i} is missing 'id' field")

    if "pilot_mech_links" in data and not isinstance(data["pilot_mech_links"], list):
        errors.append("'pilot_mech_links' must be a list")
    elif "pilot_mech_links" in data:
        pilot_ids = {p.get("id") for p in data.get("pilots", [])}
        for i, link in enumerate(data["pilot_mech_links"]):
            if not isinstance(link, dict):
                errors.append(f"Pilot mech link {i} must be a dictionary")
            elif "pilot_id" not in link:
                errors.append(f"Pilot mech link {i} is missing 'pilot_id' field")
            elif link.get("pilot_id") not in pilot_ids:
                errors.append(
                    f"Pilot mech link {i} references unknown pilot: {link['pilot_id']}"
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

    pilot_ids = {p["id"] for p in campaign.pilots}

    for link in campaign.pilot_mech_links:
        if link.pilot_id not in pilot_ids:
            errors.append(f"Pilot mech link references unknown pilot: {link.pilot_id}")

    for session in campaign.sessions:
        for mission in session.active_missions:
            for pilot_id in mission.participating_pilot_ids:
                if pilot_id not in pilot_ids:
                    errors.append(
                        f"Session mission references unknown pilot: {pilot_id}"
                    )

    for record in campaign.mission_history:
        for pilot_id in record.participating_pilot_ids:
            if pilot_id not in pilot_ids:
                errors.append(f"Mission history references unknown pilot: {pilot_id}")

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

    living_pilots = sum(1 for p in campaign.pilots if not p.get("is_dead", False))
    cloned_pilots = sum(
        1
        for p in campaign.pilots
        if p.get("clone_state") is not None and p["clone_state"].get("is_cloned", False)
    )

    return {
        "id": campaign.id,
        "name": campaign.name,
        "session_count": len(campaign.sessions),
        "pilot_count": len(campaign.pilots),
        "living_pilots": living_pilots,
        "cloned_pilots": cloned_pilots,
        "total_missions": total_missions,
        "successful_missions": successful_missions,
        "partial_missions": partial_missions,
        "failed_missions": failed_missions,
        "average_completion": round(avg_completion, 2),
        "mech_assignments": len(campaign.pilot_mech_links),
        "created_at": campaign.created_at.isoformat(),
        "modified_at": campaign.modified_at.isoformat(),
    }
