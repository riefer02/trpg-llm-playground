"""HTML template rendering for campaign briefing PDF exports."""

from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template
from typing import Iterable

from core.shared.campaign.campaign import Campaign
from core.shared.campaign.serialization import get_campaign_summary

_TEMPLATE = Template(
    (Path(__file__).parent / "templates" / "campaign_brief.html").read_text()
)


def render_campaign_brief_pdf(campaign: Campaign) -> bytes:
    """Render a campaign briefing PDF using WeasyPrint."""
    from weasyprint import HTML

    html = render_campaign_brief_html(campaign)
    return HTML(string=html).write_pdf()


def render_campaign_brief_html(campaign: Campaign) -> str:
    """Render the HTML for a campaign briefing."""
    identity = campaign.identity
    lobby = campaign.lobby_state
    mission_plan = lobby.mission_plan if lobby else None
    summary = get_campaign_summary(campaign)

    objectives = []
    if mission_plan and mission_plan.objectives:
        objectives = [
            f"{obj.title} ({obj.priority}) - {obj.success_condition}"
            for obj in mission_plan.objectives
        ]

    reserves = []
    if mission_plan and mission_plan.reserves:
        for reserve in mission_plan.reserves:
            assignment = (
                f" assigned to {reserve.assigned_character_id}"
                if reserve.assigned_character_id
                else ""
            )
            reserves.append(f"{reserve.reserve_id} ({reserve.status}){assignment}")

    mission_history_rows = _render_mission_history(campaign)

    return _TEMPLATE.safe_substitute(
        campaign_name=_safe_text(campaign.name),
        campaign_description=_safe_text(campaign.description or "No description"),
        squad_name=_safe_text(
            identity.squad_name if identity and identity.squad_name else "None"
        ),
        patron=_safe_text(identity.patron if identity and identity.patron else "None"),
        who_we_are=_safe_text(
            identity.who_we_are if identity and identity.who_we_are else "None"
        ),
        relationships=_render_list(identity.relationships if identity else []),
        themes=_render_list(identity.themes if identity else []),
        gm_prompts=_render_list(identity.gm_prompts if identity else []),
        lobby_status=_safe_text(lobby.status if lobby else "None"),
        mission_name=_safe_text(mission_plan.mission_name if mission_plan else "None"),
        mission_briefing=_safe_text(
            mission_plan.briefing_notes if mission_plan and mission_plan.briefing_notes else "None"
        ),
        stakes_type=_safe_text(
            mission_plan.stakes.stakes_type if mission_plan and mission_plan.stakes else "None"
        ),
        stakes_summary=_safe_text(
            mission_plan.stakes.summary if mission_plan and mission_plan.stakes else "None"
        ),
        objectives=_render_list(objectives),
        support_assets=_render_list(
            mission_plan.support_assets if mission_plan else []
        ),
        reserves=_render_list(reserves),
        min_pilots=str(lobby.min_pilot_count) if lobby else "-",
        preferred_pilots=str(lobby.preferred_pilot_count) if lobby else "-",
        total_missions=str(summary["total_missions"]),
        successful_missions=str(summary["successful_missions"]),
        partial_missions=str(summary["partial_missions"]),
        failed_missions=str(summary["failed_missions"]),
        average_completion=str(summary["average_completion"]),
        last_outcome=_safe_text(summary.get("last_outcome") or "None"),
        last_mission_name=_safe_text(summary.get("last_mission_name") or "None"),
        last_mission_date=_safe_text(summary.get("last_mission_date") or "None"),
        mission_history_rows=mission_history_rows,
    )


def _safe_text(value: str) -> str:
    return escape(value)


def _render_list(items: Iterable[str]) -> str:
    values = [escape(item) for item in items if item]
    if not values:
        return '<div class="empty">None</div>'
    list_items = "\n".join(f"<li>{item}</li>" for item in values)
    return f"<ul>{list_items}</ul>"


def _render_mission_history(campaign: Campaign) -> str:
    if not campaign.mission_history:
        return (
            '<tr><td class="empty" colspan="4">'
            "No mission outcomes recorded."
            "</td></tr>"
        )
    rows = []
    for record in campaign.mission_history:
        rows.append(
            "<tr>"
            f"<td>{escape(record.mission_name)}</td>"
            f"<td>{escape(record.outcome)}</td>"
            f"<td>{record.completion_score:.2f}</td>"
            f"<td>{record.mission_date.isoformat()}</td>"
            "</tr>"
        )
    return "\n".join(rows)
