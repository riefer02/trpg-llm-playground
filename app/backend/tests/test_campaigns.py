"""Campaign API integration tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_reserve_templates(client: AsyncClient) -> None:
    """Test listing all reserve templates from PR2 tables."""
    auth_headers = {"X-User-Id": "reserve_user"}

    response = await client.get(
        "/api/campaigns/reserve-templates",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 30  # 10 narrative + 10 mech + 10 tactical
    assert len(data["items"]) == 30
    assert all("id" in item for item in data["items"])
    assert all("name" in item for item in data["items"])
    assert all("reserve_type" in item for item in data["items"])
    assert all("description" in item for item in data["items"])


@pytest.mark.asyncio
async def test_filter_templates_by_category(client: AsyncClient) -> None:
    """Test filtering reserve templates by category."""
    auth_headers = {"X-User-Id": "reserve_user"}

    # Test tactical filter
    response = await client.get(
        "/api/campaigns/reserve-templates?category=tactical",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 10
    assert all(item["reserve_type"] == "tactical" for item in data["items"])

    # Test mech filter
    response = await client.get(
        "/api/campaigns/reserve-templates?category=mech",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 10
    assert all(item["reserve_type"] == "mech" for item in data["items"])

    # Test narrative filter
    response = await client.get(
        "/api/campaigns/reserve-templates?category=narrative",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 10
    assert all(item["reserve_type"] == "narrative" for item in data["items"])


@pytest.mark.asyncio
async def test_campaign_invite_and_character_flow(client: AsyncClient) -> None:
    owner_headers = {"X-User-Id": "owner_user"}
    pilot_headers = {"X-User-Id": "pilot_user"}

    # Create campaign
    create_resp = await client.post(
        "/api/campaigns",
        json={"name": "Alpha Company", "description": "First deployment"},
        headers=owner_headers,
    )
    assert create_resp.status_code == 201
    campaign = create_resp.json()
    campaign_id = campaign["id"]
    assert campaign["name"] == "Alpha Company"
    assert len(campaign["members"]) == 1

    # Owner listing should show campaign
    list_resp = await client.get("/api/campaigns", headers=owner_headers)
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] == 1
    assert list_data["items"][0]["membership_role"] == "owner"
    assert list_data["items"][0]["mission_summary"]["total_missions"] == 0

    # Issue invite
    invite_resp = await client.post(
        f"/api/campaigns/{campaign_id}/invites",
        json={"role": "player"},
        headers=owner_headers,
    )
    assert invite_resp.status_code == 200
    invite_token = invite_resp.json()["token"]

    # Accept invite as pilot_user
    accept_resp = await client.post(
        f"/api/campaigns/invites/{invite_token}/accept",
        headers=pilot_headers,
    )
    assert accept_resp.status_code == 200
    campaign_after_invite = accept_resp.json()
    assert len(campaign_after_invite["members"]) == 2

    # Create a character as the pilot
    char_resp = await client.post(
        "/api/characters",
        json={"callsign": "RANGER"},
        headers=pilot_headers,
    )
    assert char_resp.status_code == 201
    character_id = char_resp.json()["id"]
    assert char_resp.json()["campaign_ids"] == []

    # Attach character to campaign
    attach_resp = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={"character_id": character_id},
        headers=pilot_headers,
    )
    assert attach_resp.status_code == 200
    attached_detail = attach_resp.json()
    assert any(c["character_id"] == character_id for c in attached_detail["characters"])
    assert "readiness_summary" in attached_detail
    assert "member_issues" in attached_detail["readiness_summary"]
    assert attached_detail["seat_warning"] is None

    # Pilot marks themselves ready
    member_id = next(
        m["id"] for m in attached_detail["members"] if m["user_id"] == "pilot_user"
    )
    ready_resp = await client.post(
        f"/api/campaigns/{campaign_id}/members/{member_id}/settings",
        json={"ready": True},
        headers=pilot_headers,
    )
    assert ready_resp.status_code == 200
    assert ready_resp.json()["ready_state"] == "ready"

    # Character detail should reflect campaign linkage
    char_detail = await client.get(
        f"/api/characters/{character_id}", headers=pilot_headers
    )
    assert char_detail.status_code == 200
    assert campaign_id in char_detail.json()["campaign_ids"]


@pytest.mark.asyncio
async def test_campaign_invite_preview_and_management(client: AsyncClient) -> None:
    owner_headers = {"X-User-Id": "invite_mgr"}
    viewer_headers = {"X-User-Id": "invite_viewer"}

    create_resp = await client.post(
        "/api/campaigns",
        json={"name": "Preview Squad"},
        headers=owner_headers,
    )
    campaign_id = create_resp.json()["id"]

    invite_resp = await client.post(
        f"/api/campaigns/{campaign_id}/invites",
        json={"role": "player", "invite_note": "Bring LL0-ready builds."},
        headers=owner_headers,
    )
    invite = invite_resp.json()
    assert invite["invite_note"] == "Bring LL0-ready builds."

    preview_resp = await client.get(
        f"/api/campaigns/invites/{invite['token']}/preview",
        headers=viewer_headers,
    )
    assert preview_resp.status_code == 200
    preview_data = preview_resp.json()
    assert preview_data["campaign_name"] == "Preview Squad"
    assert preview_data["status"] == "pending"
    assert preview_data["invite_note"] == "Bring LL0-ready builds."
    assert "readiness_issues" in preview_data
    assert preview_data["min_pilots"] >= 1

    revoke_resp = await client.post(
        f"/api/campaigns/{campaign_id}/invites/{invite['id']}/revoke",
        headers=owner_headers,
    )
    assert revoke_resp.status_code == 200
    assert revoke_resp.json()["status"] == "revoked"

    resend_resp = await client.post(
        f"/api/campaigns/{campaign_id}/invites/{invite['id']}/resend",
        json={"expires_in_hours": 2},
        headers=owner_headers,
    )
    assert resend_resp.status_code == 200
    resend_data = resend_resp.json()
    assert resend_data["status"] == "pending"
    assert resend_data["expires_at"] is not None


@pytest.mark.asyncio
async def test_export_campaign_pdf(client: AsyncClient) -> None:
    pytest.importorskip("weasyprint")
    owner_headers = {"X-User-Id": "pdf_owner"}

    create_resp = await client.post(
        "/api/campaigns",
        json={"name": "PDF Campaign"},
        headers=owner_headers,
    )
    campaign_id = create_resp.json()["id"]

    export_resp = await client.get(
        f"/api/campaigns/{campaign_id}/export.pdf",
        headers=owner_headers,
    )
    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"].startswith("application/pdf")


@pytest.mark.asyncio
async def test_attach_requires_membership(client: AsyncClient) -> None:
    owner_headers = {"X-User-Id": "campaign_owner"}
    outsider_headers = {"X-User-Id": "outsider"}

    campaign_resp = await client.post(
        "/api/campaigns",
        json={"name": "Bravo"},
        headers=owner_headers,
    )
    campaign_id = campaign_resp.json()["id"]

    char_resp = await client.post(
        "/api/characters",
        json={"callsign": "LONE"},
        headers=outsider_headers,
    )
    character_id = char_resp.json()["id"]

    attach_resp = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={"character_id": character_id},
        headers=outsider_headers,
    )
    assert attach_resp.status_code == 403


@pytest.mark.asyncio
async def test_campaign_lobby_launch_flow(client: AsyncClient) -> None:
    owner_headers = {"X-User-Id": "owner_lobby"}
    pilot_headers = {"X-User-Id": "pilot_lobby"}

    # Create campaign and update identity
    create_resp = await client.post(
        "/api/campaigns",
        json={"name": "Echo Company"},
        headers=owner_headers,
    )
    assert create_resp.status_code == 201
    campaign_id = create_resp.json()["id"]

    identity_resp = await client.post(
        f"/api/campaigns/{campaign_id}/identity",
        json={"squad_name": "Echo", "patron": "Union Navy"},
        headers=owner_headers,
    )
    assert identity_resp.status_code == 200
    assert identity_resp.json()["data"]["identity"]["patron"] == "Union Navy"

    # Invite and accept pilot
    invite_resp = await client.post(
        f"/api/campaigns/{campaign_id}/invites",
        json={"role": "player"},
        headers=owner_headers,
    )
    invite_token = invite_resp.json()["token"]

    accept_resp = await client.post(
        f"/api/campaigns/invites/{invite_token}/accept",
        headers=pilot_headers,
    )
    assert accept_resp.status_code == 200

    # Pilot character setup
    char_resp = await client.post(
        "/api/characters",
        json={"callsign": "FLARE"},
        headers=pilot_headers,
    )
    character_id = char_resp.json()["id"]

    attach_resp = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={"character_id": character_id},
        headers=pilot_headers,
    )
    member_id = next(
        m["id"] for m in attach_resp.json()["members"] if m["user_id"] == "pilot_lobby"
    )

    ready_resp = await client.post(
        f"/api/campaigns/{campaign_id}/members/{member_id}/settings",
        json={"ready": True, "assigned_character_id": character_id},
        headers=pilot_headers,
    )
    assert ready_resp.status_code == 200

    # Configure lobby with preferred pilot count of 1 (to trigger cap warning on extra invites)
    lobby_resp = await client.post(
        f"/api/campaigns/{campaign_id}/lobby",
        json={
            "mission_name": "Operation Glass",
            "assigned_member_ids": [member_id],
            "preferred_pilot_count": 1,
            "min_pilot_count": 1,
            "objectives": [
                {
                    "id": "obj-1",
                    "title": "Secure the relay",
                    "success_condition": "Hold the point for 3 rounds",
                }
            ],
            "stakes": {
                "stakes_type": "personal",
                "summary": "Keep the colony online",
            },
        },
        headers=owner_headers,
    )
    assert lobby_resp.status_code == 200

    # Attempting another player invite should now hit the soft cap
    cap_resp = await client.post(
        f"/api/campaigns/{campaign_id}/invites",
        json={"role": "player"},
        headers=owner_headers,
    )
    assert cap_resp.status_code == 409

    # Launch mission
    launch_resp = await client.post(
        f"/api/campaigns/{campaign_id}/launch",
        json={"environment": "standard"},
        headers=owner_headers,
    )
    assert launch_resp.status_code == 200
    launch_data = launch_resp.json()
    assert launch_data["data"]["lobby_state"]["status"] == "launched"
    assert launch_data["readiness_summary"]["can_launch"] is True
    assert "issues" in launch_data["readiness_summary"]
    assert any(
        "Mission already launched" in issue
        for issue in launch_data["readiness_summary"]["issues"]
    )
    assert launch_data["data"]["lobby_state"]["combat_session_id"] is not None

    # Update lifecycle phase for the newly created session
    session_id = launch_data["data"]["sessions"][-1]["id"]
    lifecycle_resp = await client.post(
        f"/api/campaigns/{campaign_id}/sessions/{session_id}/lifecycle",
        json={"phase": "mission", "status": "complete", "summary": "Relayed"},
        headers=owner_headers,
    )
    assert lifecycle_resp.status_code == 200
    lifecycle_data = lifecycle_resp.json()["data"]["sessions"][-1]
    mission_checkpoint = next(
        c for c in lifecycle_data["lifecycle_checkpoints"] if c["phase"] == "mission"
    )
    assert mission_checkpoint["status"] == "complete"


@pytest.mark.asyncio
async def test_campaign_session_outcome_updates_history(client: AsyncClient) -> None:
    owner_headers = {"X-User-Id": "outcome_owner"}
    pilot_headers = {"X-User-Id": "outcome_pilot"}

    create_resp = await client.post(
        "/api/campaigns",
        json={"name": "Outcome Co"},
        headers=owner_headers,
    )
    campaign_id = create_resp.json()["id"]

    invite_resp = await client.post(
        f"/api/campaigns/{campaign_id}/invites",
        json={"role": "player"},
        headers=owner_headers,
    )
    invite_token = invite_resp.json()["token"]
    await client.post(
        f"/api/campaigns/invites/{invite_token}/accept",
        headers=pilot_headers,
    )

    char_resp = await client.post(
        "/api/characters",
        json={"callsign": "SPOT"},
        headers=pilot_headers,
    )
    character_id = char_resp.json()["id"]
    attach_resp = await client.post(
        f"/api/campaigns/{campaign_id}/characters",
        json={"character_id": character_id},
        headers=pilot_headers,
    )
    member_id = next(
        m["id"]
        for m in attach_resp.json()["members"]
        if m["user_id"] == "outcome_pilot"
    )
    await client.post(
        f"/api/campaigns/{campaign_id}/members/{member_id}/settings",
        json={"ready": True, "assigned_character_id": character_id},
        headers=pilot_headers,
    )
    await client.post(
        f"/api/campaigns/{campaign_id}/lobby",
        json={
            "mission_name": "Operation Outcome",
            "assigned_member_ids": [member_id],
            "min_pilot_count": 1,
            "preferred_pilot_count": 1,
        },
        headers=owner_headers,
    )
    launch_resp = await client.post(
        f"/api/campaigns/{campaign_id}/launch",
        json={},
        headers=owner_headers,
    )
    session_id = launch_resp.json()["data"]["sessions"][-1]["id"]

    outcome_resp = await client.post(
        f"/api/campaigns/{campaign_id}/sessions/{session_id}/outcome",
        json={
            "outcome": "success",
            "completion_score": 0.8,
            "debrief_notes": "Objective secured",
            "rewards": ["+1 reserve"],
        },
        headers=owner_headers,
    )
    assert outcome_resp.status_code == 200
    data = outcome_resp.json()
    session_record = data["data"]["sessions"][-1]
    assert session_record["mission_outcome"]["outcome"] == "success"
    assert data["data"]["mission_history"][-1]["rewards"] == ["+1 reserve"]
    assert data["data"]["lobby_state"]["status"] == "cooldown"
