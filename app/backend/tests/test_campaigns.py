"""Campaign API integration tests."""

import pytest
from httpx import AsyncClient


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
