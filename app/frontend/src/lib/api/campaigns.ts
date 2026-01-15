import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./client";
import type { Campaign as LancerCampaign } from "../types/lancer";

export interface CampaignMember {
  id: string;
  user_id: string;
  role: string;
  status: string;
  ready_state: string;
  assigned_character_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface CampaignInvite {
  id: string;
  token: string;
  role: string;
  status: string;
  invited_email: string | null;
  expires_at: string | null;
  invited_by_user_id: string;
  redeemed_by_user_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface CampaignCharacter {
    id: string;
    campaign_id: string;
    character_id: string;
    callsign: string;
    user_id: string;
    role: string;
    notes: string;
    created_at: string;
    updated_at: string;
}

export interface CampaignReadinessSummary {
    ready_members: number;
    ready_players: number;
    assigned_ready_players: number;
    total_members: number;
    min_pilots: number;
    preferred_pilots: number;
    can_launch: boolean;
    lobby_status: string | null;
    issues: string[];
}

export interface CampaignInvitePreviewResponse {
    campaign_id: string;
    campaign_name: string;
    squad_name: string | null;
    patron: string | null;
    role: string;
    status: string;
    expires_at: string | null;
    seat_warning: string | null;
    ready_players: number;
    preferred_pilots: number;
    can_join: boolean;
}

export interface CampaignInviteResendRequest {
    expires_in_hours?: number;
}

export interface CampaignSessionOutcomeRequest {
    outcome: "success" | "partial" | "failure" | "catastrophic";
    completion_score?: number;
    debrief_notes?: string | null;
    reserves_spent?: Array<Record<string, unknown>>;
    reserves_earned?: Array<Record<string, unknown>>;
    rewards?: string[];
}

export interface CampaignSummary {
    id: string;
    user_id: string;
    campaign_id: string | null;
    created_at: string;
    updated_at: string;
    name: string;
    description: string;
    status: string;
    visibility: string;
    membership_role: string;
    membership_status: string;
    ready_state: string | null;
    member_count: number;
    character_count: number;
    lobby_status: string | null;
}

export interface CampaignDetail extends CampaignSummary {
    data: LancerCampaign;
    members: CampaignMember[];
    invites: CampaignInvite[];
    characters: CampaignCharacter[];
    readiness_summary: CampaignReadinessSummary;
    seat_warning: string | null;
}


export interface CampaignListResponse {
  items: CampaignSummary[];
  total: number;
  limit?: number | null;
  offset?: number | null;
}

export interface CampaignCreateRequest {
  name: string;
  description?: string;
  notes?: string;
}

export interface CampaignInviteCreateRequest {
  role?: "player" | "co_gm";
  invited_email?: string;
  expires_in_hours?: number;
}

export interface CampaignCharacterAttachRequest {
  character_id: string;
  role?: "player" | "npc";
  notes?: string;
}

export interface CampaignMemberSettingsRequest {
    ready?: boolean;
    assigned_character_id?: string | null;
}

export interface CampaignIdentityUpdateRequest {
    squad_name?: string;
    patron?: string;
    who_we_are?: string;
    relationships?: string[];
    themes?: string[];
    gm_prompts?: string[];
}

export interface CampaignLobbyObjectiveInput {
    id: string;
    title: string;
    success_condition: string;
    priority?: "primary" | "secondary" | "optional";
    related_objective_id?: string | null;
}

export interface CampaignLobbyStakesInput {
    stakes_type: "personal" | "faction" | "immediate" | "gradual" | "custom";
    summary: string;
    consequences_success?: string;
    consequences_failure?: string;
    consequences_partial?: string;
}

export interface CampaignLobbyReserveInput {
    reserve_id: string;
    assigned_pilot_id?: string | null;
    usage_notes?: string | null;
    status?: "planned" | "spent" | "earned";
}

export interface CampaignLobbyUpdateRequest {
    mission_name: string;
    operation_code?: string | null;
    theater?: string | null;
    objectives?: CampaignLobbyObjectiveInput[];
    stakes?: CampaignLobbyStakesInput | null;
    reserves?: CampaignLobbyReserveInput[];
    briefing_notes?: string | null;
    support_assets?: string[];
    threats?: string[];
    assigned_member_ids?: string[];
    preferred_pilot_count?: number;
    min_pilot_count?: number;
    gm_notes?: string | null;
    status?: "draft" | "ready" | "launched" | "cooldown" | null;
}

export interface CampaignMissionLaunchRequest {
    environment?: "standard" | "zero_g" | "underwater";
    notes?: string | null;
}

export interface SessionLifecycleUpdateRequest {
    phase: "downtime" | "brief" | "prep" | "mission" | "debrief";
    status: "pending" | "in_progress" | "complete";
    summary?: string | null;
    gm_notes?: string | null;
}

export const campaignKeys = {

  all: ["campaigns"] as const,
  lists: () => [...campaignKeys.all, "list"] as const,
  detail: (id: string) => [...campaignKeys.all, "detail", id] as const,
};

export function useCampaigns() {
  return useQuery({
    queryKey: campaignKeys.lists(),
    queryFn: () => api.get<CampaignListResponse>("/campaigns"),
  });
}

export function useCampaign(id: string) {
  return useQuery({
    queryKey: campaignKeys.detail(id),
    queryFn: () => api.get<CampaignDetail>(`/campaigns/${id}`),
    enabled: !!id,
  });
}

export function useCreateCampaign() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: CampaignCreateRequest) =>
      api.post<CampaignDetail>("/campaigns", data),
    onSuccess: (campaign) => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.lists() });
      queryClient.setQueryData(campaignKeys.detail(campaign.id), campaign);
    },
  });
}

export function useCreateCampaignInvite() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      campaignId,
      data,
    }: {
      campaignId: string;
      data: CampaignInviteCreateRequest;
    }) => api.post<CampaignInvite>(`/campaigns/${campaignId}/invites`, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: campaignKeys.detail(variables.campaignId),
      });
    },
  });
}

export function useAcceptCampaignInvite() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (token: string) =>
      api.post<CampaignDetail>(`/campaigns/invites/${token}/accept`),
    onSuccess: (campaign) => {
      queryClient.invalidateQueries({ queryKey: campaignKeys.lists() });
      queryClient.setQueryData(campaignKeys.detail(campaign.id), campaign);
    },
  });
}

export function useAttachCampaignCharacter() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      campaignId,
      data,
    }: {
      campaignId: string;
      data: CampaignCharacterAttachRequest;
    }) => api.post<CampaignDetail>(`/campaigns/${campaignId}/characters`, data),
    onSuccess: (campaign, variables) => {
      queryClient.setQueryData(campaignKeys.detail(variables.campaignId), campaign);
    },
  });
}

export function useUpdateCampaignMemberSettings() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            campaignId,
            memberId,
            data,
        }: {
            campaignId: string;
            memberId: string;
            data: CampaignMemberSettingsRequest;
        }) =>
            api.post<CampaignMember>(
                `/campaigns/${campaignId}/members/${memberId}/settings`,
                data,
            ),
        onSuccess: (member, variables) => {
            queryClient.setQueryData<CampaignDetail>(
                campaignKeys.detail(variables.campaignId),
                (prev) => {
                    if (!prev) return prev;
                    return {
                        ...prev,
                        members: prev.members.map((m) =>
                            m.id === member.id ? { ...m, ...member } : m,
                        ),
                    };
                },
            );
        },
    });
}

export function useUpdateCampaignIdentity() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            campaignId,
            data,
        }: {
            campaignId: string;
            data: CampaignIdentityUpdateRequest;
        }) => api.post<CampaignDetail>(`/campaigns/${campaignId}/identity`, data),
        onSuccess: (campaign, variables) => {
            queryClient.setQueryData(
                campaignKeys.detail(variables.campaignId),
                campaign,
            );
        },
    });
}

export function useUpdateCampaignLobby() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            campaignId,
            data,
        }: {
            campaignId: string;
            data: CampaignLobbyUpdateRequest;
        }) => api.post<CampaignDetail>(`/campaigns/${campaignId}/lobby`, data),
        onSuccess: (campaign, variables) => {
            queryClient.setQueryData(
                campaignKeys.detail(variables.campaignId),
                campaign,
            );
        },
    });
}

export function useLaunchCampaignMission() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            campaignId,
            data,
        }: {
            campaignId: string;
            data: CampaignMissionLaunchRequest;
        }) => api.post<CampaignDetail>(`/campaigns/${campaignId}/launch`, data),
        onSuccess: (campaign, variables) => {
            queryClient.setQueryData(
                campaignKeys.detail(variables.campaignId),
                campaign,
            );
        },
    });
}

export function useUpdateSessionLifecycle() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            campaignId,
            sessionId,
            data,
        }: {
            campaignId: string;
            sessionId: string;
            data: SessionLifecycleUpdateRequest;
        }) =>
            api.post<CampaignDetail>(
                `/campaigns/${campaignId}/sessions/${sessionId}/lifecycle`,
                data,
            ),
        onSuccess: (campaign, variables) => {
            queryClient.setQueryData(
                campaignKeys.detail(variables.campaignId),
                campaign,
            );
        },
    });
}

export function useRecordCampaignSessionOutcome() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            campaignId,
            sessionId,
            data,
        }: {
            campaignId: string;
            sessionId: string;
            data: CampaignSessionOutcomeRequest;
        }) =>
            api.post<CampaignDetail>(
                `/campaigns/${campaignId}/sessions/${sessionId}/outcome`,
                data,
            ),
        onSuccess: (campaign, variables) => {
            queryClient.setQueryData(
                campaignKeys.detail(variables.campaignId),
                campaign,
            );
        },
    });
}

export function usePreviewCampaignInvite(token: string | null | undefined) {
    return useQuery({
        queryKey: ["campaigns", "invite-preview", token],
        queryFn: () =>
            api.get<CampaignInvitePreviewResponse>(
                `/campaigns/invites/${token}/preview`,
            ),
        enabled: Boolean(token),
    });
}

export function useRevokeCampaignInvite() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            campaignId,
            inviteId,
        }: {
            campaignId: string;
            inviteId: string;
        }) =>
            api.post<CampaignInvite>(
                `/campaigns/${campaignId}/invites/${inviteId}/revoke`,
            ),
        onSuccess: (_, variables) => {
            queryClient.invalidateQueries({
                queryKey: campaignKeys.detail(variables.campaignId),
            });
        },
    });
}

export function useResendCampaignInvite() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({
            campaignId,
            inviteId,
            data,
        }: {
            campaignId: string;
            inviteId: string;
            data?: CampaignInviteResendRequest;
        }) =>
            api.post<CampaignInvite>(
                `/campaigns/${campaignId}/invites/${inviteId}/resend`,
                data ?? {},
            ),
        onSuccess: (_, variables) => {
            queryClient.invalidateQueries({
                queryKey: campaignKeys.detail(variables.campaignId),
            });
        },
    });
}


