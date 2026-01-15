import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./client";

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
}

export interface CampaignDetail extends CampaignSummary {
  data: Record<string, unknown>;
  members: CampaignMember[];
  invites: CampaignInvite[];
  characters: CampaignCharacter[];
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
