/**
 * Pilot API hooks using generated types from core models.
 *
 * NOTE: This is an internal/low-level API. Use /characters for user-facing features.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type {
  PilotTrigger,
  Talent,
  License,
  CoreBonus,
  Background,
  SkillSet,
} from "../types/lancer";

// =============================================================================
// Request Types (use generated primitives from core)
// =============================================================================

// Re-export generated types for backwards compatibility
export type SkillSetInput = Partial<SkillSet>;
export type TriggerInput = PilotTrigger;
export type TalentInput = Talent;
export type LicenseInput = License;
export type CoreBonusInput = CoreBonus;
export type BackgroundInput = Background;

export interface PilotCreateRequest {
  callsign: string;
  name?: string;
  level?: number;
  skills?: SkillSetInput;
  triggers?: PilotTrigger[];
  talents?: Talent[];
  licenses?: License[];
  core_bonuses?: CoreBonus[];
  background?: Background | null;
  notes?: string;
}

export interface PilotUpdateRequest {
  callsign?: string;
  name?: string;
  level?: number;
  skills?: SkillSetInput;
  triggers?: PilotTrigger[];
  talents?: Talent[];
  licenses?: License[];
  core_bonuses?: CoreBonus[];
  background?: Background | null;
  notes?: string;
}

// =============================================================================
// Response Types (backend returns hydrated Pilot with DB metadata)
// =============================================================================

export interface PilotResponse {
  // Database metadata
  id: string;
  user_id: string;
  campaign_id: string | null;
  created_at: string;
  updated_at: string;

  // Core pilot data (uses generated primitives)
  callsign: string;
  name: string;
  level: number;
  skills: SkillSet;
  triggers: PilotTrigger[];
  talents: Talent[];
  licenses: License[];
  core_bonuses: CoreBonus[];
  background: Background | null;
  notes: string;

  // Computed fields from core Pilot
  grit: number;
  hp: number;
  armor: number;
  evasion: number;
  e_defense: number;
  speed: number;
  save_target: number;
  attack_bonus: number;
}

export interface PilotListResponse {
  items: PilotResponse[];
  total: number;
}

export interface PilotValidationResponse {
  valid: boolean;
  issues: Array<{
    field: string;
    message: string;
    severity: string;
  }>;
}

// =============================================================================
// Query Keys
// =============================================================================

export const pilotKeys = {
  all: ["pilots"] as const,
  lists: () => [...pilotKeys.all, "list"] as const,
  list: (filters: Record<string, string>) =>
    [...pilotKeys.lists(), filters] as const,
  details: () => [...pilotKeys.all, "detail"] as const,
  detail: (id: string) => [...pilotKeys.details(), id] as const,
  validation: (id: string) => [...pilotKeys.detail(id), "validation"] as const,
};

// =============================================================================
// Query Hooks
// =============================================================================

/**
 * Hook for fetching all pilots.
 */
export function usePilots(campaignId?: string) {
  return useQuery({
    queryKey: pilotKeys.list({ campaign_id: campaignId || "" }),
    queryFn: () => {
      const params = campaignId ? `?campaign_id=${campaignId}` : "";
      return api.get<PilotListResponse>(`/pilots${params}`);
    },
  });
}

/**
 * Hook for fetching a single pilot.
 */
export function usePilot(id: string) {
  return useQuery({
    queryKey: pilotKeys.detail(id),
    queryFn: () => api.get<PilotResponse>(`/pilots/${id}`),
    enabled: !!id,
  });
}

/**
 * Hook for validating a pilot against progression rules.
 */
export function usePilotValidation(id: string) {
  return useQuery({
    queryKey: pilotKeys.validation(id),
    queryFn: () => api.get<PilotValidationResponse>(`/pilots/${id}/validate`),
    enabled: !!id,
  });
}

// =============================================================================
// Mutation Hooks
// =============================================================================

/**
 * Hook for creating a new pilot.
 */
export function useCreatePilot() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: PilotCreateRequest) =>
      api.post<PilotResponse>("/pilots", data),
    onSuccess: (newPilot) => {
      queryClient.invalidateQueries({ queryKey: pilotKeys.lists() });
      queryClient.setQueryData(pilotKeys.detail(newPilot.id), newPilot);
    },
  });
}

/**
 * Hook for updating a pilot.
 */
export function useUpdatePilot() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: PilotUpdateRequest }) =>
      api.put<PilotResponse>(`/pilots/${id}`, data),
    onSuccess: (updatedPilot) => {
      queryClient.setQueryData(pilotKeys.detail(updatedPilot.id), updatedPilot);
      queryClient.invalidateQueries({ queryKey: pilotKeys.lists() });
    },
  });
}

/**
 * Hook for deleting a pilot.
 */
export function useDeletePilot() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => api.delete(`/pilots/${id}`),
    onSuccess: (_, id) => {
      queryClient.removeQueries({ queryKey: pilotKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: pilotKeys.lists() });
    },
  });
}
