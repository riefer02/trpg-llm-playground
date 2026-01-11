/**
 * Character API hooks using React Query.
 *
 * Characters are the unified abstraction combining Pilot + Mech(s).
 * This is the primary user-facing API for character management.
 *
 * Uses generated types from core for primitives (PilotTrigger, Talent, etc.)
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
  PilotLoadout,
  MechBuild,
} from "../types/lancer";

// =============================================================================
// Request Types (use generated primitives)
// =============================================================================

export interface CharacterCreateRequest {
  callsign: string;
  name?: string;
  use_ll0_defaults?: boolean;
  skills?: Partial<SkillSet>;
  triggers?: PilotTrigger[];
  talents?: Talent[];
  background?: Background;
  pilot_gear?: PilotLoadout;
  mech_name?: string;
  mech_frame_id?: string;
  level?: number;
  licenses?: License[];
  core_bonuses?: CoreBonus[];
  notes?: string;
}

export interface CharacterUpdateRequest {
  callsign?: string;
  name?: string;
  level?: number;
  skills?: Partial<SkillSet>;
  triggers?: PilotTrigger[];
  talents?: Talent[];
  licenses?: License[];
  core_bonuses?: CoreBonus[];
  background?: Background;
  notes?: string;
  active_mech_id?: string;
}

export interface PilotGearUpdateRequest {
  pilot_gear: PilotLoadout;
}

export interface MechBuildUpdateRequest {
  build: Omit<MechBuild, "frame_id">;
}

export interface MechAddRequest {
  name: string;
  frame_id?: string;
  build?: Record<string, unknown>;
}

// =============================================================================
// Response Types
// =============================================================================

export interface MechStats {
  hp: number;
  armor: number;
  evasion: number;
  e_defense: number;
  speed: number;
  sensor_range: number;
  tech_attack: number;
  heat_cap: number;
  repair_cap: number;
  system_points: number;
  save_target: number;
  size: string;
}

export interface MechConfig {
  id: string;
  name: string;
  frame_id: string;
  build: MechBuild;
}

export interface CharacterResponse {
  // Database metadata
  id: string;
  user_id: string;
  campaign_id: string | null;
  created_at: string;
  updated_at: string;

  // Pilot data (uses generated primitives)
  pilot_id: string;
  callsign: string;
  name: string;
  level: number;
  skills: SkillSet;
  triggers: PilotTrigger[];
  talents: Talent[];
  licenses: License[];
  core_bonuses: CoreBonus[];
  background: Background | null;
  pilot_gear: PilotLoadout | null;
  notes: string;

  // Pilot computed fields
  grit: number;
  pilot_hp: number;

  // Mech data
  mechs: MechConfig[];
  active_mech_id: string | null;

  // Computed fields
  active_mech_stats: MechStats | null;
  core_bonus_effects: Array<Record<string, unknown>>;
}

export interface CharacterListResponse {
  items: CharacterResponse[];
  total: number;
}

export interface ValidationIssue {
  code: string;
  message: string;
  severity: string;
}

export interface CharacterValidationResponse {
  valid: boolean;
  issues: ValidationIssue[];
}

// =============================================================================
// Query Keys
// =============================================================================

export const characterKeys = {
  all: ["characters"] as const,
  lists: () => [...characterKeys.all, "list"] as const,
  list: (filters: Record<string, string>) =>
    [...characterKeys.lists(), filters] as const,
  details: () => [...characterKeys.all, "detail"] as const,
  detail: (id: string) => [...characterKeys.details(), id] as const,
  validation: (id: string) => [...characterKeys.detail(id), "validation"] as const,
};

// =============================================================================
// Query Hooks
// =============================================================================

/**
 * Hook for fetching all characters.
 */
export function useCharacters(campaignId?: string) {
  return useQuery({
    queryKey: characterKeys.list({ campaign_id: campaignId || "" }),
    queryFn: () => {
      const params = campaignId ? `?campaign_id=${campaignId}` : "";
      return api.get<CharacterListResponse>(`/characters${params}`);
    },
  });
}

/**
 * Hook for fetching a single character.
 */
export function useCharacter(id: string) {
  return useQuery({
    queryKey: characterKeys.detail(id),
    queryFn: () => api.get<CharacterResponse>(`/characters/${id}`),
    enabled: !!id,
  });
}

/**
 * Hook for validating a character against game rules.
 */
export function useCharacterValidation(id: string) {
  return useQuery({
    queryKey: characterKeys.validation(id),
    queryFn: () =>
      api.get<CharacterValidationResponse>(`/characters/${id}/validate`),
    enabled: !!id,
  });
}

// =============================================================================
// Mutation Hooks
// =============================================================================

/**
 * Hook for creating a new character.
 *
 * For LL0 characters, only callsign is required - defaults are applied.
 */
export function useCreateCharacter() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CharacterCreateRequest) =>
      api.post<CharacterResponse>("/characters", data),
    onSuccess: (newCharacter) => {
      queryClient.invalidateQueries({ queryKey: characterKeys.lists() });
      queryClient.setQueryData(
        characterKeys.detail(newCharacter.id),
        newCharacter
      );
    },
  });
}

/**
 * Hook for updating a character.
 */
export function useUpdateCharacter() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: CharacterUpdateRequest }) =>
      api.put<CharacterResponse>(`/characters/${id}`, data),
    onSuccess: (updatedCharacter) => {
      queryClient.setQueryData(
        characterKeys.detail(updatedCharacter.id),
        updatedCharacter
      );
      queryClient.invalidateQueries({ queryKey: characterKeys.lists() });
    },
  });
}

/**
 * Hook for updating pilot gear loadout.
 */
export function useUpdatePilotGear() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      id,
      data,
    }: {
      id: string;
      data: PilotGearUpdateRequest;
    }) => api.put<CharacterResponse>(`/characters/${id}/pilot-gear`, data),
    onSuccess: (updatedCharacter) => {
      queryClient.setQueryData(
        characterKeys.detail(updatedCharacter.id),
        updatedCharacter
      );
      queryClient.invalidateQueries({ queryKey: characterKeys.lists() });
    },
  });
}

/**
 * Hook for updating a mech build.
 */
export function useUpdateMechBuild() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      characterId,
      mechId,
      data,
    }: {
      characterId: string;
      mechId: string;
      data: MechBuildUpdateRequest;
    }) =>
      api.put<CharacterResponse>(
        `/characters/${characterId}/mechs/${mechId}/build`,
        data
      ),
    onSuccess: (updatedCharacter) => {
      queryClient.setQueryData(
        characterKeys.detail(updatedCharacter.id),
        updatedCharacter
      );
      queryClient.invalidateQueries({ queryKey: characterKeys.lists() });
    },
  });
}

/**
 * Hook for deleting a character.
 */
export function useDeleteCharacter() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => api.delete(`/characters/${id}`),
    onSuccess: (_, id) => {
      queryClient.removeQueries({ queryKey: characterKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: characterKeys.lists() });
    },
  });
}

// =============================================================================
// Mech Management Hooks
// =============================================================================

/**
 * Hook for adding a mech to a character.
 */
export function useAddMech() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      characterId,
      data,
    }: {
      characterId: string;
      data: MechAddRequest;
    }) => api.post<CharacterResponse>(`/characters/${characterId}/mechs`, data),
    onSuccess: (updatedCharacter) => {
      queryClient.setQueryData(
        characterKeys.detail(updatedCharacter.id),
        updatedCharacter
      );
    },
  });
}

/**
 * Hook for removing a mech from a character.
 */
export function useRemoveMech() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      characterId,
      mechId,
    }: {
      characterId: string;
      mechId: string;
    }) =>
      api.delete<CharacterResponse>(
        `/characters/${characterId}/mechs/${mechId}`
      ),
    onSuccess: (updatedCharacter) => {
      queryClient.setQueryData(
        characterKeys.detail(updatedCharacter.id),
        updatedCharacter
      );
    },
  });
}

/**
 * Hook for setting the active mech.
 */
export function useSetActiveMech() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      characterId,
      mechId,
    }: {
      characterId: string;
      mechId: string;
    }) =>
      api.put<CharacterResponse>(
        `/characters/${characterId}/mechs/${mechId}/activate`
      ),
    onSuccess: (updatedCharacter) => {
      queryClient.setQueryData(
        characterKeys.detail(updatedCharacter.id),
        updatedCharacter
      );
    },
  });
}
