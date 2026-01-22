/**
 * Compendium API hooks for reference data.
 *
 * Read-only data for character creation and compendium views.
 * Uses generated types from core where possible.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import type {
  Background as CoreBackground,
  TriggerDefinition,
  TalentDefinition,
  MechFrameDefinition,
  MechWeaponDefinition,
  MechSystemDefinition,
  PilotGearItemDefinition,
} from "../types/lancer";

// =============================================================================
// Types - Simplified API response types (subset of core types)
// =============================================================================

// API returns simplified versions for the compendium endpoints
export interface Background {
  id: string;
  name: string;
  triggers: string[]; // Suggested trigger IDs
}

export interface Trigger {
  id: string;
  name: string;
}

export interface Talent {
  id: string;
  name: string;
  ranks: number;
}

export interface License {
  id: string;
  name: string;
  manufacturer: "GMS" | "IPS-N" | "SSC" | "HORUS" | "HA";
  frame_id: string;
}

interface ListResponse<T> {
  items: T[];
  total: number;
}

// Re-export core types for consumers who need full type info
export type { CoreBackground, TriggerDefinition, TalentDefinition };

// =============================================================================
// Query Keys
// =============================================================================

export const compendiumKeys = {
  all: ["compendium"] as const,
  backgrounds: () => [...compendiumKeys.all, "backgrounds"] as const,
  triggers: () => [...compendiumKeys.all, "triggers"] as const,
  talents: () => [...compendiumKeys.all, "talents"] as const,
  frames: () => [...compendiumKeys.all, "frames"] as const,
  weapons: () => [...compendiumKeys.all, "weapons"] as const,
  systems: () => [...compendiumKeys.all, "systems"] as const,
  pilotGear: () => [...compendiumKeys.all, "pilot-gear"] as const,
  licenses: () => [...compendiumKeys.all, "licenses"] as const,
};

// =============================================================================
// Hooks
// =============================================================================

export function useBackgrounds() {
  return useQuery({
    queryKey: compendiumKeys.backgrounds(),
    queryFn: () => api.get<ListResponse<Background>>("/compendium/backgrounds"),
    staleTime: Infinity, // Reference data doesn't change
    select: (data) => data.items,
  });
}

export function useTriggers() {
  return useQuery({
    queryKey: compendiumKeys.triggers(),
    queryFn: () => api.get<ListResponse<Trigger>>("/compendium/triggers"),
    staleTime: Infinity,
    select: (data) => data.items,
  });
}

export function useTalents() {
  return useQuery({
    queryKey: compendiumKeys.talents(),
    queryFn: () => api.get<ListResponse<Talent>>("/compendium/talents"),
    staleTime: Infinity,
    select: (data) => data.items,
  });
}

export function useFrames() {
  return useQuery({
    queryKey: compendiumKeys.frames(),
    queryFn: () =>
      api.get<ListResponse<MechFrameDefinition>>("/compendium/frames"),
    staleTime: Infinity,
    select: (data) => data.items,
  });
}

export function useWeapons() {
  return useQuery({
    queryKey: compendiumKeys.weapons(),
    queryFn: () =>
      api.get<ListResponse<MechWeaponDefinition>>("/compendium/weapons"),
    staleTime: Infinity,
    select: (data) => data.items,
  });
}

export function useSystems() {
  return useQuery({
    queryKey: compendiumKeys.systems(),
    queryFn: () =>
      api.get<ListResponse<MechSystemDefinition>>("/compendium/systems"),
    staleTime: Infinity,
    select: (data) => data.items,
  });
}

export function usePilotGear() {
  return useQuery({
    queryKey: compendiumKeys.pilotGear(),
    queryFn: () =>
      api.get<ListResponse<PilotGearItemDefinition>>("/compendium/pilot-gear"),
    staleTime: Infinity,
    select: (data) => data.items,
  });
}

export function useLicenses() {
  return useQuery({
    queryKey: compendiumKeys.licenses(),
    queryFn: () => api.get<ListResponse<License>>("/compendium/licenses"),
    staleTime: Infinity,
    select: (data) => data.items,
  });
}
