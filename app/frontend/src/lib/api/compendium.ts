/**
 * Compendium API hooks for reference data.
 *
 * Read-only data for character creation forms (backgrounds, triggers, talents).
 * Uses generated types from core where possible.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import type {
  Background as CoreBackground,
  TriggerDefinition,
  TalentDefinition,
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
