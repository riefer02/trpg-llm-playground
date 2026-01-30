/**
 * Quarters API hooks.
 * Provides hooks for accessing active character and quarters-related data.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import { characterKeys, useCharacters, type CharacterResponse } from "./characters";

/**
 * Hook to get the active character (first character in list for MVP).
 * Returns loading state, error, and character data.
 */
export function useActiveCharacter() {
  const { data, isLoading, error } = useCharacters();

  // For MVP: first character is considered active
  const character = data?.items?.[0] || null;

  return {
    character,
    isLoading,
    error,
  };
}

/**
 * Hook to check if there's at least one mission available (placeholder).
 * Returns count and loading state.
 */
export function useMissionCount() {
  // TODO: Replace with actual mission count API when missions are implemented
  return {
    count: 3, // Placeholder: 3 available missions
    isLoading: false,
    error: null,
  };
}