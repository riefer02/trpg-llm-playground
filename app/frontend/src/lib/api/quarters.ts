/**
 * Quarters API hooks.
 * Provides hooks for accessing active character and quarters-related data.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import { characterKeys, useCharacters, useCharacter, type CharacterResponse } from "./characters";
import { getActiveCharacterId, clearActiveCharacterId } from "../save/saveSystem";

/**
 * Hook to get the active character.
 * Priority: active character ID from localStorage → first character in list.
 * Returns loading state, error, and character data.
 */
export function useActiveCharacter() {
  const activeCharacterId = getActiveCharacterId();
  const {
    data: activeCharData,
    isLoading: isLoadingActive,
    error: activeError,
  } = useCharacter(activeCharacterId || '');
  const {
    data: charactersData,
    isLoading: isLoadingChars,
    error: charsError,
  } = useCharacters();

  // Determine which data to use
  let character = null;
  let isLoading = false;
  let error = null;

  if (activeCharacterId) {
    if (activeCharData) {
      // Active character found
      character = activeCharData;
      isLoading = isLoadingActive;
      error = activeError;
    } else if (!isLoadingActive) {
      // Active character query finished but no data (not found or error)
      // Fall back to first character
      character = charactersData?.items?.[0] || null;
      isLoading = isLoadingChars;
      error = charsError;
      // Clear invalid active ID
      clearActiveCharacterId();
    } else {
      // Still loading active character
      isLoading = isLoadingActive;
      error = activeError;
    }
  } else {
    // No active ID, use first character
    character = charactersData?.items?.[0] || null;
    isLoading = isLoadingChars;
    error = charsError;
  }

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