/**
 * Title screen API hooks.
 */

import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import { characterKeys, useCharacters } from "./characters";

/**
 * Hook to check if there is at least one saved pilot (character).
 * Returns loading state and boolean result.
 */
export function useSavedPilot() {
  const { data, isLoading, error } = useCharacters();

  const hasSavedPilot = !isLoading && !error && (data?.items?.length ?? 0) > 0;

  return {
    hasSavedPilot,
    isLoading,
    error,
  };
}