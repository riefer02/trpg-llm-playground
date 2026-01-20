/**
 * Hook for real-time loadout validation during editing.
 */

import { useMemo } from "react";
import {
  useWeapons,
  useSystems,
  type CharacterResponse,
} from "../api";
import type { MechFrameDefinition } from "../types/lancer";
import {
  validateLoadout,
  type BuildDraft,
  type LoadoutValidationResult,
} from "../validation/loadout";

/**
 * Hook that provides real-time validation of a mech loadout draft.
 *
 * @param draft - Current draft state of the mech build
 * @param frame - The frame definition for the active mech
 * @param character - The character response containing licenses and level
 * @returns LoadoutValidationResult with valid flag and issues array
 */
export function useLoadoutValidation(
  draft: BuildDraft,
  frame: MechFrameDefinition | null,
  character: CharacterResponse
): LoadoutValidationResult {
  const { data: weapons } = useWeapons();
  const { data: systems } = useSystems();

  return useMemo(() => {
    const weaponMap = new Map(weapons?.map((w) => [w.id, w]) ?? []);
    const systemMap = new Map(systems?.map((s) => [s.id, s]) ?? []);
    const licenseMap = new Map(
      character.licenses.map((l) => [l.license_id, l.rank])
    );

    return validateLoadout({
      draft,
      frame,
      weapons: weaponMap,
      systems: systemMap,
      licenses: licenseMap,
      spLimit: character.active_mech_stats?.system_points ?? 0,
      pilotLevel: character.level,
    });
  }, [draft, frame, weapons, systems, character]);
}
