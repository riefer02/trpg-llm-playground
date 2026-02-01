/**
 * React hooks for managing save slots.
 */

import { useCallback, useState, useEffect } from 'react';
import type { CharacterResponse } from '../api/characters';
import type { MissionState } from './saveSystem';
import * as saveSystem from './saveSystem';

export type { MissionState };

// =============================================================================
// Hook: useSaveSlots
// =============================================================================

export function useSaveSlots() {
  const [slots, setSlots] = useState(() => saveSystem.getSaveSlots());
  const [isLoading, setIsLoading] = useState(false);

  const refreshSlots = useCallback(() => {
    setSlots(saveSystem.getSaveSlots());
  }, []);

  const saveToSlot = useCallback((
    slotIndex: number,
    character: CharacterResponse,
    name?: string
  ) => {
    setIsLoading(true);
    try {
      const savedSlot = saveSystem.saveToSlot(slotIndex, character, name);
      refreshSlots();
      return savedSlot;
    } finally {
      setIsLoading(false);
    }
  }, [refreshSlots]);

  const deleteSlot = useCallback((slotIndex: number) => {
    setIsLoading(true);
    try {
      saveSystem.deleteSaveSlot(slotIndex);
      refreshSlots();
    } finally {
      setIsLoading(false);
    }
  }, [refreshSlots]);

  const autoSave = useCallback((character: CharacterResponse) => {
    setIsLoading(true);
    try {
      const savedSlot = saveSystem.autoSave(character);
      refreshSlots();
      return savedSlot;
    } finally {
      setIsLoading(false);
    }
  }, [refreshSlots]);

  const autoSaveMissionLaunch = useCallback((
    character: CharacterResponse,
    missionState: MissionState
  ) => {
    setIsLoading(true);
    try {
      const savedSlot = saveSystem.autoSaveMissionLaunch(character, missionState);
      refreshSlots();
      return savedSlot;
    } finally {
      setIsLoading(false);
    }
  }, [refreshSlots]);

  const getMissionInProgress = useCallback(() => {
    return saveSystem.getMissionInProgress();
  }, []);

  const getCombatInProgress = useCallback(() => {
    return saveSystem.getCombatInProgress();
  }, []);

  const clearMissionState = useCallback((slotIndex: number) => {
    saveSystem.clearMissionState(slotIndex);
    refreshSlots();
  }, [refreshSlots]);

  const loadSlot = useCallback((slotIndex: number) => {
    setIsLoading(true);
    try {
      const slot = saveSystem.loadFromSave(slotIndex);
      // Note: active character ID is set by loadFromSave
      // We don't need to refresh slots as they haven't changed
      return slot;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadMostRecent = useCallback(() => {
    setIsLoading(true);
    try {
      const slot = saveSystem.loadMostRecentSave();
      return slot;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getMostRecent = useCallback(() => {
    return saveSystem.getMostRecentSave();
  }, []);

  const exportSlot = useCallback((slotIndex: number) => {
    saveSystem.exportSaveSlot(slotIndex);
  }, []);

  const importSlot = useCallback((jsonData: string) => {
    const slotIndex = saveSystem.importSaveSlot(jsonData);
    if (slotIndex >= 0) {
      refreshSlots();
    }
    return slotIndex;
  }, [refreshSlots]);

  const clearAll = useCallback(() => {
    setIsLoading(true);
    try {
      saveSystem.clearAllSaves();
      refreshSlots();
    } finally {
      setIsLoading(false);
    }
  }, [refreshSlots]);

  // Refresh slots on mount
  useEffect(() => {
    refreshSlots();
  }, [refreshSlots]);

  return {
    slots,
    isLoading,
    refreshSlots,
    saveToSlot,
    deleteSlot,
    autoSave,
    autoSaveMissionLaunch,
    loadSlot,
    loadMostRecent,
    getMostRecent,
    getMissionInProgress,
    getCombatInProgress,
    clearMissionState,
    exportSlot,
    importSlot,
    clearAll,
  };
}

// =============================================================================
// Hook: useAutoSave
// =============================================================================

export function useAutoSave() {
  const { autoSave, isLoading } = useSaveSlots();

  const triggerAutoSave = useCallback(async (character: CharacterResponse) => {
    if (!character) return null;
    return autoSave(character);
  }, [autoSave]);

  return {
    triggerAutoSave,
    isLoading,
  };
}

// =============================================================================
// Hook: useContinueSave
// =============================================================================

export function useContinueSave() {
  const { slots, getMostRecent, loadMostRecent, isLoading } = useSaveSlots();

  const getMostRecentSave = useCallback(() => {
    return getMostRecent();
  }, [getMostRecent]);

  const hasSavedGame = slots.length > 0;

  const loadMostRecentSave = useCallback(() => {
    return loadMostRecent();
  }, [loadMostRecent]);

  return {
    hasSavedGame,
    mostRecentSave: getMostRecentSave(),
    allSlots: slots,
    loadMostRecent: loadMostRecentSave,
    isLoading,
  };
}