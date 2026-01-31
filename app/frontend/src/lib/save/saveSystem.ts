/**
 * Save/Load System for Lancer Tactics AI.
 * 
 * Implements localStorage-based save slots with auto-save functionality.
 * Follows core-first principle - uses CharacterResponse types directly.
 */

import type { CharacterResponse } from '../api/characters';

// =============================================================================
// Save Slot Schema
// =============================================================================

export interface SaveSlot {
  /** Slot index (0-2) */
  slot: number;
  /** Character data at time of save */
  character: CharacterResponse;
  /** Timestamp of save (ISO string) */
  timestamp: string;
  /** Optional name for the save (e.g., "Mission 3 Complete") */
  name?: string;
  /** Game version for compatibility checking */
  version: string;
}

export interface SaveSlotsData {
  /** Array of save slots (max 3) */
  slots: SaveSlot[];
  /** ID of the most recently used slot (for auto-save) */
  lastUsedSlot: number | null;
}

// =============================================================================
// Constants
// =============================================================================

const STORAGE_KEY = 'lancer_tactics_saves';
const MAX_SLOTS = 3;
const CURRENT_VERSION = '1.0.0';

// =============================================================================
// LocalStorage Utilities
// =============================================================================

function loadSaveSlots(): SaveSlotsData {
  if (typeof window === 'undefined') {
    return { slots: [], lastUsedSlot: null };
  }
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      // Validate basic structure
      if (parsed.slots && Array.isArray(parsed.slots)) {
        // Ensure slots have required fields
        const validSlots = parsed.slots
          .filter((slot: any) => 
            slot && 
            typeof slot.slot === 'number' && 
            slot.slot >= 0 && 
            slot.slot < MAX_SLOTS &&
            slot.character &&
            slot.timestamp
          )
          .slice(0, MAX_SLOTS);
        return {
          slots: validSlots,
          lastUsedSlot: typeof parsed.lastUsedSlot === 'number' ? parsed.lastUsedSlot : null,
        };
      }
    }
  } catch {
    // Ignore parse errors
  }
  return { slots: [], lastUsedSlot: null };
}

function saveSaveSlots(data: SaveSlotsData): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    // Ignore storage errors (quota exceeded, etc.)
  }
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Get all save slots.
 */
export function getSaveSlots(): SaveSlot[] {
  return loadSaveSlots().slots;
}

/**
 * Get a specific save slot by index (0-2).
 * Returns undefined if slot empty.
 */
export function getSaveSlot(slotIndex: number): SaveSlot | undefined {
  if (slotIndex < 0 || slotIndex >= MAX_SLOTS) {
    throw new Error(`Invalid slot index: ${slotIndex}`);
  }
  const slots = loadSaveSlots().slots;
  return slots.find(s => s.slot === slotIndex);
}

/**
 * Save a character to a specific slot.
 * Overwrites any existing save in that slot.
 */
export function saveToSlot(
  slotIndex: number,
  character: CharacterResponse,
  name?: string
): SaveSlot {
  if (slotIndex < 0 || slotIndex >= MAX_SLOTS) {
    throw new Error(`Invalid slot index: ${slotIndex}`);
  }

  const data = loadSaveSlots();
  const existingIndex = data.slots.findIndex(s => s.slot === slotIndex);
  const newSlot: SaveSlot = {
    slot: slotIndex,
    character,
    timestamp: new Date().toISOString(),
    name,
    version: CURRENT_VERSION,
  };

  if (existingIndex >= 0) {
    data.slots[existingIndex] = newSlot;
  } else {
    data.slots.push(newSlot);
  }

  // Keep slots sorted by slot index for consistency
  data.slots.sort((a, b) => a.slot - b.slot);
  data.lastUsedSlot = slotIndex;

  saveSaveSlots(data);
  return newSlot;
}

/**
 * Auto-save character to the last used slot, or slot 0 if none.
 */
export function autoSave(character: CharacterResponse): SaveSlot {
  const data = loadSaveSlots();
  const slotIndex = data.lastUsedSlot !== null ? data.lastUsedSlot : 0;
  return saveToSlot(slotIndex, character, `Auto-save ${new Date().toLocaleDateString()}`);
}

/**
 * Delete a save slot.
 */
export function deleteSaveSlot(slotIndex: number): void {
  if (slotIndex < 0 || slotIndex >= MAX_SLOTS) {
    throw new Error(`Invalid slot index: ${slotIndex}`);
  }

  const data = loadSaveSlots();
  data.slots = data.slots.filter(s => s.slot !== slotIndex);
  if (data.lastUsedSlot === slotIndex) {
    data.lastUsedSlot = data.slots.length > 0 ? data.slots[0].slot : null;
  }
  saveSaveSlots(data);
}

/**
 * Get the most recent save (by timestamp).
 */
export function getMostRecentSave(): SaveSlot | undefined {
  const slots = loadSaveSlots().slots;
  if (slots.length === 0) return undefined;
  
  return slots.reduce((latest, current) => 
    new Date(current.timestamp) > new Date(latest.timestamp) ? current : latest
  );
}

/**
 * Export a save slot as a JSON file for download.
 */
export function exportSaveSlot(slotIndex: number): void {
  const slot = getSaveSlot(slotIndex);
  if (!slot) {
    throw new Error(`No save data in slot ${slotIndex}`);
  }

  const dataStr = JSON.stringify(slot, null, 2);
  const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
  
  const exportFileDefaultName = `lancer_save_slot_${slotIndex}_${new Date().toISOString().slice(0, 10)}.json`;
  
  const linkElement = document.createElement('a');
  linkElement.setAttribute('href', dataUri);
  linkElement.setAttribute('download', exportFileDefaultName);
  linkElement.click();
}

/**
 * Import a save slot from a JSON file.
 * Returns the imported slot index or -1 on error.
 */
export function importSaveSlot(jsonData: string): number {
  try {
    const parsed = JSON.parse(jsonData);
    // Basic validation
    if (!parsed.slot || typeof parsed.slot !== 'number' || parsed.slot < 0 || parsed.slot >= MAX_SLOTS) {
      throw new Error('Invalid slot index in imported data');
    }
    if (!parsed.character || !parsed.timestamp) {
      throw new Error('Missing required fields');
    }

    const data = loadSaveSlots();
    const existingIndex = data.slots.findIndex(s => s.slot === parsed.slot);
    const importedSlot: SaveSlot = {
      slot: parsed.slot,
      character: parsed.character,
      timestamp: parsed.timestamp,
      name: parsed.name,
      version: parsed.version || CURRENT_VERSION,
    };

    if (existingIndex >= 0) {
      data.slots[existingIndex] = importedSlot;
    } else {
      data.slots.push(importedSlot);
    }

    data.slots.sort((a, b) => a.slot - b.slot);
    data.lastUsedSlot = parsed.slot;

    saveSaveSlots(data);
    return parsed.slot;
  } catch (error) {
    console.error('Failed to import save slot:', error);
    return -1;
  }
}

/**
 * Clear all save slots (debug/reset).
 */
export function clearAllSaves(): void {
  saveSaveSlots({ slots: [], lastUsedSlot: null });
}

// =============================================================================
// Active Character Management
// =============================================================================

const ACTIVE_CHARACTER_KEY = 'lancer_tactics_active_character';

/**
 * Get the active character ID from localStorage.
 */
export function getActiveCharacterId(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return localStorage.getItem(ACTIVE_CHARACTER_KEY);
  } catch {
    return null;
  }
}

/**
 * Set the active character ID in localStorage.
 */
export function setActiveCharacterId(characterId: string): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(ACTIVE_CHARACTER_KEY, characterId);
  } catch {
    // Ignore storage errors
  }
}

/**
 * Clear the active character ID.
 */
export function clearActiveCharacterId(): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.removeItem(ACTIVE_CHARACTER_KEY);
  } catch {
    // Ignore errors
  }
}

/**
 * Load a character from a save slot and set it as active.
 * Returns the loaded save slot, or undefined if slot is empty.
 */
export function loadFromSave(slotIndex: number): SaveSlot | undefined {
  const slot = getSaveSlot(slotIndex);
  if (!slot) {
    return undefined;
  }
  setActiveCharacterId(slot.character.id);
  return slot;
}

/**
 * Load the most recent save and set it as active.
 * Returns the loaded save slot, or undefined if no saves exist.
 */
export function loadMostRecentSave(): SaveSlot | undefined {
  const slot = getMostRecentSave();
  if (!slot) {
    return undefined;
  }
  setActiveCharacterId(slot.character.id);
  return slot;
}