/**
 * API module exports.
 *
 * Import from this file for cleaner imports:
 *   import { useHealth, useCharacters, api } from '@/lib/api'
 *
 * For core domain types, import from '@/lib/types/lancer'
 */

// Core client
export { apiClient, api, APIError } from './client'
export type { RequestOptions } from './client'

// Health hooks
export { useHealth, useDatabaseHealth, healthKeys } from './health'
export type { HealthResponse, DatabaseHealthResponse } from './health'

// Character hooks (primary user-facing API)
export {
  useCharacters,
  useCharacter,
  useCharacterValidation,
  useCreateCharacter,
  useUpdateCharacter,
  useDeleteCharacter,
  useAddMech,
  useRemoveMech,
  useSetActiveMech,
  characterKeys,
} from './characters'
export type {
  CharacterResponse,
  CharacterListResponse,
  CharacterCreateRequest,
  CharacterUpdateRequest,
  CharacterValidationResponse,
  MechAddRequest,
  MechStats,
  MechConfig,
  ValidationIssue,
} from './characters'

// Compendium hooks (reference data for character creation)
export {
  useBackgrounds,
  useTriggers,
  useTalents,
  compendiumKeys,
} from './compendium'
// Note: Compendium types are simplified API responses, not full core types
export type {
  Background as CompendiumBackground,
  Trigger as CompendiumTrigger,
  Talent as CompendiumTalent,
} from './compendium'

// Pilot hooks (internal/low-level primitive)
export {
  usePilots,
  usePilot,
  usePilotValidation,
  useCreatePilot,
  useUpdatePilot,
  useDeletePilot,
  pilotKeys,
} from './pilots'
export type {
  PilotResponse,
  PilotListResponse,
  PilotCreateRequest,
  PilotUpdateRequest,
  PilotValidationResponse,
} from './pilots'

// Re-export generated core types for convenience
// These are the source of truth - use these for domain data
export type {
  PilotTrigger,
  Talent,
  License,
  CoreBonus,
  Background,
  SkillSet,
  Character,
  MechConfiguration,
  Pilot,
} from '../types/lancer'
