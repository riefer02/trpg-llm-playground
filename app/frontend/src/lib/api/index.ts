/**
 * API module exports.
 * 
 * Import from this file for cleaner imports:
 *   import { useHealth, useCharacters, api } from '@/lib/api'
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
  SkillSetInput,
  TriggerInput,
  TalentInput,
  LicenseInput,
  CoreBonusInput,
  BackgroundInput,
} from './pilots'
