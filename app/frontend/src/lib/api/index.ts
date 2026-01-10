/**
 * API module exports.
 * 
 * Import from this file for cleaner imports:
 *   import { useHealth, usePilots, api } from '@/lib/api'
 */

// Core client
export { apiClient, api, APIError } from './client'
export type { RequestOptions } from './client'

// Health hooks
export { useHealth, useDatabaseHealth, healthKeys } from './health'
export type { HealthResponse, DatabaseHealthResponse } from './health'

// Pilot hooks
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
