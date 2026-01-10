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
  useCreatePilot,
  useUpdatePilot,
  useDeletePilot,
  pilotKeys,
} from './pilots'
export type {
  PilotResponse,
  PilotListResponse,
  PilotCreate,
  PilotUpdate,
} from './pilots'
