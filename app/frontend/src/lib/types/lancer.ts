/**
 * Placeholder for auto-generated Lancer types.
 * 
 * Run `npm run generate:types` to generate types from Python models.
 * 
 * This file will be overwritten by the type generation script.
 */

// Placeholder types until generation is run
export interface Pilot {
  id: string
  name: string
  callsign: string
  // Add more fields as needed
}

export interface Mech {
  id: string
  name: string
  frame_id: string
  // Add more fields as needed
}

// Re-export API types for convenience
export type {
  PilotResponse,
  PilotListResponse,
  PilotCreate,
  PilotUpdate,
} from '../api/pilots'
