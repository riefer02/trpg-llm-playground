/**
 * Pilot API hooks.
 * 
 * Usage:
 *   const { data: pilots } = usePilots()
 *   const { data: pilot } = usePilot('pilot_123')
 *   const createMutation = useCreatePilot()
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from './client'

// Response types (will be replaced by generated types)
export interface PilotResponse {
  id: string
  name: string
  data: Record<string, unknown>
  user_id: string
  campaign_id: string | null
}

export interface PilotListResponse {
  items: PilotResponse[]
  total: number
}

export interface PilotCreate {
  name: string
  callsign?: string
  data?: Record<string, unknown>
}

export interface PilotUpdate {
  name?: string
  callsign?: string
  data?: Record<string, unknown>
}

// Query keys for cache management
export const pilotKeys = {
  all: ['pilots'] as const,
  lists: () => [...pilotKeys.all, 'list'] as const,
  list: (filters: Record<string, string>) => [...pilotKeys.lists(), filters] as const,
  details: () => [...pilotKeys.all, 'detail'] as const,
  detail: (id: string) => [...pilotKeys.details(), id] as const,
}

/**
 * Hook for fetching all pilots.
 */
export function usePilots(campaignId?: string) {
  return useQuery({
    queryKey: pilotKeys.list({ campaign_id: campaignId || '' }),
    queryFn: () => {
      const params = campaignId ? `?campaign_id=${campaignId}` : ''
      return api.get<PilotListResponse>(`/pilots${params}`)
    },
  })
}

/**
 * Hook for fetching a single pilot.
 */
export function usePilot(id: string) {
  return useQuery({
    queryKey: pilotKeys.detail(id),
    queryFn: () => api.get<PilotResponse>(`/pilots/${id}`),
    enabled: !!id,
  })
}

/**
 * Hook for creating a new pilot.
 */
export function useCreatePilot() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: PilotCreate) => api.post<PilotResponse>('/pilots', data),
    onSuccess: (newPilot) => {
      // Invalidate list queries to refetch
      queryClient.invalidateQueries({ queryKey: pilotKeys.lists() })
      // Pre-populate the detail cache
      queryClient.setQueryData(pilotKeys.detail(newPilot.id), newPilot)
    },
  })
}

/**
 * Hook for updating a pilot.
 */
export function useUpdatePilot() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: PilotUpdate }) =>
      api.put<PilotResponse>(`/pilots/${id}`, data),
    onSuccess: (updatedPilot) => {
      // Update the detail cache
      queryClient.setQueryData(pilotKeys.detail(updatedPilot.id), updatedPilot)
      // Invalidate list to refetch
      queryClient.invalidateQueries({ queryKey: pilotKeys.lists() })
    },
  })
}

/**
 * Hook for deleting a pilot.
 */
export function useDeletePilot() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => api.delete(`/pilots/${id}`),
    onSuccess: (_, id) => {
      // Remove from detail cache
      queryClient.removeQueries({ queryKey: pilotKeys.detail(id) })
      // Invalidate list to refetch
      queryClient.invalidateQueries({ queryKey: pilotKeys.lists() })
    },
  })
}
