/**
 * Health check API hooks.
 * 
 * Usage:
 *   const { data, isLoading, error } = useHealth()
 */

import { useQuery } from '@tanstack/react-query'
import { api } from './client'

// Response types
export interface HealthResponse {
  status: string
  version: string
}

export interface DatabaseHealthResponse {
  status: string
  database: string
}

// Query keys for cache management
export const healthKeys = {
  all: ['health'] as const,
  basic: () => [...healthKeys.all, 'basic'] as const,
  database: () => [...healthKeys.all, 'database'] as const,
}

/**
 * Hook for basic health check.
 */
export function useHealth() {
  return useQuery({
    queryKey: healthKeys.basic(),
    queryFn: () => api.get<HealthResponse>('/health'),
    staleTime: 30_000, // 30 seconds
    refetchInterval: 60_000, // Refetch every minute
  })
}

/**
 * Hook for database health check.
 */
export function useDatabaseHealth() {
  return useQuery({
    queryKey: healthKeys.database(),
    queryFn: () => api.get<DatabaseHealthResponse>('/health/db'),
    staleTime: 30_000,
    refetchInterval: 60_000,
  })
}
