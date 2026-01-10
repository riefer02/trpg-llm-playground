/**
 * API client for backend communication.
 * 
 * Provides a type-safe fetch wrapper with consistent error handling.
 * 
 * Usage:
 *   const data = await apiClient<Pilot[]>('/pilots')
 *   const pilot = await apiClient<Pilot>('/pilots/123')
 */

// API base URL - use relative path in development (Vite proxy handles it)
// In production, set VITE_API_URL to the actual backend URL
const API_BASE = import.meta.env.VITE_API_URL || '/api'

/**
 * Custom error class for API errors.
 * Contains status code and parsed error details from the backend.
 */
export class APIError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public code?: string,
    public errors?: Array<{ field: string; message: string }>,
  ) {
    super(detail)
    this.name = 'APIError'
  }
}

/**
 * Options for API requests.
 */
export interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: unknown
}

/**
 * Type-safe API client.
 * 
 * @param endpoint - API endpoint (e.g., '/pilots')
 * @param options - Fetch options with typed body
 * @returns Parsed JSON response
 * @throws APIError on non-2xx responses
 * 
 * @example
 * // GET request
 * const pilots = await apiClient<PilotListResponse>('/pilots')
 * 
 * @example
 * // POST request
 * const newPilot = await apiClient<PilotResponse>('/pilots', {
 *   method: 'POST',
 *   body: { name: 'Ace', callsign: 'ACE' }
 * })
 */
export async function apiClient<T>(
  endpoint: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, headers: customHeaders, ...restOptions } = options

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...customHeaders,
  }

  const config: RequestInit = {
    ...restOptions,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  }

  const response = await fetch(`${API_BASE}${endpoint}`, config)

  // Handle non-2xx responses
  if (!response.ok) {
    let errorData: { detail?: string; code?: string; errors?: Array<{ field: string; message: string }> } = {}
    
    try {
      errorData = await response.json()
    } catch {
      // Response wasn't JSON
    }

    throw new APIError(
      response.status,
      errorData.detail || `Request failed with status ${response.status}`,
      errorData.code,
      errorData.errors,
    )
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T
  }

  return response.json()
}

/**
 * Convenience methods for common HTTP verbs.
 */
export const api = {
  get: <T>(endpoint: string, options?: RequestOptions) =>
    apiClient<T>(endpoint, { ...options, method: 'GET' }),

  post: <T>(endpoint: string, body?: unknown, options?: RequestOptions) =>
    apiClient<T>(endpoint, { ...options, method: 'POST', body }),

  put: <T>(endpoint: string, body?: unknown, options?: RequestOptions) =>
    apiClient<T>(endpoint, { ...options, method: 'PUT', body }),

  patch: <T>(endpoint: string, body?: unknown, options?: RequestOptions) =>
    apiClient<T>(endpoint, { ...options, method: 'PATCH', body }),

  delete: <T>(endpoint: string, options?: RequestOptions) =>
    apiClient<T>(endpoint, { ...options, method: 'DELETE' }),
}
