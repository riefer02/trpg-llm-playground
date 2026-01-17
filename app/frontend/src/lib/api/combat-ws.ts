/**
 * WebSocket hook for real-time combat state synchronization.
 *
 * This module provides a React hook that maintains a WebSocket connection
 * to the combat session endpoint and updates the React Query cache when
 * state changes are received from the server.
 */

import { useEffect, useRef, useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { combatKeys, type CombatSessionResponse } from "./combat";

/** WebSocket connection state */
export interface CombatWebSocketState {
  /** Whether the WebSocket is currently connected */
  isConnected: boolean;
  /** Number of reconnection attempts since last successful connection */
  reconnectAttempts: number;
}

/** Options for the WebSocket hook */
export interface UseCombatWebSocketOptions {
  /** Whether to enable the WebSocket connection (default: true) */
  enabled?: boolean;
  /** Reconnection delay in milliseconds (default: 2000) */
  reconnectDelay?: number;
  /** Maximum reconnection attempts before giving up (default: 10) */
  maxReconnectAttempts?: number;
}

/**
 * Hook that maintains a WebSocket connection for real-time combat updates.
 *
 * When connected, the hook will receive state updates from the server and
 * automatically update the React Query cache. This allows all components
 * using useCombatSession to receive updates without polling.
 *
 * @param sessionId - The combat session ID to connect to
 * @param options - Configuration options
 * @returns Connection state information
 *
 * @example
 * ```tsx
 * function CombatPage() {
 *   const { combatId } = useParams();
 *   const { isConnected } = useCombatWebSocket(combatId);
 *
 *   // Fallback to polling if WS disconnected
 *   const { data } = useCombatSession(combatId, {
 *     pollingInterval: isConnected ? undefined : 5000,
 *   });
 *
 *   return (
 *     <div>
 *       <ConnectionIndicator connected={isConnected} />
 *       <CombatCanvas data={data} />
 *     </div>
 *   );
 * }
 * ```
 */
export function useCombatWebSocket(
  sessionId: string | null,
  options: UseCombatWebSocketOptions = {}
): CombatWebSocketState {
  const {
    enabled = true,
    reconnectDelay = 2000,
    maxReconnectAttempts = 10,
  } = options;

  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const connect = useCallback(() => {
    if (!sessionId || !enabled) return;

    // Don't connect if already connected or connecting
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Build WebSocket URL
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = import.meta.env.DEV ? "localhost:8000" : window.location.host;
    const wsUrl = `${protocol}//${host}/api/combat/${sessionId}/ws`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setReconnectAttempts(0);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "state" && message.data) {
          // Update React Query cache with new state
          queryClient.setQueryData<CombatSessionResponse>(
            combatKeys.detail(sessionId),
            message.data
          );
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      wsRef.current = null;

      // Don't reconnect if intentionally closed or disabled
      if (event.code === 1000 || !enabled) {
        return;
      }

      // Schedule reconnection if within attempt limit
      setReconnectAttempts((prev) => {
        const newAttempts = prev + 1;
        if (newAttempts <= maxReconnectAttempts) {
          // Exponential backoff with jitter
          const delay = Math.min(
            reconnectDelay * Math.pow(1.5, newAttempts - 1) + Math.random() * 500,
            30000
          );
          reconnectTimeoutRef.current = window.setTimeout(connect, delay);
        }
        return newAttempts;
      });
    };

    ws.onerror = () => {
      // Error handling is done in onclose
    };
  }, [sessionId, enabled, reconnectDelay, maxReconnectAttempts, queryClient]);

  // Connect on mount and when sessionId changes
  useEffect(() => {
    if (enabled && sessionId) {
      connect();
    }

    return () => {
      // Clean up on unmount or when sessionId changes
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.close(1000, "Component unmounting");
        wsRef.current = null;
      }
    };
  }, [connect, enabled, sessionId]);

  return { isConnected, reconnectAttempts };
}
