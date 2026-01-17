"""WebSocket connection manager for real-time combat state synchronization.

This module provides a session-scoped WebSocket manager that broadcasts
combat state updates to all connected clients.

Note: This is an in-memory implementation suitable for single-instance
deployments. For multi-instance deployments, use Redis pub/sub.
"""

from fastapi import WebSocket
from typing import Dict, Set


class CombatSessionManager:
    """Manages WebSocket connections per combat session."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """Accept a WebSocket connection and add it to the session's connection pool."""
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from the session's connection pool."""
        if session_id in self._connections:
            self._connections[session_id].discard(websocket)
            if not self._connections[session_id]:
                del self._connections[session_id]

    async def broadcast(self, session_id: str, message: dict) -> None:
        """Broadcast a message to all connected clients in a session.

        Automatically handles disconnected clients by removing them from the pool.
        """
        if session_id not in self._connections:
            return

        disconnected: list[WebSocket] = []
        for ws in self._connections[session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            self._connections[session_id].discard(ws)

    def get_connection_count(self, session_id: str) -> int:
        """Get the number of active connections for a session."""
        if session_id not in self._connections:
            return 0
        return len(self._connections[session_id])


# Singleton instance for the application
combat_ws_manager = CombatSessionManager()
