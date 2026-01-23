import { useCallback, useState } from "react";
import type { HexCoord } from "../types/lancer";
import { axialToPixel, type HexLayout } from "../combat-render/hex";

export type ViewportState = {
  pan: { x: number; y: number };
  zoom: number;
};

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3.0;
const DEFAULT_ZOOM = 1.0;

function clampZoom(zoom: number): number {
  return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom));
}

export function useCanvasViewport() {
  const [viewport, setViewport] = useState<ViewportState>({
    pan: { x: 0, y: 0 },
    zoom: DEFAULT_ZOOM,
  });

  const setPan = useCallback((x: number, y: number) => {
    setViewport((prev) => ({
      ...prev,
      pan: { x, y },
    }));
  }, []);

  const setZoom = useCallback((level: number) => {
    setViewport((prev) => ({
      ...prev,
      zoom: clampZoom(level),
    }));
  }, []);

  const resetViewport = useCallback(() => {
    setViewport({
      pan: { x: 0, y: 0 },
      zoom: DEFAULT_ZOOM,
    });
  }, []);

  /**
   * Center the viewport on a specific hex coordinate.
   * Requires the current layout to calculate the pixel position.
   */
  const centerOnCoord = useCallback(
    (coord: HexCoord, layout: HexLayout, canvasSize: { width: number; height: number }) => {
      // Calculate where the hex is in pixel space (without current pan)
      const baseLayout: HexLayout = {
        size: layout.size,
        origin: { x: canvasSize.width / 2, y: canvasSize.height / 2 },
      };
      const hexPixel = axialToPixel(coord, baseLayout);

      // Pan offset to center this hex
      const panX = canvasSize.width / 2 - hexPixel.x;
      const panY = canvasSize.height / 2 - hexPixel.y;

      setViewport((prev) => ({
        ...prev,
        pan: { x: panX, y: panY },
      }));
    },
    []
  );

  /**
   * Zoom while keeping a specific pixel point fixed (zoom under cursor).
   * This calculates the new pan offset to maintain the cursor position.
   */
  const zoomAtPoint = useCallback(
    (
      delta: number,
      cursorPoint: { x: number; y: number },
      currentLayout: HexLayout
    ) => {
      setViewport((prev) => {
        const newZoom = clampZoom(prev.zoom + delta);
        if (newZoom === prev.zoom) return prev;

        // Calculate the scale change ratio
        const zoomRatio = newZoom / prev.zoom;

        // The cursor point relative to the current origin
        const dx = cursorPoint.x - currentLayout.origin.x;
        const dy = cursorPoint.y - currentLayout.origin.y;

        // After zooming, we need to adjust pan so the hex under cursor stays there
        // New pan = old pan - (cursor offset * (zoomRatio - 1))
        const newPanX = prev.pan.x - dx * (zoomRatio - 1);
        const newPanY = prev.pan.y - dy * (zoomRatio - 1);

        return {
          pan: { x: newPanX, y: newPanY },
          zoom: newZoom,
        };
      });
    },
    []
  );

  return {
    viewport,
    setPan,
    setZoom,
    resetViewport,
    centerOnCoord,
    zoomAtPoint,
    MIN_ZOOM,
    MAX_ZOOM,
  };
}
