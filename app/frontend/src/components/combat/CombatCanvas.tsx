import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type {
  CanvasSize,
  ClickCallbacks,
  CombatRenderState,
  RenderStyles,
  MovementPathStyle,
} from "../../lib/combat-render/canvas";
import {
  attachClickHandlers,
  attachHoverHandlers,
  renderCombatCanvas,
  drawMovementPath,
  preloadTerrainPatternsForContext,
} from "../../lib/combat-render/canvas";
import type { HexLayout } from "../../lib/combat-render/hex";
import { createHexLayout, pixelToAxial } from "../../lib/combat-render/hex";
import type { HexCoord } from "../../lib/types/lancer";
import type { ViewportState } from "../../lib/hooks/useCanvasViewport";

type LayoutResolver = HexLayout | ((size: CanvasSize) => HexLayout);

export type TargetingMode = {
  active: boolean;
  validTargetIds?: string[];
  selectedTargetIds?: string[];
  maxTargets?: number;
};

export type ContextMenuInfo = {
  coord: HexCoord;
  tokenId?: string;
  markerId?: string;
  screenPosition: { x: number; y: number };
};

export type CombatCanvasProps = {
  width: number;
  height: number;
  layout: LayoutResolver;
  state: CombatRenderState;
  styles?: RenderStyles;
  className?: string;
  resizeToParent?: boolean;
  targetingMode?: TargetingMode;
  movementPath?: HexCoord[];
  movementPathStyle?: MovementPathStyle;
  isPathMode?: boolean;
  /** Viewport state for pan/zoom */
  viewport?: ViewportState;
  /** Callback to update zoom at a specific point (for scroll wheel zoom) */
  onZoomAtPoint?: (
    delta: number,
    cursorPoint: { x: number; y: number },
    currentLayout: HexLayout
  ) => void;
  /** Callback to update pan offset */
  onPan?: (x: number, y: number) => void;
  /** Callback to zoom by delta (for keyboard zoom) */
  onZoomDelta?: (delta: number) => void;
  /** Callback to center viewport on current actor */
  onCenterOnActor?: () => void;
  onHover?: ClickCallbacks["onSelect"];
  onSelect?: ClickCallbacks["onSelect"];
  onTarget?: ClickCallbacks["onTarget"];
  onTokenClick?: (tokenId: string) => void;
  onHexClick?: (coord: HexCoord) => void;
  /** Callback for right-click context menu */
  onContextMenu?: (info: ContextMenuInfo) => void;
};

export function CombatCanvas({
  width,
  height,
  layout,
  state,
  styles,
  className,
  resizeToParent = false,
  targetingMode,
  movementPath,
  movementPathStyle,
  isPathMode = false,
  viewport,
  onZoomAtPoint,
  onPan,
  onZoomDelta,
  onCenterOnActor,
  onHover,
  onSelect,
  onTarget,
  onTokenClick,
  onHexClick,
  onContextMenu,
}: CombatCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({ width, height });
  const [devicePixelRatio, setDevicePixelRatio] = useState(1);

  // Pan drag state
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);

  // Compute resolved layout with viewport transforms applied
  const resolvedLayout = useMemo(() => {
    const baseLayout = typeof layout === "function" ? layout(canvasSize) : layout;

    // If no viewport, return base layout
    if (!viewport) {
      return baseLayout;
    }

    // Apply zoom to size and pan to origin
    return createHexLayout(baseLayout.size * viewport.zoom, {
      x: baseLayout.origin.x + viewport.pan.x,
      y: baseLayout.origin.y + viewport.pan.y,
    });
  }, [layout, canvasSize, viewport]);

  // Modify render state to highlight valid/selected targets when in targeting mode
  const renderState = useMemo(() => {
    if (!targetingMode?.active || !targetingMode.validTargetIds?.length) {
      return state;
    }

    const validSet = new Set(targetingMode.validTargetIds);
    const selectedSet = new Set(targetingMode.selectedTargetIds ?? []);

    return {
      ...state,
      tokens: state.tokens.map((token) => ({
        ...token,
        // Selected targets get blue, valid targets get green, others keep original
        color: selectedSet.has(token.id)
          ? "#3b82f6" // blue for selected
          : validSet.has(token.id)
            ? "#22c55e" // green for valid
            : token.color,
      })),
    };
  }, [state, targetingMode]);

  useEffect(() => {
    if (resizeToParent) {
      return;
    }
    setCanvasSize({ width, height });
  }, [height, resizeToParent, width]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    if (resizeToParent) {
      const parent = canvas.parentElement;
      if (!parent) {
        return;
      }
      const observer = new ResizeObserver((entries) => {
        const entry = entries[0];
        if (!entry) {
          return;
        }
        const nextWidth = Math.max(1, Math.floor(entry.contentRect.width));
        const nextHeight = Math.max(1, Math.floor(entry.contentRect.height));
        setCanvasSize((prev) =>
          prev.width === nextWidth && prev.height === nextHeight
            ? prev
            : { width: nextWidth, height: nextHeight },
        );
      });
      observer.observe(parent);
      return () => {
        observer.disconnect();
      };
    }

    return;
  }, [resizeToParent]);

  useEffect(() => {
    const updatePixelRatio = () => {
      setDevicePixelRatio(window.devicePixelRatio || 1);
    };
    updatePixelRatio();
    window.addEventListener("resize", updatePixelRatio);
    return () => {
      window.removeEventListener("resize", updatePixelRatio);
    };
  }, []);

  // Preload terrain patterns when canvas context is first available
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Preload terrain patterns for SVG-based terrain rendering
    preloadTerrainPatternsForContext(ctx).catch((err) => {
      console.warn("Failed to preload terrain patterns:", err);
    });
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    if (canvasSize.width <= 0 || canvasSize.height <= 0) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    canvas.style.width = `${canvasSize.width}px`;
    canvas.style.height = `${canvasSize.height}px`;
    canvas.width = Math.floor(canvasSize.width * devicePixelRatio);
    canvas.height = Math.floor(canvasSize.height * devicePixelRatio);
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

    renderCombatCanvas(ctx, resolvedLayout, renderState, styles, canvasSize);

    // Draw movement path overlay if present
    if (movementPath && movementPath.length > 0) {
      drawMovementPath(ctx, resolvedLayout, movementPath, movementPathStyle);
    }
  }, [canvasSize, devicePixelRatio, resolvedLayout, renderState, styles, movementPath, movementPathStyle]);

  // Wheel zoom handler
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !onZoomAtPoint) {
      return;
    }

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const cursorPoint = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
      // Normalize delta: scroll up zooms in (positive delta), scroll down zooms out
      const delta = -e.deltaY * 0.001;
      onZoomAtPoint(delta, cursorPoint, resolvedLayout);
    };

    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      canvas.removeEventListener("wheel", handleWheel);
    };
  }, [onZoomAtPoint, resolvedLayout]);

  // Pan handlers (middle mouse button drag)
  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      // Middle mouse button (button === 1)
      if (e.button === 1 && onPan && viewport) {
        e.preventDefault();
        setIsPanning(true);
        panStartRef.current = {
          x: e.clientX,
          y: e.clientY,
          panX: viewport.pan.x,
          panY: viewport.pan.y,
        };
        (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
      }
    },
    [onPan, viewport]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!isPanning || !panStartRef.current || !onPan) {
        return;
      }
      const dx = e.clientX - panStartRef.current.x;
      const dy = e.clientY - panStartRef.current.y;
      onPan(panStartRef.current.panX + dx, panStartRef.current.panY + dy);
    },
    [isPanning, onPan]
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (isPanning) {
        setIsPanning(false);
        panStartRef.current = null;
        (e.target as HTMLCanvasElement).releasePointerCapture(e.pointerId);
      }
    },
    [isPanning]
  );

  // Keyboard navigation handler
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLCanvasElement>) => {
      const PAN_STEP = 40; // pixels per keypress
      const ZOOM_STEP = 0.2;

      switch (e.key) {
        case "ArrowUp":
        case "w":
        case "W":
          e.preventDefault();
          onPan?.(viewport?.pan.x ?? 0, (viewport?.pan.y ?? 0) + PAN_STEP);
          break;
        case "ArrowDown":
        case "s":
        case "S":
          e.preventDefault();
          onPan?.(viewport?.pan.x ?? 0, (viewport?.pan.y ?? 0) - PAN_STEP);
          break;
        case "ArrowLeft":
        case "a":
        case "A":
          e.preventDefault();
          onPan?.((viewport?.pan.x ?? 0) + PAN_STEP, viewport?.pan.y ?? 0);
          break;
        case "ArrowRight":
        case "d":
        case "D":
          e.preventDefault();
          onPan?.((viewport?.pan.x ?? 0) - PAN_STEP, viewport?.pan.y ?? 0);
          break;
        case "+":
        case "=":
          e.preventDefault();
          onZoomDelta?.(ZOOM_STEP);
          break;
        case "-":
        case "_":
          e.preventDefault();
          onZoomDelta?.(-ZOOM_STEP);
          break;
        case "c":
        case "C":
          e.preventDefault();
          onCenterOnActor?.();
          break;
        case "Home":
          e.preventDefault();
          // Reset viewport
          onPan?.(0, 0);
          onZoomDelta?.(1 - (viewport?.zoom ?? 1)); // Reset to 1.0
          break;
      }
    },
    [viewport, onPan, onZoomDelta, onCenterOnActor]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const cleanup: Array<() => void> = [];
    if (onHover) {
      cleanup.push(attachHoverHandlers(canvas, resolvedLayout, state.grid, onHover));
    }

    // Wrap onSelect to also check for token clicks and hex clicks in path mode
    const handleSelect: ClickCallbacks["onSelect"] = (coord, point) => {
      // In path mode, call onHexClick for any hex click
      if (isPathMode && coord && onHexClick) {
        onHexClick(coord);
        return; // Don't process other handlers in path mode
      }

      // First, check if we clicked a token
      if (coord && onTokenClick) {
        const clickedToken = state.tokens.find(
          (t) => t.coord.q === coord.q && t.coord.r === coord.r
        );
        if (clickedToken) {
          onTokenClick(clickedToken.id);
        } else {
          // Check if we clicked a marker (deployable) - Phase 60
          const clickedMarker = state.markers?.find(
            (m) => m.coord.q === coord.q && m.coord.r === coord.r
          );
          if (clickedMarker && clickedMarker.id.startsWith("deployable:")) {
            // Extract deployable ID and pass to token click handler
            const deployableId = clickedMarker.id.replace("deployable:", "");
            onTokenClick(deployableId);
          }
        }
      }
      // Then call the original onSelect
      onSelect?.(coord, point);
    };

    if (handleSelect || onTarget) {
      cleanup.push(
        attachClickHandlers(canvas, resolvedLayout, state.grid, {
          onSelect: handleSelect,
          onTarget,
        }),
      );
    }

    // Right-click context menu handler
    if (onContextMenu) {
      const handleContextMenu = (e: MouseEvent) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Convert pixel to hex coordinate using the layout
        const hexCoord = pixelToAxial({ x, y }, resolvedLayout);

        // Check if this hex is in the grid
        const coordKey = `${hexCoord.q},${hexCoord.r}`;
        if (!state.grid.coordSet.has(coordKey)) {
          return;
        }

        // Check if we right-clicked on a token
        const clickedToken = state.tokens.find(
          (t) => t.coord.q === hexCoord.q && t.coord.r === hexCoord.r
        );

        // Check if we right-clicked on a marker (deployable)
        const clickedMarker = state.markers?.find(
          (m) => m.coord.q === hexCoord.q && m.coord.r === hexCoord.r
        );

        onContextMenu({
          coord: hexCoord,
          tokenId: clickedToken?.id,
          markerId: clickedMarker?.id,
          screenPosition: { x: e.clientX, y: e.clientY },
        });
      };

      canvas.addEventListener("contextmenu", handleContextMenu);
      cleanup.push(() => canvas.removeEventListener("contextmenu", handleContextMenu));
    }

    return () => {
      cleanup.forEach((fn) => fn());
    };
  }, [onHover, onSelect, onTarget, onTokenClick, onHexClick, isPathMode, resolvedLayout, state.grid, state.tokens, state.markers, onContextMenu]);

  // Apply targeting/path mode/panning cursor
  const cursorClass = isPanning
    ? "cursor-grabbing"
    : isPathMode
      ? "cursor-pointer"
      : targetingMode?.active
        ? "cursor-crosshair"
        : "";

  return (
    <canvas
      ref={canvasRef}
      tabIndex={0}
      className={`${className ?? ""} ${cursorClass} outline-none focus:ring-2 focus:ring-primary/50`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onKeyDown={handleKeyDown}
    />
  );
}
