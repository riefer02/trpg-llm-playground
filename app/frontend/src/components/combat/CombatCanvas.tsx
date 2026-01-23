import { useEffect, useMemo, useRef, useState } from "react";

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
} from "../../lib/combat-render/canvas";
import type { HexLayout } from "../../lib/combat-render/hex";
import type { HexCoord } from "../../lib/types/lancer";

type LayoutResolver = HexLayout | ((size: CanvasSize) => HexLayout);

export type TargetingMode = {
  active: boolean;
  validTargetIds?: string[];
  selectedTargetIds?: string[];
  maxTargets?: number;
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
  onHover?: ClickCallbacks["onSelect"];
  onSelect?: ClickCallbacks["onSelect"];
  onTarget?: ClickCallbacks["onTarget"];
  onTokenClick?: (tokenId: string) => void;
  onHexClick?: (coord: HexCoord) => void;
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
  onHover,
  onSelect,
  onTarget,
  onTokenClick,
  onHexClick,
}: CombatCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({ width, height });
  const [devicePixelRatio, setDevicePixelRatio] = useState(1);

  const resolvedLayout = useMemo(
    () => (typeof layout === "function" ? layout(canvasSize) : layout),
    [layout, canvasSize],
  );

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

    return () => {
      cleanup.forEach((fn) => fn());
    };
  }, [onHover, onSelect, onTarget, onTokenClick, onHexClick, isPathMode, resolvedLayout, state.grid, state.tokens, state.markers]);

  // Apply targeting/path mode cursor
  const cursorClass = isPathMode
    ? "cursor-pointer"
    : targetingMode?.active
      ? "cursor-crosshair"
      : "";

  return <canvas ref={canvasRef} className={`${className ?? ""} ${cursorClass}`} />;
}
