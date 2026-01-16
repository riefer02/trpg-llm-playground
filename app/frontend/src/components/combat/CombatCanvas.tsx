import { useEffect, useMemo, useRef, useState } from "react";

import type {
  CanvasSize,
  ClickCallbacks,
  CombatRenderState,
  RenderStyles,
} from "../../lib/combat-render/canvas";
import {
  attachClickHandlers,
  attachHoverHandlers,
  renderCombatCanvas,
} from "../../lib/combat-render/canvas";
import type { HexLayout } from "../../lib/combat-render/hex";

type LayoutResolver = HexLayout | ((size: CanvasSize) => HexLayout);

export type TargetingMode = {
  active: boolean;
  validTargetIds?: string[];
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
  onHover?: ClickCallbacks["onSelect"];
  onSelect?: ClickCallbacks["onSelect"];
  onTarget?: ClickCallbacks["onTarget"];
  onTokenClick?: (tokenId: string) => void;
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
  onHover,
  onSelect,
  onTarget,
  onTokenClick,
}: CombatCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({ width, height });
  const [devicePixelRatio, setDevicePixelRatio] = useState(1);

  const resolvedLayout = useMemo(
    () => (typeof layout === "function" ? layout(canvasSize) : layout),
    [layout, canvasSize],
  );

  // Modify render state to highlight valid targets when in targeting mode
  const renderState = useMemo(() => {
    if (!targetingMode?.active || !targetingMode.validTargetIds?.length) {
      return state;
    }

    const validSet = new Set(targetingMode.validTargetIds);
    return {
      ...state,
      tokens: state.tokens.map((token) => ({
        ...token,
        // Add pulsing ring effect for valid targets
        color: validSet.has(token.id)
          ? "#22c55e" // green for valid targets
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
  }, [canvasSize, devicePixelRatio, resolvedLayout, renderState, styles]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const cleanup: Array<() => void> = [];
    if (onHover) {
      cleanup.push(attachHoverHandlers(canvas, resolvedLayout, state.grid, onHover));
    }

    // Wrap onSelect to also check for token clicks
    const handleSelect: ClickCallbacks["onSelect"] = (coord, point) => {
      // First, check if we clicked a token
      if (coord && onTokenClick) {
        const clickedToken = state.tokens.find(
          (t) => t.coord.q === coord.q && t.coord.r === coord.r
        );
        if (clickedToken) {
          onTokenClick(clickedToken.id);
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
  }, [onHover, onSelect, onTarget, onTokenClick, resolvedLayout, state.grid, state.tokens]);

  // Apply targeting mode cursor
  const cursorClass = targetingMode?.active ? "cursor-crosshair" : "";

  return <canvas ref={canvasRef} className={`${className ?? ""} ${cursorClass}`} />;
}
