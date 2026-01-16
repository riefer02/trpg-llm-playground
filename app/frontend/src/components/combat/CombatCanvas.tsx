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

export type CombatCanvasProps = {
  width: number;
  height: number;
  layout: LayoutResolver;
  state: CombatRenderState;
  styles?: RenderStyles;
  className?: string;
  resizeToParent?: boolean;
  onHover?: ClickCallbacks["onSelect"];
  onSelect?: ClickCallbacks["onSelect"];
  onTarget?: ClickCallbacks["onTarget"];
};

export function CombatCanvas({
  width,
  height,
  layout,
  state,
  styles,
  className,
  resizeToParent = false,
  onHover,
  onSelect,
  onTarget,
}: CombatCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [canvasSize, setCanvasSize] = useState<CanvasSize>({ width, height });
  const [devicePixelRatio, setDevicePixelRatio] = useState(1);

  const resolvedLayout = useMemo(
    () => (typeof layout === "function" ? layout(canvasSize) : layout),
    [layout, canvasSize],
  );

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

    renderCombatCanvas(ctx, resolvedLayout, state, styles, canvasSize);
  }, [canvasSize, devicePixelRatio, resolvedLayout, state, styles]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const cleanup: Array<() => void> = [];
    if (onHover) {
      cleanup.push(attachHoverHandlers(canvas, resolvedLayout, state.grid, onHover));
    }

    if (onSelect || onTarget) {
      cleanup.push(
        attachClickHandlers(canvas, resolvedLayout, state.grid, {
          onSelect,
          onTarget,
        }),
      );
    }

    return () => {
      cleanup.forEach((fn) => fn());
    };
  }, [onHover, onSelect, onTarget, resolvedLayout, state.grid]);

  return <canvas ref={canvasRef} className={className} />;
}
