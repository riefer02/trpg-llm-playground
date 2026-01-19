import type { ActionLogEffect, HexCoord } from "../types/lancer";

import { getMarkerIconConfig } from "../combat-effects";

import type { HexLayout, PixelPoint } from "./hex";
import { axialToPixel, hexCorners, pixelToAxial } from "./hex";

export type HexGrid = {
  coords: HexCoord[];
  coordSet: Set<string>;
};

export type CanvasSize = {
  width: number;
  height: number;
};

export type GridStyle = {
  strokeStyle?: string;
  lineWidth?: number;
  fillStyle?: string;
};

export type RenderToken = {
  id: string;
  coord: HexCoord;
  color?: string;
  label?: string;
  radius?: number;
};

export type DeployableKind = "mine" | "drone" | "deployable" | "other";
export type MarkerKind = ActionLogEffect["type"] | DeployableKind;

export type RenderMarker = {
  id: string;
  coord: HexCoord;
  kind: MarkerKind;
  count?: number;
  armed?: boolean;
};

export type TokenStyle = {
  fillStyle?: string;
  strokeStyle?: string;
  lineWidth?: number;
  radius?: number;
  labelColor?: string;
  font?: string;
};

export type HoverStyle = {
  fillStyle?: string;
  strokeStyle?: string;
  lineWidth?: number;
};

export type AreaOverlay = {
  coords: HexCoord[];
  style?: HoverStyle;
};

export type MarkerStyle = {
  fillStyle?: string;
  strokeStyle?: string;
  lineWidth?: number;
  radius?: number;
  labelColor?: string;
  font?: string;
};

export type CombatRenderState = {
  grid: HexGrid;
  tokens: RenderToken[];
  markers?: RenderMarker[];
  overlays?: AreaOverlay[];
  hover?: HexCoord | null;
};

export type RenderStyles = {
  grid?: GridStyle;
  tokens?: TokenStyle;
  markers?: MarkerStyle;
  overlays?: HoverStyle;
  hover?: HoverStyle;
};

export type HoverCallback = (
  coord: HexCoord | null,
  point: PixelPoint | null,
) => void;

export type ClickCallbacks = {
  onSelect?: HoverCallback;
  onTarget?: HoverCallback;
};

export function buildHexGrid(
  radius: number,
  origin: HexCoord = { q: 0, r: 0 },
): HexGrid {
  const coords: HexCoord[] = [];
  for (let q = -radius; q <= radius; q += 1) {
    const rMin = Math.max(-radius, -q - radius);
    const rMax = Math.min(radius, -q + radius);
    for (let r = rMin; r <= rMax; r += 1) {
      coords.push({ q: q + origin.q, r: r + origin.r });
    }
  }
  const coordSet = new Set(coords.map(hexKey));
  return { coords, coordSet };
}

export function gridContains(grid: HexGrid, coord: HexCoord): boolean {
  return grid.coordSet.has(hexKey(coord));
}

export function renderCombatCanvas(
  ctx: CanvasRenderingContext2D,
  layout: HexLayout,
  state: CombatRenderState,
  styles: RenderStyles = {},
  size?: CanvasSize,
): void {
  const clearWidth = size?.width ?? ctx.canvas.width;
  const clearHeight = size?.height ?? ctx.canvas.height;
  ctx.clearRect(0, 0, clearWidth, clearHeight);
  drawHexGrid(ctx, layout, state.grid, styles.grid);
  if (state.overlays?.length) {
    drawAreaOverlays(ctx, layout, state.overlays, styles.overlays);
  }
  if (state.markers?.length) {
    drawMarkers(ctx, layout, state.markers, styles.markers);
  }
  drawTokens(ctx, layout, state.tokens, styles.tokens);
  if (state.hover) {
    drawHoverOverlay(ctx, layout, state.hover, styles.hover);
  }
}

export function drawHexGrid(
  ctx: CanvasRenderingContext2D,
  layout: HexLayout,
  grid: HexGrid,
  style: GridStyle = {},
): void {
  for (const coord of grid.coords) {
    const center = axialToPixel(coord, layout);
    drawHex(ctx, center, layout, style);
  }
}

export function drawTokens(
  ctx: CanvasRenderingContext2D,
  layout: HexLayout,
  tokens: RenderToken[],
  style: TokenStyle = {},
): void {
  for (const token of tokens) {
    const center = axialToPixel(token.coord, layout);
    const radius = token.radius ?? style.radius ?? layout.size * 0.45;
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = token.color ?? style.fillStyle ?? "#334155";
    ctx.fill();

    if (style.strokeStyle || token.color) {
      ctx.strokeStyle = style.strokeStyle ?? "#0f172a";
      ctx.lineWidth = style.lineWidth ?? 2;
      ctx.stroke();
    }

    if (token.label) {
      ctx.fillStyle = style.labelColor ?? "#f8fafc";
      ctx.font = style.font ?? "12px 'Space Grotesk', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(token.label, center.x, center.y);
    }
  }
}

export function drawMarkers(
  ctx: CanvasRenderingContext2D,
  layout: HexLayout,
  markers: RenderMarker[],
  style: MarkerStyle = {},
): void {
  for (const marker of markers) {
    const center = axialToPixel(marker.coord, layout);
    const config = getMarkerIconConfig(marker.kind, marker.armed);
    const radius = style.radius ?? layout.size * 0.22;
    const label =
      marker.count && marker.count > 1
        ? `${config.glyph}${marker.count}`
        : config.glyph;

    ctx.beginPath();
    ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = style.fillStyle ?? config.color;
    ctx.fill();

    ctx.strokeStyle = style.strokeStyle ?? "#0f172a";
    ctx.lineWidth = style.lineWidth ?? 1.5;
    ctx.stroke();

    ctx.fillStyle = style.labelColor ?? "#fff";
    ctx.font = style.font ?? `bold ${Math.round(radius)}px 'Space Grotesk', sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, center.x, center.y);
  }
}

export function drawHoverOverlay(
  ctx: CanvasRenderingContext2D,
  layout: HexLayout,
  coord: HexCoord,
  style: HoverStyle = {},
): void {
  const center = axialToPixel(coord, layout);
  const corners = hexCorners(center, layout);
  ctx.beginPath();
  ctx.moveTo(corners[0].x, corners[0].y);
  for (let i = 1; i < corners.length; i += 1) {
    ctx.lineTo(corners[i].x, corners[i].y);
  }
  ctx.closePath();

  const fillStyle = style.fillStyle ?? "rgba(59, 130, 246, 0.2)";
  ctx.fillStyle = fillStyle;
  ctx.fill();

  if (style.strokeStyle) {
    ctx.strokeStyle = style.strokeStyle;
    ctx.lineWidth = style.lineWidth ?? 2;
    ctx.stroke();
  }
}

export function drawAreaOverlays(
  ctx: CanvasRenderingContext2D,
  layout: HexLayout,
  overlays: AreaOverlay[],
  fallbackStyle: HoverStyle = {},
): void {
  for (const overlay of overlays) {
    const style = overlay.style ?? fallbackStyle;
    for (const coord of overlay.coords) {
      drawHoverOverlay(ctx, layout, coord, style);
    }
  }
}

export function getHoveredHex(
  point: PixelPoint,
  layout: HexLayout,
  grid: HexGrid,
): HexCoord | null {
  const coord = pixelToAxial(point, layout);
  if (!gridContains(grid, coord)) {
    return null;
  }
  if (!isPointInHex(point, layout, coord)) {
    return null;
  }
  return coord;
}

export function attachHoverHandlers(
  canvas: HTMLCanvasElement,
  layout: HexLayout,
  grid: HexGrid,
  onHover: HoverCallback,
): () => void {
  const handleMove = (event: PointerEvent) => {
    const point = getCanvasPoint(canvas, event);
    const hover = getHoveredHex(point, layout, grid);
    onHover(hover, point);
  };

  const handleLeave = () => {
    onHover(null, null);
  };

  canvas.addEventListener("pointermove", handleMove);
  canvas.addEventListener("pointerleave", handleLeave);

  return () => {
    canvas.removeEventListener("pointermove", handleMove);
    canvas.removeEventListener("pointerleave", handleLeave);
  };
}

export function attachClickHandlers(
  canvas: HTMLCanvasElement,
  layout: HexLayout,
  grid: HexGrid,
  callbacks: ClickCallbacks,
): () => void {
  const handleClick = (event: MouseEvent) => {
    const point = getCanvasPoint(canvas, event);
    const coord = getHoveredHex(point, layout, grid);
    callbacks.onSelect?.(coord, coord ? point : null);
  };

  const handleContextMenu = (event: MouseEvent) => {
    event.preventDefault();
    const point = getCanvasPoint(canvas, event);
    const coord = getHoveredHex(point, layout, grid);
    callbacks.onTarget?.(coord, coord ? point : null);
  };

  canvas.addEventListener("click", handleClick);
  canvas.addEventListener("contextmenu", handleContextMenu);

  return () => {
    canvas.removeEventListener("click", handleClick);
    canvas.removeEventListener("contextmenu", handleContextMenu);
  };
}

function getCanvasPoint(
  canvas: HTMLCanvasElement,
  event: MouseEvent | PointerEvent,
): PixelPoint {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function drawHex(
  ctx: CanvasRenderingContext2D,
  center: PixelPoint,
  layout: HexLayout,
  style: GridStyle,
): void {
  const corners = hexCorners(center, layout);
  ctx.beginPath();
  ctx.moveTo(corners[0].x, corners[0].y);
  for (let i = 1; i < corners.length; i += 1) {
    ctx.lineTo(corners[i].x, corners[i].y);
  }
  ctx.closePath();

  if (style.fillStyle) {
    ctx.fillStyle = style.fillStyle;
    ctx.fill();
  }

  ctx.strokeStyle = style.strokeStyle ?? "rgba(148, 163, 184, 0.6)";
  ctx.lineWidth = style.lineWidth ?? 1;
  ctx.stroke();
}

function isPointInHex(
  point: PixelPoint,
  layout: HexLayout,
  coord: HexCoord,
): boolean {
  const center = axialToPixel(coord, layout);
  const corners = hexCorners(center, layout);
  return pointInPolygon(point, corners);
}

function pointInPolygon(point: PixelPoint, polygon: PixelPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    const intersects =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

function hexKey(coord: HexCoord): string {
  return `${coord.q},${coord.r}`;
}

export type MovementPathStyle = {
  startFillStyle?: string;
  pathFillStyle?: string;
  strokeStyle?: string;
  lineWidth?: number;
  lineDash?: number[];
};

/**
 * Draw a movement path overlay on the canvas.
 * Shows start hex in blue and path hexes in green with a dashed connecting line.
 */
export function drawMovementPath(
  ctx: CanvasRenderingContext2D,
  layout: HexLayout,
  path: HexCoord[],
  style: MovementPathStyle = {},
): void {
  if (path.length === 0) return;

  const startFill = style.startFillStyle ?? "rgba(59, 130, 246, 0.3)";
  const pathFill = style.pathFillStyle ?? "rgba(34, 197, 94, 0.3)";
  const strokeColor = style.strokeStyle ?? "rgba(34, 197, 94, 0.8)";
  const lineWidth = style.lineWidth ?? 2;

  // Draw path hexes
  path.forEach((coord, index) => {
    const center = axialToPixel(coord, layout);
    const corners = hexCorners(center, layout);

    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < corners.length; i++) {
      ctx.lineTo(corners[i].x, corners[i].y);
    }
    ctx.closePath();

    // Fill with different color for start vs path
    ctx.fillStyle = index === 0 ? startFill : pathFill;
    ctx.fill();

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  });

  // Draw connecting dashed line through hex centers
  if (path.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = style.strokeStyle ?? "rgba(34, 197, 94, 0.9)";
    ctx.lineWidth = 3;
    ctx.setLineDash(style.lineDash ?? [5, 5]);

    const start = axialToPixel(path[0], layout);
    ctx.moveTo(start.x, start.y);

    for (let i = 1; i < path.length; i++) {
      const point = axialToPixel(path[i], layout);
      ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
