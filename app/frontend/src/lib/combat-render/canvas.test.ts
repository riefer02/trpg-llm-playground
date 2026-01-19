import { describe, expect, it } from "vitest";

import { axialToPixel, createHexLayout } from "./hex";

import {
  attachClickHandlers,
  buildHexGrid,
  getHoveredHex,
  getRenderPassOrder,
  gridContains,
} from "./canvas";

describe("combat-render canvas helpers", () => {
  it("builds a hex range with the expected tile count", () => {
    const radius0 = buildHexGrid(0);
    const radius1 = buildHexGrid(1);
    const radius2 = buildHexGrid(2);

    expect(radius0.coords).toHaveLength(1);
    expect(radius1.coords).toHaveLength(7);
    expect(radius2.coords).toHaveLength(19);
  });

  it("tracks grid membership by axial coordinate", () => {
    const grid = buildHexGrid(1);
    expect(gridContains(grid, { q: 0, r: 0 })).toBe(true);
    expect(gridContains(grid, { q: 2, r: 0 })).toBe(false);
  });

  it("applies the grid origin offset", () => {
    const grid = buildHexGrid(0, { q: 2, r: -1 });
    expect(grid.coords).toEqual([{ q: 2, r: -1 }]);
    expect(gridContains(grid, { q: 2, r: -1 })).toBe(true);
  });

  it("detects hoverable hexes within the grid bounds", () => {
    const grid = buildHexGrid(1);
    const layout = createHexLayout(10);
    const center = axialToPixel({ q: 0, r: 0 }, layout);

    const hover = getHoveredHex({ x: center.x + 1, y: center.y + 1 }, layout, grid);
    expect(hover).toEqual({ q: 0, r: 0 });

    const outside = getHoveredHex({ x: 500, y: 500 }, layout, grid);
    expect(outside).toBeNull();
  });

  it("handles hover detection with a shifted layout origin", () => {
    const grid = buildHexGrid(1);
    const layout = createHexLayout(10, { x: 80, y: 40 });
    const center = axialToPixel({ q: 1, r: -1 }, layout);
    const hover = getHoveredHex({ x: center.x + 2, y: center.y - 1 }, layout, grid);
    expect(hover).toEqual({ q: 1, r: -1 });
  });

  it("routes click and contextmenu events to select/target callbacks", () => {
    const grid = buildHexGrid(1);
    const layout = createHexLayout(10);
    const center = axialToPixel({ q: 0, r: 0 }, layout);

    const handlers: Record<string, Array<(event: MouseEvent) => void>> = {};
    const canvas = {
      addEventListener: (type: string, handler: (event: MouseEvent) => void) => {
        handlers[type] = handlers[type] ?? [];
        handlers[type].push(handler);
      },
      removeEventListener: (
        type: string,
        handler: (event: MouseEvent) => void,
      ) => {
        handlers[type] = (handlers[type] ?? []).filter(
          (existing) => existing !== handler,
        );
      },
      getBoundingClientRect: () =>
        ({
          left: 0,
          top: 0,
        }) as DOMRect,
    } as unknown as HTMLCanvasElement;

    let selected = null;
    let targeted = null;
    const detach = attachClickHandlers(canvas, layout, grid, {
      onSelect: (coord) => {
        selected = coord;
      },
      onTarget: (coord) => {
        targeted = coord;
      },
    });

    const clickEvent = {
      clientX: center.x + 1,
      clientY: center.y + 1,
      preventDefault: () => {},
    } as MouseEvent;
    handlers.click?.forEach((handler) => handler(clickEvent));
    expect(selected).toEqual({ q: 0, r: 0 });

    const targetEvent = {
      clientX: center.x + 1,
      clientY: center.y + 1,
      preventDefault: () => {},
    } as MouseEvent;
    handlers.contextmenu?.forEach((handler) => handler(targetEvent));
    expect(targeted).toEqual({ q: 0, r: 0 });

    detach();
  });

  it("builds render passes in terrain-first order", () => {
    const state = {
      grid: buildHexGrid(0),
      tokens: [{ id: "alpha", coord: { q: 0, r: 0 } }],
      terrain: [{ coord: { q: 0, r: 0 }, difficult: true }],
      overlays: [{ coords: [{ q: 0, r: 0 }] }],
      markers: [{ id: "marker:1", coord: { q: 0, r: 0 }, kind: "mine" }],
      hover: { q: 0, r: 0 },
    };

    expect(getRenderPassOrder(state)).toEqual([
      "grid",
      "terrain",
      "overlays",
      "markers",
      "tokens",
      "hover",
    ]);
  });
});
