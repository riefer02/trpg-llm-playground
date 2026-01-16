import { describe, expect, it } from "vitest";

import type { HexCoord } from "../types/lancer";

import {
  axialToPixel,
  createHexLayout,
  hexCorners,
  pixelToAxial,
} from "./hex";

const EPSILON = 1e-6;

describe("combat-render hex helpers", () => {
  it("round-trips axial coords through pixel space", () => {
    const layout = createHexLayout(10, { x: 3, y: -7 });
    const samples: HexCoord[] = [
      { q: 0, r: 0 },
      { q: 1, r: 0 },
      { q: 0, r: 1 },
      { q: -2, r: 3 },
      { q: 4, r: -1 },
    ];

    for (const coord of samples) {
      const pixel = axialToPixel(coord, layout);
      const back = pixelToAxial(pixel, layout);
      expect(back).toEqual(coord);
    }
  });

  it("uses flat-top corner offsets at the expected radius", () => {
    const layout = createHexLayout(12);
    const center = { x: 10, y: 20 };
    const corners = hexCorners(center, layout);
    expect(corners).toHaveLength(6);

    for (const corner of corners) {
      const dx = corner.x - center.x;
      const dy = corner.y - center.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      expect(Math.abs(distance - layout.size)).toBeLessThan(EPSILON);
    }
  });

  it("matches known flat-top axial-to-pixel values", () => {
    const layout = createHexLayout(10);
    const origin = axialToPixel({ q: 0, r: 0 }, layout);
    expect(origin).toEqual({ x: 0, y: 0 });

    const q1 = axialToPixel({ q: 1, r: 0 }, layout);
    expect(q1.x).toBeCloseTo(15);
    expect(q1.y).toBeCloseTo(Math.sqrt(3) * 5);

    const r1 = axialToPixel({ q: 0, r: 1 }, layout);
    expect(r1.x).toBeCloseTo(0);
    expect(r1.y).toBeCloseTo(Math.sqrt(3) * 10);
  });
});
