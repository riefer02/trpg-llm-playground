import { describe, expect, it } from "vitest";

import {
  axialToPixel,
  createHexLayout,
  findPath,
  getHexNeighbors,
  hex,
  hexCorners,
  hexEquals,
  pixelToAxial,
} from "./hex";

const EPSILON = 1e-6;

describe("combat-render hex helpers", () => {
  it("round-trips axial coords through pixel space", () => {
    const layout = createHexLayout(10, { x: 3, y: -7 });
    const samples = [
      hex(0, 0),
      hex(1, 0),
      hex(0, 1),
      hex(-2, 3),
      hex(4, -1),
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
    const origin = axialToPixel(hex(0, 0), layout);
    expect(origin).toEqual({ x: 0, y: 0 });

    const q1 = axialToPixel(hex(1, 0), layout);
    expect(q1.x).toBeCloseTo(15);
    expect(q1.y).toBeCloseTo(Math.sqrt(3) * 5);

    const r1 = axialToPixel(hex(0, 1), layout);
    expect(r1.x).toBeCloseTo(0);
    expect(r1.y).toBeCloseTo(Math.sqrt(3) * 10);
  });

  it("getHexNeighbors returns 6 adjacent hexes", () => {
    const neighbors = getHexNeighbors(hex(0, 0));
    expect(neighbors).toHaveLength(6);

    // Check all neighbors are at distance 1
    const expectedNeighbors = [
      hex(1, 0),   // East
      hex(1, -1),  // Northeast
      hex(0, -1),  // Northwest
      hex(-1, 0),  // West
      hex(-1, 1),  // Southwest
      hex(0, 1),   // Southeast
    ];

    for (const expected of expectedNeighbors) {
      expect(neighbors.some(n => hexEquals(n, expected))).toBe(true);
    }
  });

  it("findPath returns single-element path for same start and end", () => {
    const start = hex(0, 0);
    const path = findPath(start, start);
    expect(path).toEqual([start]);
  });

  it("findPath finds path to adjacent hex", () => {
    const start = hex(0, 0);
    const end = hex(1, 0);
    const path = findPath(start, end);
    expect(path).toHaveLength(2);
    expect(hexEquals(path![0], start)).toBe(true);
    expect(hexEquals(path![1], end)).toBe(true);
  });

  it("findPath finds shortest path to distant hex", () => {
    const start = hex(0, 0);
    const end = hex(3, 0);
    const path = findPath(start, end);
    // Distance is 3, so path should be 4 hexes (including start and end)
    expect(path).toHaveLength(4);
    expect(hexEquals(path![0], start)).toBe(true);
    expect(hexEquals(path![path!.length - 1], end)).toBe(true);
  });

  it("findPath respects maxDistance limit", () => {
    const start = hex(0, 0);
    const end = hex(10, 0);
    // Path would need 11 hexes, but limit to 5
    const path = findPath(start, end, 5);
    expect(path).toBeNull();
  });

  it("findPath avoids blocked hexes", () => {
    const start = hex(0, 0);
    const end = hex(2, 0);
    // Block the direct path at (1, 0)
    const isBlocked = (coord: { q: number; r: number }) =>
      coord.q === 1 && coord.r === 0;
    const path = findPath(start, end, 10, isBlocked);
    // Should find alternate path that goes around the blocked hex
    expect(path).not.toBeNull();
    expect(path!.some(h => h.q === 1 && h.r === 0)).toBe(false);
    expect(hexEquals(path![path!.length - 1], end)).toBe(true);
  });
});
