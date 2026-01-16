import { describe, expect, it } from "vitest";

import type { HexCoord } from "../types/lancer";
import {
  hexCone,
  hexConeCentered,
  hexLineFromDirection,
  hexesInRadius,
  normalizeHexDirection,
} from "./aoe";

const key = (coord: HexCoord) => `${coord.q},${coord.r}`;

describe("aoe helpers", () => {
  it("normalizes axial directions", () => {
    expect(normalizeHexDirection({ q: 0, r: 0 })).toBeNull();
    expect(normalizeHexDirection({ q: 2, r: 0 })).toEqual({ q: 1, r: 0 });
    expect(normalizeHexDirection({ q: -2, r: 0 })).toEqual({ q: -1, r: 0 });
    expect(normalizeHexDirection({ q: 1, r: 1 })).toBeNull();
  });

  it("returns hexes in a radius", () => {
    const center = { q: 0, r: 0 };
    const radius1 = hexesInRadius(center, 1);
    const radius2 = hexesInRadius(center, 2);
    expect(radius1).toHaveLength(7);
    expect(radius2).toHaveLength(19);
    expect(radius1.map(key)).toContain("0,0");
  });

  it("builds a line from direction", () => {
    const origin = { q: 0, r: 0 };
    const line = hexLineFromDirection(origin, { q: 2, r: 0 }, 3);
    expect(line).toEqual([
      { q: 1, r: 0 },
      { q: 2, r: 0 },
      { q: 3, r: 0 },
    ]);
    expect(hexLineFromDirection(origin, { q: 0, r: 0 }, 3)).toEqual([]);
  });

  it("builds a cone from direction", () => {
    const origin = { q: 0, r: 0 };
    const cone = hexCone(origin, { q: 1, r: 0 }, 2);
    expect(cone.map(key)).toEqual(["1,0", "2,0", "2,1"]);
  });

  it("builds a centered cone from direction", () => {
    const origin = { q: 0, r: 0 };
    const cone = hexConeCentered(origin, { q: 1, r: 0 }, 2);
    expect(cone.map(key)).toEqual(["1,0", "2,-1", "2,0", "2,1"]);
  });
});
