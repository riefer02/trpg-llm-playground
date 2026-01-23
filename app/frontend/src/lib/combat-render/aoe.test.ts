import { describe, expect, it } from "vitest";

import type { HexCoord } from "../types/lancer";
import {
  hexCone,
  hexConeCentered,
  hexLineFromDirection,
  hexesInRadius,
  normalizeHexDirection,
} from "./aoe";
import { hex } from "./hex";

const key = (coord: HexCoord) => `${coord.q},${coord.r}`;

describe("aoe helpers", () => {
  it("normalizes axial directions", () => {
    expect(normalizeHexDirection(hex(0, 0))).toBeNull();
    expect(normalizeHexDirection(hex(2, 0))).toEqual(hex(1, 0));
    expect(normalizeHexDirection(hex(-2, 0))).toEqual(hex(-1, 0));
    expect(normalizeHexDirection(hex(1, 1))).toBeNull();
  });

  it("returns hexes in a radius", () => {
    const center = hex(0, 0);
    const radius1 = hexesInRadius(center, 1);
    const radius2 = hexesInRadius(center, 2);
    expect(radius1).toHaveLength(7);
    expect(radius2).toHaveLength(19);
    expect(radius1.map(key)).toContain("0,0");
  });

  it("builds a line from direction", () => {
    const origin = hex(0, 0);
    const line = hexLineFromDirection(origin, hex(2, 0), 3);
    expect(line).toEqual([
      hex(1, 0),
      hex(2, 0),
      hex(3, 0),
    ]);
    expect(hexLineFromDirection(origin, hex(0, 0), 3)).toEqual([]);
  });

  it("builds a cone from direction", () => {
    const origin = hex(0, 0);
    const cone = hexCone(origin, hex(1, 0), 2);
    expect(cone.map(key)).toEqual(["1,0", "2,0", "2,1"]);
  });

  it("builds a centered cone from direction", () => {
    const origin = hex(0, 0);
    const cone = hexConeCentered(origin, hex(1, 0), 2);
    expect(cone.map(key)).toEqual(["1,0", "2,-1", "2,0", "2,1"]);
  });
});
