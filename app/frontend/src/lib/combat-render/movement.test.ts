import { describe, expect, it } from "vitest";
import { getHexNeighbors, calculateReachableHexes } from "./movement";
import { hex } from "./hex";

describe("getHexNeighbors", () => {
  it("returns 6 neighbors for origin hex", () => {
    const neighbors = getHexNeighbors(hex(0, 0));
    expect(neighbors).toHaveLength(6);

    const keys = neighbors.map((n) => `${n.q},${n.r}`);
    expect(keys).toContain("1,0");
    expect(keys).toContain("1,-1");
    expect(keys).toContain("0,-1");
    expect(keys).toContain("-1,0");
    expect(keys).toContain("-1,1");
    expect(keys).toContain("0,1");
  });

  it("returns correct neighbors for offset hex", () => {
    const neighbors = getHexNeighbors(hex(2, 3));
    expect(neighbors).toHaveLength(6);

    const keys = neighbors.map((n) => `${n.q},${n.r}`);
    expect(keys).toContain("3,3");
    expect(keys).toContain("3,2");
    expect(keys).toContain("2,2");
    expect(keys).toContain("1,3");
    expect(keys).toContain("1,4");
    expect(keys).toContain("2,4");
  });
});

describe("calculateReachableHexes", () => {
  it("returns empty array for speed 0", () => {
    const validHexes = new Set(["0,0", "1,0", "0,1"]);
    const reachable = calculateReachableHexes(hex(0, 0), 0, validHexes);
    expect(reachable).toHaveLength(0);
  });

  it("returns adjacent hexes for speed 1", () => {
    // Build a small valid grid centered on 0,0
    const validHexes = new Set<string>();
    for (let q = -2; q <= 2; q++) {
      for (let r = -2; r <= 2; r++) {
        validHexes.add(`${q},${r}`);
      }
    }

    const reachable = calculateReachableHexes(hex(0, 0), 1, validHexes);

    // Should have 6 neighbors at cost 1
    expect(reachable).toHaveLength(6);
    expect(reachable.every((r) => r.cost === 1)).toBe(true);
  });

  it("returns all hexes within speed 2", () => {
    const validHexes = new Set<string>();
    for (let q = -3; q <= 3; q++) {
      for (let r = -3; r <= 3; r++) {
        validHexes.add(`${q},${r}`);
      }
    }

    const reachable = calculateReachableHexes(hex(0, 0), 2, validHexes);

    // Speed 2 should reach: 6 at distance 1 + 12 at distance 2 = 18 total
    expect(reachable).toHaveLength(18);
  });

  it("respects blocked hexes", () => {
    const validHexes = new Set<string>();
    for (let q = -2; q <= 2; q++) {
      for (let r = -2; r <= 2; r++) {
        validHexes.add(`${q},${r}`);
      }
    }

    // Block the hex at 1,0
    const blockedHexes = new Set(["1,0"]);

    const reachable = calculateReachableHexes(hex(0, 0), 1, validHexes, blockedHexes);

    // Should have 5 neighbors instead of 6 (1,0 is blocked)
    expect(reachable).toHaveLength(5);
    expect(reachable.some((r) => r.coord.q === 1 && r.coord.r === 0)).toBe(false);
  });

  it("accounts for difficult terrain costing 2 movement", () => {
    const validHexes = new Set<string>();
    for (let q = -3; q <= 3; q++) {
      for (let r = -3; r <= 3; r++) {
        validHexes.add(`${q},${r}`);
      }
    }

    // Make all hexes difficult
    const difficultHexes = new Set(validHexes);

    const reachable = calculateReachableHexes(hex(0, 0), 2, validHexes, undefined, difficultHexes);

    // With difficult terrain, speed 2 only reaches adjacent hexes (each costs 2)
    expect(reachable).toHaveLength(6);
    expect(reachable.every((r) => r.cost === 2)).toBe(true);
  });

  it("excludes hexes not in validHexes", () => {
    // Only 3 valid hexes
    const validHexes = new Set(["0,0", "1,0", "0,1"]);

    const reachable = calculateReachableHexes(hex(0, 0), 4, validHexes);

    // Can only reach 2 hexes (the valid neighbors)
    expect(reachable).toHaveLength(2);
  });

  it("sorts results by cost", () => {
    const validHexes = new Set<string>();
    for (let q = -3; q <= 3; q++) {
      for (let r = -3; r <= 3; r++) {
        validHexes.add(`${q},${r}`);
      }
    }

    const reachable = calculateReachableHexes(hex(0, 0), 3, validHexes);

    // Verify sorted by cost
    for (let i = 1; i < reachable.length; i++) {
      expect(reachable[i].cost).toBeGreaterThanOrEqual(reachable[i - 1].cost);
    }
  });
});
