import type { HexCoord } from "../types/lancer";
import { hex } from "./hex";

/** Get all 6 neighbors of a hex in axial coordinates */
export function getHexNeighbors(coord: HexCoord): HexCoord[] {
  const directions = [
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
  ] as const;
  return directions.map(([dq, dr]) => hex(coord.q + dq, coord.r + dr));
}

export type ReachableHex = {
  coord: HexCoord;
  cost: number;
};

/**
 * BFS to find all reachable hexes within a speed budget.
 * Takes into account difficult terrain (costs 2 movement instead of 1).
 * Returns hexes sorted by cost (cheapest first).
 */
export function calculateReachableHexes(
  origin: HexCoord,
  speed: number,
  validHexes: Set<string>,
  blockedHexes?: Set<string>,
  difficultHexes?: Set<string>,
): ReachableHex[] {
  const reachable: ReachableHex[] = [];
  const visited = new Map<string, number>(); // key -> lowest cost to reach
  const queue: Array<{ coord: HexCoord; cost: number }> = [
    { coord: origin, cost: 0 },
  ];

  while (queue.length > 0) {
    const { coord, cost } = queue.shift()!;
    const key = `${coord.q},${coord.r}`;

    // Skip if we've found a cheaper path
    if (visited.has(key) && visited.get(key)! <= cost) continue;
    visited.set(key, cost);

    // Add to reachable (excluding origin)
    if (cost > 0 && cost <= speed) {
      reachable.push({ coord, cost });
    }

    // Explore neighbors
    for (const neighbor of getHexNeighbors(coord)) {
      const neighborKey = `${neighbor.q},${neighbor.r}`;

      // Only consider valid grid hexes
      if (!validHexes.has(neighborKey)) continue;

      // Skip blocked hexes (occupied by other tokens, blocking terrain)
      if (blockedHexes?.has(neighborKey)) continue;

      // Calculate movement cost (difficult terrain = 2)
      const isDifficult = difficultHexes?.has(neighborKey) ?? false;
      const moveCost = isDifficult ? 2 : 1;
      const newCost = cost + moveCost;

      // Only queue if within speed and cheaper than previous visit
      if (
        newCost <= speed &&
        (!visited.has(neighborKey) || visited.get(neighborKey)! > newCost)
      ) {
        queue.push({ coord: neighbor, cost: newCost });
      }
    }
  }

  // Sort by cost for consistent ordering
  reachable.sort((a, b) => a.cost - b.cost);
  return reachable;
}
