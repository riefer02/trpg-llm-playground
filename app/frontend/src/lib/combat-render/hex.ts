import type { HexCoord } from "../types/lancer";

export type PixelPoint = {
  x: number;
  y: number;
};

export type HexLayout = {
  size: number;
  origin: PixelPoint;
};

type CubeCoord = {
  x: number;
  y: number;
  z: number;
};

const SQRT_3 = Math.sqrt(3);

export function createHexLayout(
  size: number,
  origin: PixelPoint = { x: 0, y: 0 },
): HexLayout {
  return { size, origin };
}

export function axialToPixel(coord: HexCoord, layout: HexLayout): PixelPoint {
  const x = layout.size * 1.5 * coord.q + layout.origin.x;
  const y = layout.size * SQRT_3 * (coord.r + coord.q / 2) + layout.origin.y;
  return { x, y };
}

export function pixelToAxial(point: PixelPoint, layout: HexLayout): HexCoord {
  const px = (point.x - layout.origin.x) / layout.size;
  const py = (point.y - layout.origin.y) / layout.size;
  const q = (2 / 3) * px;
  const r = (-1 / 3) * px + (SQRT_3 / 3) * py;
  const rounded = cubeRound(axialToCube({ q, r }));
  return cubeToAxial(rounded);
}

export function hexCorners(center: PixelPoint, layout: HexLayout): PixelPoint[] {
  const corners: PixelPoint[] = [];
  for (let i = 0; i < 6; i += 1) {
    const angle = (Math.PI / 180) * (60 * i);
    corners.push({
      x: center.x + layout.size * Math.cos(angle),
      y: center.y + layout.size * Math.sin(angle),
    });
  }
  return corners;
}

function axialToCube(coord: { q: number; r: number }): CubeCoord {
  const x = coord.q;
  const z = coord.r;
  const y = -x - z;
  return { x, y, z };
}

function cubeToAxial(cube: CubeCoord): HexCoord {
  return { q: cube.x, r: cube.z };
}

function cubeRound(cube: CubeCoord): CubeCoord {
  let rx = Math.round(cube.x);
  let ry = Math.round(cube.y);
  let rz = Math.round(cube.z);

  const xDiff = Math.abs(rx - cube.x);
  const yDiff = Math.abs(ry - cube.y);
  const zDiff = Math.abs(rz - cube.z);

  if (xDiff > yDiff && xDiff > zDiff) {
    rx = -ry - rz;
  } else if (yDiff > zDiff) {
    ry = -rx - rz;
  } else {
    rz = -rx - ry;
  }

  return { x: rx, y: ry, z: rz };
}

/**
 * Calculate the distance between two hexes in axial coordinates.
 * Uses the cube coordinate formula: (|dx| + |dy| + |dz|) / 2
 */
export function hexDistance(a: HexCoord, b: HexCoord): number {
  return (
    (Math.abs(a.q - b.q) +
      Math.abs(a.q + a.r - b.q - b.r) +
      Math.abs(a.r - b.r)) /
    2
  );
}

/**
 * Check if two hexes are adjacent (distance === 1).
 */
export function isAdjacent(a: HexCoord, b: HexCoord): boolean {
  return hexDistance(a, b) === 1;
}

/**
 * Calculate the total distance of a movement path.
 * Sums the distance between consecutive hexes.
 */
export function calculatePathDistance(path: HexCoord[]): number {
  if (path.length < 2) return 0;
  let total = 0;
  for (let i = 1; i < path.length; i++) {
    total += hexDistance(path[i - 1], path[i]);
  }
  return total;
}

/**
 * Check if two hex coordinates are equal.
 */
export function hexEquals(a: HexCoord, b: HexCoord): boolean {
  return a.q === b.q && a.r === b.r;
}
