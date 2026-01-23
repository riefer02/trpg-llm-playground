import type { HexCoord } from "../types/lancer";
import { hex } from "./hex";

export const HEX_DIRECTIONS: HexCoord[] = [
  hex(1, 0),
  hex(1, -1),
  hex(0, -1),
  hex(-1, 0),
  hex(-1, 1),
  hex(0, 1),
];

export function normalizeHexDirection(direction: HexCoord): HexCoord | null {
  if (direction.q === 0 && direction.r === 0) {
    return null;
  }
  for (const unit of HEX_DIRECTIONS) {
    if (unit.q === 0) {
      if (direction.q !== 0) {
        continue;
      }
      if (direction.r % unit.r === 0 && direction.r / unit.r > 0) {
        return unit;
      }
      continue;
    }
    if (unit.r === 0) {
      if (direction.r !== 0) {
        continue;
      }
      if (direction.q % unit.q === 0 && direction.q / unit.q > 0) {
        return unit;
      }
      continue;
    }
    if (direction.q % unit.q !== 0 || direction.r % unit.r !== 0) {
      continue;
    }
    const scaleQ = direction.q / unit.q;
    const scaleR = direction.r / unit.r;
    if (scaleQ === scaleR && scaleQ > 0) {
      return unit;
    }
  }
  return null;
}

export function hexAdd(a: HexCoord, b: HexCoord): HexCoord {
  return hex(a.q + b.q, a.r + b.r);
}

export function hexScale(coord: HexCoord, scale: number): HexCoord {
  return hex(coord.q * scale, coord.r * scale);
}

export function hexDistance(a: HexCoord, b: HexCoord): number {
  const aS = -a.q - a.r;
  const bS = -b.q - b.r;
  return Math.max(
    Math.abs(a.q - b.q),
    Math.abs(a.r - b.r),
    Math.abs(aS - bS),
  );
}

export function hexesInRadius(center: HexCoord, radius: number): HexCoord[] {
  const results: HexCoord[] = [];
  for (let dq = -radius; dq <= radius; dq += 1) {
    for (let dr = -radius; dr <= radius; dr += 1) {
      const coord = hex(center.q + dq, center.r + dr);
      if (hexDistance(center, coord) <= radius) {
        results.push(coord);
      }
    }
  }
  return results;
}

export function hexLineFromDirection(
  origin: HexCoord,
  direction: HexCoord,
  length: number,
): HexCoord[] {
  const step = normalizeHexDirection(direction);
  if (!step || length <= 0) {
    return [];
  }
  const results: HexCoord[] = [];
  for (let distance = 1; distance <= length; distance += 1) {
    results.push(hex(
      origin.q + step.q * distance,
      origin.r + step.r * distance,
    ));
  }
  return results;
}

export function hexCone(
  origin: HexCoord,
  direction: HexCoord,
  length: number,
): HexCoord[] {
  const step = normalizeHexDirection(direction);
  if (!step || length <= 0) {
    return [];
  }
  const directionIndex = HEX_DIRECTIONS.findIndex(
    (coord) => coord.q === step.q && coord.r === step.r,
  );
  const left =
    HEX_DIRECTIONS[
      (directionIndex - 1 + HEX_DIRECTIONS.length) % HEX_DIRECTIONS.length
    ];
  const results: HexCoord[] = [];
  for (let distance = 1; distance <= length; distance += 1) {
    for (let offset = 0; offset < distance; offset += 1) {
      results.push(
        hexAdd(
          origin,
          hexAdd(hexScale(step, distance), hexScale(left, offset)),
        ),
      );
    }
  }
  return results;
}

export function hexConeCentered(
  origin: HexCoord,
  direction: HexCoord,
  length: number,
): HexCoord[] {
  const step = normalizeHexDirection(direction);
  if (!step || length <= 0) {
    return [];
  }
  const stepCube = axialToCube(step);
  const axes = [stepCube[0], stepCube[1], stepCube[2]];
  const forwardIdx = axes.indexOf(1);
  const lateralIdx = axes.indexOf(0);
  const backwardIdx = axes.indexOf(-1);
  const originCube = axialToCube(origin);
  const results: HexCoord[] = [];

  for (let distance = 1; distance <= length; distance += 1) {
    for (let offset = -(distance - 1); offset < distance; offset += 1) {
      const cube = [0, 0, 0];
      cube[forwardIdx] = distance;
      cube[lateralIdx] = offset;
      cube[backwardIdx] = -distance - offset;
      results.push(
        cubeToAxial([
          originCube[0] + cube[0],
          originCube[1] + cube[1],
          originCube[2] + cube[2],
        ]),
      );
    }
  }

  return results;
}

function axialToCube(coord: HexCoord): [number, number, number] {
  return [coord.q, coord.r, -coord.q - coord.r];
}

function cubeToAxial(cube: [number, number, number]): HexCoord {
  return hex(cube[0], cube[1]);
}
