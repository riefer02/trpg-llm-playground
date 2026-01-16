import type { HexCoord } from "../../lib/types/lancer";
import { HEX_DIRECTIONS } from "../../lib/combat-render/aoe";
import { Button } from "../ui";

export interface DirectionPickerProps {
  onSelect: (direction: HexCoord) => void;
  onCancel: () => void;
  selectedDirection: HexCoord | null;
  patternType: "line" | "cone";
  isOpen: boolean;
}

/**
 * 6-direction picker for line/cone attacks.
 * Shows a hexagonal arrangement of direction buttons.
 */
export function DirectionPicker({
  onSelect,
  onCancel,
  selectedDirection,
  patternType,
  isOpen,
}: DirectionPickerProps) {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
      <div className="text-sm font-medium text-foreground">
        Select Direction ({patternType})
      </div>
      <div className="text-xs text-muted-foreground">
        Choose the direction for your {patternType} attack
      </div>

      <div className="flex justify-center py-2">
        <HexDirectionGrid
          selectedDirection={selectedDirection}
          onSelect={onSelect}
        />
      </div>

      <div className="flex gap-2 justify-end">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

interface HexDirectionGridProps {
  selectedDirection: HexCoord | null;
  onSelect: (direction: HexCoord) => void;
}

/**
 * Visual grid showing 6 hex directions as clickable buttons.
 * Layout follows standard hex direction naming.
 */
function HexDirectionGrid({ selectedDirection, onSelect }: HexDirectionGridProps) {
  // Direction labels for user display
  const directionLabels: Record<string, string> = {
    "1,0": "E",      // East
    "1,-1": "NE",    // Northeast
    "0,-1": "NW",    // Northwest
    "-1,0": "W",     // West
    "-1,1": "SW",    // Southwest
    "0,1": "SE",     // Southeast
  };

  const isSelected = (dir: HexCoord): boolean => {
    if (!selectedDirection) return false;
    return selectedDirection.q === dir.q && selectedDirection.r === dir.r;
  };

  const getKey = (dir: HexCoord): string => `${dir.q},${dir.r}`;

  return (
    <div className="relative w-32 h-28">
      {/* Top row: NW and NE */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 flex gap-8">
        <DirectionButton
          direction={HEX_DIRECTIONS[2]} // NW
          label={directionLabels[getKey(HEX_DIRECTIONS[2])]}
          isSelected={isSelected(HEX_DIRECTIONS[2])}
          onClick={() => onSelect(HEX_DIRECTIONS[2])}
        />
        <DirectionButton
          direction={HEX_DIRECTIONS[1]} // NE
          label={directionLabels[getKey(HEX_DIRECTIONS[1])]}
          isSelected={isSelected(HEX_DIRECTIONS[1])}
          onClick={() => onSelect(HEX_DIRECTIONS[1])}
        />
      </div>

      {/* Middle row: W and E */}
      <div className="absolute top-1/2 -translate-y-1/2 w-full flex justify-between px-0">
        <DirectionButton
          direction={HEX_DIRECTIONS[3]} // W
          label={directionLabels[getKey(HEX_DIRECTIONS[3])]}
          isSelected={isSelected(HEX_DIRECTIONS[3])}
          onClick={() => onSelect(HEX_DIRECTIONS[3])}
        />
        <DirectionButton
          direction={HEX_DIRECTIONS[0]} // E
          label={directionLabels[getKey(HEX_DIRECTIONS[0])]}
          isSelected={isSelected(HEX_DIRECTIONS[0])}
          onClick={() => onSelect(HEX_DIRECTIONS[0])}
        />
      </div>

      {/* Bottom row: SW and SE */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex gap-8">
        <DirectionButton
          direction={HEX_DIRECTIONS[4]} // SW
          label={directionLabels[getKey(HEX_DIRECTIONS[4])]}
          isSelected={isSelected(HEX_DIRECTIONS[4])}
          onClick={() => onSelect(HEX_DIRECTIONS[4])}
        />
        <DirectionButton
          direction={HEX_DIRECTIONS[5]} // SE
          label={directionLabels[getKey(HEX_DIRECTIONS[5])]}
          isSelected={isSelected(HEX_DIRECTIONS[5])}
          onClick={() => onSelect(HEX_DIRECTIONS[5])}
        />
      </div>
    </div>
  );
}

interface DirectionButtonProps {
  direction: HexCoord;
  label: string;
  isSelected: boolean;
  onClick: () => void;
}

function DirectionButton({ label, isSelected, onClick }: DirectionButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-xs font-medium transition-colors ${
        isSelected
          ? "bg-primary border-primary text-primary-foreground"
          : "bg-muted/50 border-border text-muted-foreground hover:bg-primary/10 hover:border-primary/50"
      }`}
    >
      {label}
    </button>
  );
}

/**
 * Calculate direction from one hex to another, normalized to nearest standard direction.
 */
export function calculateDirectionFromHover(
  origin: HexCoord,
  hover: HexCoord
): HexCoord | null {
  const dq = hover.q - origin.q;
  const dr = hover.r - origin.r;

  if (dq === 0 && dr === 0) {
    return null;
  }

  // Find the closest standard hex direction
  let bestDirection = HEX_DIRECTIONS[0];
  let bestDot = -Infinity;

  for (const dir of HEX_DIRECTIONS) {
    // Simple dot product approximation
    const dot = dq * dir.q + dr * dir.r;
    if (dot > bestDot) {
      bestDot = dot;
      bestDirection = dir;
    }
  }

  return bestDirection;
}

/**
 * Get preview coordinates for a pattern at given origin and direction.
 */
export function getPatternPreviewCoords(
  patternType: "line" | "cone",
  size: number,
  origin: HexCoord,
  direction: HexCoord | null
): HexCoord[] {
  if (!direction) {
    return [];
  }

  // Import dynamically to avoid circular dependencies
  // These functions are already available from aoe.ts
  if (patternType === "line") {
    return lineCoords(origin, direction, size);
  } else {
    return coneCoords(origin, direction, size);
  }
}

// Simple line calculation (duplicate of hexLineFromDirection for isolation)
function lineCoords(origin: HexCoord, direction: HexCoord, length: number): HexCoord[] {
  const results: HexCoord[] = [];
  for (let distance = 1; distance <= length; distance += 1) {
    results.push({
      q: origin.q + direction.q * distance,
      r: origin.r + direction.r * distance,
    });
  }
  return results;
}

// Simple cone calculation (uses standard Lancer cone logic)
function coneCoords(origin: HexCoord, direction: HexCoord, length: number): HexCoord[] {
  const directionIndex = HEX_DIRECTIONS.findIndex(
    (coord) => coord.q === direction.q && coord.r === direction.r
  );
  if (directionIndex === -1) {
    return [];
  }

  const left = HEX_DIRECTIONS[(directionIndex - 1 + 6) % 6];
  const results: HexCoord[] = [];

  for (let distance = 1; distance <= length; distance += 1) {
    for (let offset = 0; offset < distance; offset += 1) {
      results.push({
        q: origin.q + direction.q * distance + left.q * offset,
        r: origin.r + direction.r * distance + left.r * offset,
      });
    }
  }

  return results;
}
