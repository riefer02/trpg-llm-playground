import type {
  ActionUse,
  AttackPatternDefinition,
  CombatRound,
  CombatTurn,
  CombatantState,
  DeployableState,
  HexCoord,
  HexPosition,
  MechCombatScenario,
  Side,
  TerrainHex,
} from "../types/lancer";

import type {
  AreaOverlay,
  CombatRenderState,
  HexGrid,
  HoverStyle,
  RenderMarker,
  RenderTerrainTile,
  RenderToken,
} from "./canvas";
import { buildHexGrid } from "./canvas";
import { hex } from "./hex";
import { mapMissionTerrainToPattern } from "./terrain-patterns";
import {
  hexCone,
  hexConeCentered,
  hexDistance,
  hexesInRadius,
  hexLineFromDirection,
  normalizeHexDirection,
} from "./aoe";
import { calculateReachableHexes } from "./movement";

export type CombatRenderAdapterInput = {
  scenario: MechCombatScenario;
  round?: CombatRound | null;
  turn?: CombatTurn | null;
  action?: ActionUse | null;
  attackPattern?: AttackPatternDefinition | null;
  patternOrigin?: HexPosition | null;
  patternDirection?: HexCoord | null;
  actorId?: string | null;
  hover?: HexCoord | null;
  gridRadius?: number;
  gridOrigin?: HexCoord;
  overlayStyle?: HoverStyle;
  tokenColors?: Partial<Record<Side, string>>;
  /** Terrain type for visual pattern rendering (forest, urban, water, etc.) */
  terrainType?: string;
};

export type RenderTokenMetadata = {
  id: string;
  name: string;
  side: CombatantState["side"];
  kind: CombatantState["kind"];
  size: CombatantState["stats"]["size"];
  position: HexPosition | null;
  hasPosition: boolean;
  statusCount: number;
};

export type OverlayMetadata = {
  id: string;
  actionId?: string;
  actorId?: string | null;
  targetId?: string | null;
  pattern: AttackPatternDefinition["pattern"];
  size: AttackPatternDefinition["size"];
  origin?: HexPosition | null;
  direction?: HexCoord | null;
  source: "action" | "pattern";
  coordCount: number;
};

export type CombatRenderAdapterOutput = {
  state: CombatRenderState;
  tokenMetadata: Record<string, RenderTokenMetadata>;
  overlayMetadata: OverlayMetadata[];
};

const DEFAULT_GRID_RADIUS = 4;

const DEFAULT_SIDE_COLORS: Record<Side, string> = {
  players: "#1d4ed8",     // Blue for player team
  hostiles: "#dc2626",    // Red/crimson for enemy team (hostile)
  neutral: "#10b981",     // Green for neutral
};

const PATTERN_OVERLAY_STYLES: Record<string, HoverStyle> = {
  line: {
    fillStyle: "rgba(248, 113, 113, 0.18)",
    strokeStyle: "rgba(248, 113, 113, 0.5)",
    lineWidth: 1.5,
  },
  cone: {
    fillStyle: "rgba(251, 191, 36, 0.18)",
    strokeStyle: "rgba(251, 191, 36, 0.5)",
    lineWidth: 1.5,
  },
  blast: {
    fillStyle: "rgba(16, 185, 129, 0.2)",
    strokeStyle: "rgba(16, 185, 129, 0.6)",
    lineWidth: 1.5,
  },
  burst: {
    fillStyle: "rgba(139, 92, 246, 0.2)",
    strokeStyle: "rgba(139, 92, 246, 0.6)",
    lineWidth: 1.5,
  },
};

export const MOVEMENT_OVERLAY_STYLES = {
  easy: {
    fillStyle: "rgba(34, 197, 94, 0.2)", // green-500
    strokeStyle: "rgba(34, 197, 94, 0.5)",
    lineWidth: 1,
  },
  atMax: {
    fillStyle: "rgba(234, 179, 8, 0.2)", // yellow-500
    strokeStyle: "rgba(234, 179, 8, 0.5)",
    lineWidth: 1,
  },
} as const;

export function adaptCombatScenario(
  input: CombatRenderAdapterInput,
): CombatRenderAdapterOutput {
  const combatants = input.scenario.combatants ?? [];
  const combatantsById = new Map(
    combatants.map((combatant) => [combatant.id, combatant]),
  );

  const activeActorId = input.actorId ?? input.turn?.actor_id ?? null;
  const { tokens, tokenMetadata } = buildTokens(
    combatants,
    input.tokenColors,
    activeActorId,
  );
  const terrainTiles = buildTerrainTiles(
    input.scenario.terrain ?? null,
    input.terrainType,
  );
  const markers = buildMarkers(combatants, input.scenario.deployables);

  const overlayBuild = buildOverlays({
    action: input.action ?? resolveActionFromTurn(input.turn),
    actorId: input.actorId ?? input.turn?.actor_id ?? null,
    combatantsById,
    overlayStyle: input.overlayStyle,
  });

  if (input.attackPattern) {
    const patternOverlay = buildOverlayFromPattern({
      pattern: input.attackPattern,
      origin: input.patternOrigin ?? null,
      direction: input.patternDirection ?? null,
      overlayStyle: input.overlayStyle,
    });
    if (patternOverlay) {
      overlayBuild.overlays.push(patternOverlay.overlay);
      overlayBuild.overlayMetadata.push(patternOverlay.meta);
    }
  }

  const grid = resolveGrid({
    origin: input.gridOrigin ?? hex(0, 0),
    radius: input.gridRadius,
    tokens,
    markers,
    overlays: overlayBuild.overlays,
    scenario: input.scenario,
  });

  return {
    state: {
      grid,
      tokens,
      terrain: terrainTiles.length ? terrainTiles : undefined,
      markers,
      overlays: overlayBuild.overlays,
      hover: input.hover ?? null,
    },
    tokenMetadata,
    overlayMetadata: overlayBuild.overlayMetadata,
  };
}

type OverlayBuildResult = {
  overlays: AreaOverlay[];
  overlayMetadata: OverlayMetadata[];
};

type OverlayBuildInput = {
  action?: ActionUse | null;
  actorId: string | null;
  combatantsById: Map<string, CombatantState>;
  overlayStyle?: HoverStyle;
};

type PatternOverlayInput = {
  pattern: AttackPatternDefinition;
  origin: HexPosition | null;
  direction: HexCoord | null;
  overlayStyle?: HoverStyle;
};

function buildTokens(
  combatants: CombatantState[],
  tokenColors?: Partial<Record<Side, string>>,
  activeActorId?: string | null,
): {
  tokens: RenderToken[];
  tokenMetadata: Record<string, RenderTokenMetadata>;
} {
  const tokens: RenderToken[] = [];
  const tokenMetadata: Record<string, RenderTokenMetadata> = {};
  const colors = { ...DEFAULT_SIDE_COLORS, ...tokenColors };

  for (const combatant of combatants) {
    const position = combatant.position ?? null;
    const hasPosition = Boolean(position?.coord);
    tokenMetadata[combatant.id] = {
      id: combatant.id,
      name: combatant.name,
      side: combatant.side,
      kind: combatant.kind,
      size: combatant.stats.size,
      position,
      hasPosition,
      statusCount: combatant.statuses?.length ?? 0,
    };

    if (!position?.coord) {
      continue;
    }

    tokens.push({
      id: combatant.id,
      coord: position.coord,
      color: colors[combatant.side],
      label: labelFromName(combatant.name),
      isActive: combatant.id === activeActorId,
      side: combatant.side,
    });
  }

  return { tokens, tokenMetadata };
}

function buildOverlays(input: OverlayBuildInput): OverlayBuildResult {
  const overlays: AreaOverlay[] = [];
  const overlayMetadata: OverlayMetadata[] = [];
  const action = input.action;
  if (!action?.area_pattern) {
    return { overlays, overlayMetadata };
  }

  const actor = input.actorId
    ? input.combatantsById.get(input.actorId)
    : undefined;
  const actorPosition = actor?.position ?? null;
  const targetPosition = resolveTargetPosition(action, input.combatantsById);

  const overlayResult = buildOverlayFromAction({
    action,
    actorPosition,
    targetPosition,
    overlayStyle: input.overlayStyle,
    actorId: input.actorId ?? undefined,
  });

  if (!overlayResult) {
    return { overlays, overlayMetadata };
  }

  overlays.push(overlayResult.overlay);
  overlayMetadata.push(overlayResult.meta);
  return { overlays, overlayMetadata };
}

function buildMarkers(
  combatants: CombatantState[],
  deployables: Record<string, DeployableState> | undefined,
): RenderMarker[] {
  const markers: RenderMarker[] = [];

  // Thrown weapons from combatant inventories
  const thrownByCoord = new Map<
    string,
    { coord: HexCoord; count: number }
  >();

  for (const combatant of combatants) {
    for (const mount of combatant.inventory?.mounts ?? []) {
      for (const weapon of mount.weapons ?? []) {
        if (!weapon.thrown_coord) {
          continue;
        }
        const coord = weapon.thrown_coord;
        const key = `${coord.q},${coord.r}`;
        const entry = thrownByCoord.get(key) ?? { coord, count: 0 };
        entry.count += 1;
        thrownByCoord.set(key, entry);
      }
    }
  }

  for (const [key, entry] of thrownByCoord.entries()) {
    markers.push({
      id: `weapon_thrown:${key}`,
      coord: entry.coord,
      kind: "weapon_thrown",
      count: entry.count,
    });
  }

  // Deployables (mines, drones, etc.)
  if (deployables) {
    for (const [id, deployable] of Object.entries(deployables)) {
      // Skip destroyed deployables
      if (deployable.is_destroyed) {
        continue;
      }

      markers.push({
        id: `deployable:${id}`,
        coord: deployable.position.coord,
        kind: deployable.kind,
        armed: deployable.is_armed,
      });
    }
  }

  return markers;
}

function buildTerrainTiles(
  terrain: MechCombatScenario["terrain"] | null,
  scenarioTileSet?: string,
): RenderTerrainTile[] {
  if (!terrain?.tiles?.length) {
    return [];
  }

  // Map scenario tile_set to pattern type
  const terrainType = scenarioTileSet 
    ? mapMissionTerrainToPattern(scenarioTileSet) 
    : undefined;

  return terrain.tiles.map((tile: TerrainHex) => ({
    coord: tile.coord,
    elevation: tile.elevation ?? 0,
    difficult: tile.difficult ?? false,
    dangerous: tile.dangerous ?? false,
    providesSoftCover: tile.provides_soft_cover ?? false,
    providesHardCover: tile.provides_hard_cover ?? false,
    blocksLineOfSight: tile.blocks_line_of_sight ?? false,
    terrainType,
  }));
}

function buildOverlayFromAction({
  action,
  actorPosition,
  targetPosition,
  overlayStyle,
  actorId,
}: {
  action: ActionUse;
  actorPosition: HexPosition | null;
  targetPosition: HexPosition | null;
  overlayStyle?: HoverStyle;
  actorId?: string;
}): { overlay: AreaOverlay; meta: OverlayMetadata } | null {
  if (!action.area_pattern) {
    return null;
  }

  const pattern = action.area_pattern;
  const origin = resolveAreaOrigin(action, actorPosition, targetPosition);

  let coords: HexCoord[] = [];
  let source: OverlayMetadata["source"] = "pattern";

  if (action.area_affected?.length) {
    coords = dedupeCoords(action.area_affected);
    source = "action";
  } else if (origin) {
    coords = calculatePatternCoords(
      pattern,
      origin,
      action.area_direction ?? null,
    );
  }

  if (!coords.length) {
    return null;
  }

  const overlay: AreaOverlay = {
    coords,
    style: overlayStyle ?? PATTERN_OVERLAY_STYLES[pattern.pattern],
  };

  const meta: OverlayMetadata = {
    id: `overlay:${action.action_id}`,
    actionId: action.action_id,
    actorId,
    targetId: action.target_id,
    pattern: pattern.pattern,
    size: pattern.size,
    origin,
    direction: action.area_direction ?? null,
    source,
    coordCount: coords.length,
  };

  return { overlay, meta };
}

function buildOverlayFromPattern(
  input: PatternOverlayInput,
): { overlay: AreaOverlay; meta: OverlayMetadata } | null {
  if (!input.origin) {
    return null;
  }
  const coords = calculatePatternCoords(
    input.pattern,
    input.origin,
    input.direction,
  );
  if (!coords.length) {
    return null;
  }

  const overlay: AreaOverlay = {
    coords,
    style: input.overlayStyle ?? PATTERN_OVERLAY_STYLES[input.pattern.pattern],
  };

  const meta: OverlayMetadata = {
    id: `overlay:pattern:${input.pattern.pattern}:${input.pattern.size}`,
    pattern: input.pattern.pattern,
    size: input.pattern.size,
    origin: input.origin,
    direction: input.direction,
    source: "pattern",
    coordCount: coords.length,
  };

  return { overlay, meta };
}

function resolveActionFromTurn(turn?: CombatTurn | null): ActionUse | null {
  if (!turn?.actions?.length) {
    return null;
  }
  return turn.actions[turn.actions.length - 1] ?? null;
}

function resolveTargetPosition(
  action: ActionUse,
  combatantsById: Map<string, CombatantState>,
): HexPosition | null {
  if (action.target_position) {
    return action.target_position;
  }
  if (action.target_positions?.length) {
    return action.target_positions[0] ?? null;
  }
  if (action.target_id) {
    return combatantsById.get(action.target_id)?.position ?? null;
  }
  const targetIds = action.target_ids ?? [];
  for (const targetId of targetIds) {
    const position = combatantsById.get(targetId)?.position;
    if (position) {
      return position;
    }
  }
  return null;
}

function resolveAreaOrigin(
  action: ActionUse,
  actorPosition: HexPosition | null,
  targetPosition: HexPosition | null,
): HexPosition | null {
  const pattern = action.area_pattern?.pattern;
  if (!pattern) {
    return null;
  }
  if (pattern === "burst") {
    return actorPosition;
  }
  if (action.area_origin) {
    return action.area_origin;
  }
  if (pattern === "blast") {
    return targetPosition;
  }
  if (pattern === "line" || pattern === "cone") {
    return actorPosition;
  }
  return null;
}

function calculatePatternCoords(
  pattern: AttackPatternDefinition,
  origin: HexPosition,
  direction: HexCoord | null,
): HexCoord[] {
  if (pattern.pattern === "line") {
    if (!direction || !normalizeHexDirection(direction)) {
      return [];
    }
    return hexLineFromDirection(origin.coord, direction, pattern.size);
  }

  if (pattern.pattern === "cone") {
    if (!direction || !normalizeHexDirection(direction)) {
      return [];
    }
    if (pattern.cone_mode === "axis") {
      return hexConeCentered(origin.coord, direction, pattern.size);
    }
    return hexCone(origin.coord, direction, pattern.size);
  }

  if (pattern.pattern === "blast" || pattern.pattern === "burst") {
    return hexesInRadius(origin.coord, pattern.size);
  }

  return [];
}

function resolveGrid({
  origin,
  radius,
  tokens,
  markers,
  overlays,
  scenario,
}: {
  origin: HexCoord;
  radius?: number;
  tokens: RenderToken[];
  markers: RenderMarker[];
  overlays: AreaOverlay[];
  scenario: MechCombatScenario;
}): HexGrid {
  if (radius !== undefined) {
    return buildHexGrid(radius, origin);
  }

  let maxDistance = 0;
  for (const token of tokens) {
    maxDistance = Math.max(maxDistance, hexDistance(origin, token.coord));
  }
  for (const overlay of overlays) {
    for (const coord of overlay.coords) {
      maxDistance = Math.max(maxDistance, hexDistance(origin, coord));
    }
  }
  for (const marker of markers) {
    maxDistance = Math.max(maxDistance, hexDistance(origin, marker.coord));
  }
  for (const tile of scenario.terrain?.tiles ?? []) {
    maxDistance = Math.max(maxDistance, hexDistance(origin, tile.coord));
  }

  const resolvedRadius =
    maxDistance > 0
      ? Math.max(DEFAULT_GRID_RADIUS, maxDistance + 1)
      : DEFAULT_GRID_RADIUS;
  return buildHexGrid(resolvedRadius, origin);
}

function labelFromName(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) {
    return "?";
  }
  // Extract first word (before space or apostrophe) for better labels
  // "VANGUARD's Everest" → "VA", "GMS Grunt (Grunt 1)" → "G1"
  const firstWordMatch = trimmed.match(/^([A-Za-z]+)/);
  const firstWord = firstWordMatch?.[1] ?? trimmed;

  // Check for numbered suffix like "(Grunt 1)" → extract number
  const numberMatch = trimmed.match(/\(.*?(\d+)\)/);
  if (numberMatch) {
    return `${firstWord[0]?.toUpperCase() ?? "?"}${numberMatch[1]}`;
  }

  // Use first 2 characters of first word
  return firstWord.slice(0, 2).toUpperCase() || "?";
}

function dedupeCoords(coords: HexCoord[]): HexCoord[] {
  const seen = new Set<string>();
  const result: HexCoord[] = [];
  for (const coord of coords) {
    const key = `${coord.q},${coord.r}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(coord);
  }
  return result;
}

/**
 * Build movement range overlays showing reachable hexes with color coding:
 * - Green (easy): cost <= 50% of speed
 * - Yellow (atMax): cost > 50% but <= 100% of speed
 */
export function buildMovementRangeOverlays(
  origin: HexCoord,
  speed: number,
  validHexes: Set<string>,
  blockedHexes?: Set<string>,
  difficultHexes?: Set<string>,
): AreaOverlay[] {
  const reachable = calculateReachableHexes(
    origin,
    speed,
    validHexes,
    blockedHexes,
    difficultHexes,
  );

  const easyThreshold = Math.floor(speed / 2);

  const easy: HexCoord[] = [];
  const atMax: HexCoord[] = [];

  for (const { coord, cost } of reachable) {
    if (cost <= easyThreshold) {
      easy.push(coord);
    } else {
      atMax.push(coord);
    }
  }

  const overlays: AreaOverlay[] = [];

  if (easy.length > 0) {
    overlays.push({ coords: easy, style: MOVEMENT_OVERLAY_STYLES.easy });
  }
  if (atMax.length > 0) {
    overlays.push({ coords: atMax, style: MOVEMENT_OVERLAY_STYLES.atMax });
  }

  return overlays;
}
