/**
 * Terrain SVG Pattern Definitions for Lancer Combat Canvas
 * 
 * This module provides SVG-based patterns for different terrain types:
 * - Forest: Green pattern with tree/leaf motif
 * - Urban: Gray pattern with building/grid motif  
 * - Water: Blue pattern with wave motif
 * - Hazard: Red/orange pattern with warning motif
 * 
 * These patterns are rendered on the HTML5 Canvas using createPattern()
 * with SVG data URLs for crisp, scalable graphics.
 */

export type TerrainPatternType = "forest" | "urban" | "water" | "hazard" | "desert" | "facility";

export type TerrainPatternConfig = {
  /** SVG pattern definition as data URL */
  patternSvg: string;
  /** Base fill color to use if pattern fails to load */
  fallbackColor: string;
  /** Pattern scale factor */
  scale: number;
  /** Opacity for pattern overlay (0-1) */
  opacity: number;
};

// Forest pattern: Trees and leaf motif on green background
const FOREST_PATTERN_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60">
  <defs>
    <pattern id="forest" x="0" y="0" width="30" height="30" patternUnits="userSpaceOnUse">
      <!-- Background -->
      <rect width="30" height="30" fill="#166534" fill-opacity="0.15"/>
      <!-- Tree shape -->
      <path d="M15 2 L22 12 L18 12 L24 20 L19 20 L25 28 L5 28 L11 20 L6 20 L12 12 L8 12 Z" 
            fill="#15803d" fill-opacity="0.4"/>
      <!-- Small leaves -->
      <circle cx="8" cy="8" r="2" fill="#22c55e" fill-opacity="0.3"/>
      <circle cx="22" cy="6" r="1.5" fill="#22c55e" fill-opacity="0.25"/>
      <circle cx="25" cy="24" r="2" fill="#22c55e" fill-opacity="0.3"/>
    </pattern>
  </defs>
  <rect width="60" height="60" fill="url(#forest)"/>
</svg>
`;

// Urban pattern: Building/grid motif on gray background
const URBAN_PATTERN_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60">
  <defs>
    <pattern id="urban" x="0" y="0" width="30" height="30" patternUnits="userSpaceOnUse">
      <!-- Background -->
      <rect width="30" height="30" fill="#475569" fill-opacity="0.12"/>
      <!-- Building blocks -->
      <rect x="2" y="8" width="8" height="20" fill="#64748b" fill-opacity="0.3" stroke="#475569" stroke-width="0.5"/>
      <rect x="12" y="4" width="6" height="24" fill="#64748b" fill-opacity="0.25" stroke="#475569" stroke-width="0.5"/>
      <rect x="20" y="12" width="8" height="16" fill="#64748b" fill-opacity="0.35" stroke="#475569" stroke-width="0.5"/>
      <!-- Windows -->
      <rect x="4" y="10" width="2" height="2" fill="#94a3b8" fill-opacity="0.5"/>
      <rect x="4" y="14" width="2" height="2" fill="#94a3b8" fill-opacity="0.5"/>
      <rect x="4" y="18" width="2" height="2" fill="#94a3b8" fill-opacity="0.5"/>
      <rect x="14" y="6" width="2" height="2" fill="#94a3b8" fill-opacity="0.4"/>
      <rect x="14" y="12" width="2" height="2" fill="#94a3b8" fill-opacity="0.4"/>
      <rect x="22" y="14" width="2" height="2" fill="#94a3b8" fill-opacity="0.5"/>
    </pattern>
  </defs>
  <rect width="60" height="60" fill="url(#urban)"/>
</svg>
`;

// Water pattern: Wave motif on blue background
const WATER_PATTERN_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60">
  <defs>
    <pattern id="water" x="0" y="0" width="40" height="20" patternUnits="userSpaceOnUse">
      <!-- Background -->
      <rect width="40" height="20" fill="#0369a1" fill-opacity="0.15"/>
      <!-- Wave lines -->
      <path d="M0 8 Q10 4, 20 8 T40 8" stroke="#0ea5e9" stroke-width="1.5" stroke-opacity="0.4" fill="none"/>
      <path d="M0 14 Q10 10, 20 14 T40 14" stroke="#0284c7" stroke-width="1.5" stroke-opacity="0.35" fill="none"/>
      <path d="M0 18 Q10 14, 20 18 T40 18" stroke="#38bdf8" stroke-width="1" stroke-opacity="0.3" fill="none"/>
      <!-- Sparkles -->
      <circle cx="10" cy="6" r="1" fill="#bae6fd" fill-opacity="0.5"/>
      <circle cx="30" cy="10" r="0.8" fill="#bae6fd" fill-opacity="0.4"/>
    </pattern>
  </defs>
  <rect width="60" height="60" fill="url(#water)"/>
</svg>
`;

// Hazard pattern: Warning motif on red/orange background
const HAZARD_PATTERN_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60">
  <defs>
    <pattern id="hazard" x="0" y="0" width="30" height="30" patternUnits="userSpaceOnUse">
      <!-- Diagonal stripes -->
      <rect width="30" height="30" fill="#7f1d1d" fill-opacity="0.2"/>
      <path d="M0 30 L30 0" stroke="#dc2626" stroke-width="4" stroke-opacity="0.35"/>
      <path d="M-5 10 L10 -5" stroke="#dc2626" stroke-width="4" stroke-opacity="0.35"/>
      <path d="M20 35 L35 20" stroke="#dc2626" stroke-width="4" stroke-opacity="0.35"/>
      <!-- Warning triangles -->
      <path d="M8 20 L12 12 L16 20 Z" fill="#f97316" fill-opacity="0.5" stroke="#ea580c" stroke-width="0.5"/>
      <path d="M22 28 L26 20 L30 28 Z" fill="#f97316" fill-opacity="0.5" stroke="#ea580c" stroke-width="0.5"/>
    </pattern>
  </defs>
  <rect width="60" height="60" fill="url(#hazard)"/>
</svg>
`;

// Desert pattern: Sandy texture with sparse vegetation
const DESERT_PATTERN_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60">
  <defs>
    <pattern id="desert" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
      <!-- Sandy background -->
      <rect width="40" height="40" fill="#d97706" fill-opacity="0.08"/>
      <!-- Sand ripples -->
      <path d="M0 10 Q20 8, 40 10" stroke="#b45309" stroke-width="0.8" stroke-opacity="0.2" fill="none"/>
      <path d="M0 25 Q20 23, 40 25" stroke="#b45309" stroke-width="0.8" stroke-opacity="0.2" fill="none"/>
      <path d="M0 35 Q20 33, 40 35" stroke="#b45309" stroke-width="0.8" stroke-opacity="0.2" fill="none"/>
      <!-- Sparse rocks/vegetation -->
      <ellipse cx="10" cy="15" rx="2" ry="1.5" fill="#92400e" fill-opacity="0.25"/>
      <ellipse cx="30" cy="30" rx="2.5" ry="2" fill="#92400e" fill-opacity="0.2"/>
      <circle cx="35" cy="12" r="1" fill="#65a30d" fill-opacity="0.2"/>
    </pattern>
  </defs>
  <rect width="60" height="60" fill="url(#desert)"/>
</svg>
`;

// Facility pattern: Industrial/mechanical motif
const FACILITY_PATTERN_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60">
  <defs>
    <pattern id="facility" x="0" y="0" width="30" height="30" patternUnits="userSpaceOnUse">
      <!-- Industrial background -->
      <rect width="30" height="30" fill="#334155" fill-opacity="0.1"/>
      <!-- Metal panels -->
      <rect x="0" y="0" width="14" height="14" fill="#475569" fill-opacity="0.15" stroke="#334155" stroke-width="0.5"/>
      <rect x="15" y="0" width="15" height="14" fill="#475569" fill-opacity="0.12" stroke="#334155" stroke-width="0.5"/>
      <rect x="0" y="15" width="14" height="15" fill="#475569" fill-opacity="0.12" stroke="#334155" stroke-width="0.5"/>
      <rect x="15" y="15" width="15" height="15" fill="#475569" fill-opacity="0.15" stroke="#334155" stroke-width="0.5"/>
      <!-- Rivets -->
      <circle cx="2" cy="2" r="0.8" fill="#64748b" fill-opacity="0.5"/>
      <circle cx="12" cy="2" r="0.8" fill="#64748b" fill-opacity="0.5"/>
      <circle cx="2" cy="12" r="0.8" fill="#64748b" fill-opacity="0.5"/>
      <circle cx="17" cy="2" r="0.8" fill="#64748b" fill-opacity="0.5"/>
      <circle cx="27" cy="2" r="0.8" fill="#64748b" fill-opacity="0.5"/>
      <circle cx="2" cy="17" r="0.8" fill="#64748b" fill-opacity="0.5"/>
      <!-- Warning stripes on some panels -->
      <line x1="15" y1="18" x2="30" y2="25" stroke="#f59e0b" stroke-width="1" stroke-opacity="0.25"/>
      <line x1="15" y1="22" x2="30" y2="29" stroke="#f59e0b" stroke-width="1" stroke-opacity="0.25"/>
    </pattern>
  </defs>
  <rect width="60" height="60" fill="url(#facility)"/>
</svg>
`;

/** Map terrain types to their pattern configurations */
export const TERRAIN_PATTERNS: Record<TerrainPatternType, TerrainPatternConfig> = {
  forest: {
    patternSvg: FOREST_PATTERN_SVG,
    fallbackColor: "rgba(22, 101, 52, 0.15)",
    scale: 1,
    opacity: 0.7,
  },
  urban: {
    patternSvg: URBAN_PATTERN_SVG,
    fallbackColor: "rgba(71, 85, 105, 0.12)",
    scale: 1,
    opacity: 0.7,
  },
  water: {
    patternSvg: WATER_PATTERN_SVG,
    fallbackColor: "rgba(3, 105, 161, 0.15)",
    scale: 1,
    opacity: 0.7,
  },
  hazard: {
    patternSvg: HAZARD_PATTERN_SVG,
    fallbackColor: "rgba(127, 29, 29, 0.2)",
    scale: 1,
    opacity: 0.8,
  },
  desert: {
    patternSvg: DESERT_PATTERN_SVG,
    fallbackColor: "rgba(217, 119, 6, 0.08)",
    scale: 1,
    opacity: 0.6,
  },
  facility: {
    patternSvg: FACILITY_PATTERN_SVG,
    fallbackColor: "rgba(51, 65, 85, 0.1)",
    scale: 1,
    opacity: 0.7,
  },
};

/** Map mission terrain strings to pattern types */
export function mapMissionTerrainToPattern(terrain: string): TerrainPatternType | undefined {
  const normalized = terrain.toLowerCase().trim();
  
  switch (normalized) {
    case "forest":
    case "wilderness":
      return "forest";
    case "urban":
      return "urban";
    case "water":
    case "aquatic":
      return "water";
    case "desert":
      return "desert";
    case "facility":
    case "industrial":
    case "space station":
      return "facility";
    case "hazard":
    case "dangerous":
      return "hazard";
    default:
      return undefined;
  }
}

/** Create a canvas pattern from SVG string (async - use preloadTerrainPatterns instead) */
export function createPatternFromSVG(
  _ctx: CanvasRenderingContext2D,
  _svgString: string,
  _scale: number = 1,
): CanvasPattern | null {
  // This function is kept for API compatibility but patterns should be preloaded
  // using preloadTerrainPatterns() for better performance
  return null;
}

/** Preload terrain patterns and return a map of pattern types to CanvasPatterns */
export async function preloadTerrainPatterns(
  ctx: CanvasRenderingContext2D,
): Promise<Partial<Record<TerrainPatternType, CanvasPattern>>> {
  const patterns: Partial<Record<TerrainPatternType, CanvasPattern>> = {};
  
  for (const [type, config] of Object.entries(TERRAIN_PATTERNS)) {
    try {
      const patternType = type as TerrainPatternType;
      const svgDataUrl = `data:image/svg+xml;base64,${btoa(config.patternSvg)}`;
      
      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = reject;
        img.src = svgDataUrl;
      });
      
      const pattern = ctx.createPattern(img, "repeat");
      if (pattern) {
        patterns[patternType] = pattern;
      }
    } catch (e) {
      console.warn(`Failed to load terrain pattern ${type}:`, e);
    }
  }
  
  return patterns;
}

/** Get fallback fill style for a terrain type */
export function getTerrainFallbackFill(type: TerrainPatternType): string {
  return TERRAIN_PATTERNS[type]?.fallbackColor ?? "rgba(100, 100, 100, 0.1)";
}
