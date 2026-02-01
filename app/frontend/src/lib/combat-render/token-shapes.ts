/**
 * Token shapes for frame-based visual identification (E9-US-003)
 * 
 * Each manufacturer has distinct visual characteristics:
 * - GMS: Standard circle (baseline)
 * - IPS-N: Angular/aggressive shapes (triangles, diamonds)
 * - SSC: Sleek/pointed shapes (hexagons with points)
 * - HORUS: Unusual/asymmetric shapes (distorted polygons)
 * - HA: Heavy/blocky shapes (squares, rounded rectangles)
 * - NPCs: Simple geometric shapes (triangle for grunts, diamond for specialists)
 */

export type TokenShape = 
  | "circle"      // GMS default, baseline
  | "triangle"    // IPS-N aggressive, NPC grunts
  | "diamond"     // IPS-N angular, NPC specialists
  | "hex_point"   // SSC sleek with pointed edges
  | "asymmetric"  // HORUS unusual/distorted
  | "square"      // HA heavy/blocky
  | "rounded_rect"; // HA blocky with rounded corners

/**
 * Map frame_id to token shape based on manufacturer
 */
export function getTokenShapeForFrame(frameId: string | null | undefined): TokenShape {
  if (!frameId) {
    return "circle";
  }

  const lowerFrameId = frameId.toLowerCase();

  // GMS frames - standard circle (baseline)
  if (lowerFrameId.includes("gms") || lowerFrameId.includes("everest")) {
    return "circle";
  }

  // IPS-N frames - angular/aggressive (triangle, diamond)
  if (lowerFrameId.includes("ipsn") || lowerFrameId.includes("ips-n")) {
    // Raleigh and Blackbeard use triangle
    if (lowerFrameId.includes("raleigh") || lowerFrameId.includes("blackbeard")) {
      return "triangle";
    }
    // Drake and others use diamond
    return "diamond";
  }

  // SSC frames - sleek/pointed (hex_point)
  if (lowerFrameId.includes("ssc")) {
    return "hex_point";
  }

  // HORUS frames - unusual/asymmetric
  if (lowerFrameId.includes("horus")) {
    return "asymmetric";
  }

  // HA frames - heavy/blocky (square, rounded_rect)
  if (lowerFrameId.includes("ha_")) {
    // Sherman uses square
    if (lowerFrameId.includes("sherman")) {
      return "square";
    }
    // Barbarossa and others use rounded rectangle
    return "rounded_rect";
  }

  // Default fallback
  return "circle";
}

/**
 * Map NPC kind/role to token shape
 */
export function getTokenShapeForNPC(kind: string, npcRole?: string | null): TokenShape {
  // Grunts get simple triangles
  if (kind === "npc" && !npcRole) {
    return "triangle";
  }

  // Role-based shapes for NPCs
  switch (npcRole) {
    case "striker":
      return "triangle";  // Aggressive
    case "defender":
      return "square";    // Solid/heavy
    case "controller":
      return "diamond";   // Technical
    case "supporter":
      return "circle";    // Neutral/supportive
    default:
      return "triangle";  // Default NPC shape
  }
}

/**
 * Draw a token shape on the canvas
 * 
 * @param ctx - Canvas rendering context
 * @param center - Center point in pixels
 * @param radius - Base radius of the token
 * @param shape - Shape type to draw
 * @param color - Fill color
 * @param strokeStyle - Stroke color
 * @param lineWidth - Line width
 * @returns The bounding box for label placement
 */
export function drawTokenShape(
  ctx: CanvasRenderingContext2D,
  center: { x: number; y: number },
  radius: number,
  shape: TokenShape,
  color: string,
  strokeStyle: string,
  lineWidth: number
): { x: number; y: number; width: number; height: number } {
  ctx.fillStyle = color;
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;

  // Start path
  ctx.beginPath();

  switch (shape) {
    case "circle":
      drawCircle(ctx, center, radius);
      break;
    case "triangle":
      drawTriangle(ctx, center, radius);
      break;
    case "diamond":
      drawDiamond(ctx, center, radius);
      break;
    case "hex_point":
      drawHexPoint(ctx, center, radius);
      break;
    case "asymmetric":
      drawAsymmetric(ctx, center, radius);
      break;
    case "square":
      drawSquare(ctx, center, radius);
      break;
    case "rounded_rect":
      drawRoundedRect(ctx, center, radius);
      break;
  }

  // Close and render
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Return bounding box for label positioning
  return {
    x: center.x - radius,
    y: center.y - radius,
    width: radius * 2,
    height: radius * 2,
  };
}

function drawCircle(
  ctx: CanvasRenderingContext2D,
  center: { x: number; y: number },
  radius: number
): void {
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
}

function drawTriangle(
  ctx: CanvasRenderingContext2D,
  center: { x: number; y: number },
  radius: number
): void {
  // Pointing up
  const top = { x: center.x, y: center.y - radius };
  const bottomLeft = { 
    x: center.x - radius * 0.866, 
    y: center.y + radius * 0.5 
  };
  const bottomRight = { 
    x: center.x + radius * 0.866, 
    y: center.y + radius * 0.5 
  };

  ctx.moveTo(top.x, top.y);
  ctx.lineTo(bottomRight.x, bottomRight.y);
  ctx.lineTo(bottomLeft.x, bottomLeft.y);
}

function drawDiamond(
  ctx: CanvasRenderingContext2D,
  center: { x: number; y: number },
  radius: number
): void {
  // Rotated square (45 degrees)
  ctx.moveTo(center.x, center.y - radius);
  ctx.lineTo(center.x + radius * 0.9, center.y);
  ctx.lineTo(center.x, center.y + radius);
  ctx.lineTo(center.x - radius * 0.9, center.y);
}

function drawHexPoint(
  ctx: CanvasRenderingContext2D,
  center: { x: number; y: number },
  radius: number
): void {
  // Hexagon with slightly elongated top/bottom points for sleek look
  const numPoints = 6;
  for (let i = 0; i < numPoints; i++) {
    // Skip top and bottom points, make them longer
    const isVertical = i === 0 || i === 3;
    const r = isVertical ? radius * 1.1 : radius * 0.85;
    const angle = (Math.PI / 3) * i - Math.PI / 2; // Start at top
    const x = center.x + r * Math.cos(angle);
    const y = center.y + r * Math.sin(angle);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
}

function drawAsymmetric(
  ctx: CanvasRenderingContext2D,
  center: { x: number; y: number },
  radius: number
): void {
  // Distorted asymmetric shape for HORUS
  // 5-sided irregular polygon
  const points = [
    { x: center.x, y: center.y - radius }, // top
    { x: center.x + radius * 0.9, y: center.y - radius * 0.2 },
    { x: center.x + radius * 0.5, y: center.y + radius * 0.8 },
    { x: center.x - radius * 0.7, y: center.y + radius * 0.6 },
    { x: center.x - radius * 0.8, y: center.y - radius * 0.4 },
  ];

  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
}

function drawSquare(
  ctx: CanvasRenderingContext2D,
  center: { x: number; y: number },
  radius: number
): void {
  // Axis-aligned square (heavy/blocky)
  const halfSize = radius * 0.85;
  ctx.moveTo(center.x - halfSize, center.y - halfSize);
  ctx.lineTo(center.x + halfSize, center.y - halfSize);
  ctx.lineTo(center.x + halfSize, center.y + halfSize);
  ctx.lineTo(center.x - halfSize, center.y + halfSize);
}

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  center: { x: number; y: number },
  radius: number
): void {
  // Rounded rectangle (heavy but refined)
  const width = radius * 1.6;
  const height = radius * 1.4;
  const cornerRadius = radius * 0.3;
  const x = center.x - width / 2;
  const y = center.y - height / 2;

  ctx.moveTo(x + cornerRadius, y);
  ctx.lineTo(x + width - cornerRadius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + cornerRadius);
  ctx.lineTo(x + width, y + height - cornerRadius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - cornerRadius, y + height);
  ctx.lineTo(x + cornerRadius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - cornerRadius);
  ctx.lineTo(x, y + cornerRadius);
  ctx.quadraticCurveTo(x, y, x + cornerRadius, y);
}

/**
 * Get a display label from a combatant name
 * Extracts initials or first 2 characters
 */
export function getTokenLabel(name: string): string {
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

/**
 * Get manufacturer display name from frame_id
 */
export function getManufacturerFromFrame(frameId: string | null | undefined): string {
  if (!frameId) return "Unknown";

  const lowerFrameId = frameId.toLowerCase();

  if (lowerFrameId.includes("gms")) return "GMS";
  if (lowerFrameId.includes("ipsn") || lowerFrameId.includes("ips-n")) return "IPS-N";
  if (lowerFrameId.includes("ssc")) return "SSC";
  if (lowerFrameId.includes("horus")) return "HORUS";
  if (lowerFrameId.includes("ha_")) return "HA";

  return "Unknown";
}
