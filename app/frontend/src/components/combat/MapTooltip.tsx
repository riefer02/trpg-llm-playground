import { useEffect, useState } from "react";
import {
  Heart,
  Flame,
  Shield,
  AlertTriangle,
  Mountain,
  TreePine,
  Building,
  Droplets,
  type LucideIcon,
} from "lucide-react";

import type { CombatantState, DeployableState } from "../../lib/types/lancer";

/**
 * Tooltip that appears when hovering over map elements.
 * Shows contextual information about the hovered element.
 */

// Terrain type icons
const TERRAIN_ICONS: Record<string, LucideIcon> = {
  open: Mountain,
  forest: TreePine,
  urban: Building,
  water: Droplets,
  hazard: AlertTriangle,
};

export type HoverTarget =
  | { type: "empty"; coord: { q: number; r: number }; terrain?: TerrainInfo }
  | {
      type: "combatant";
      combatant: CombatantState;
      isEnemy: boolean;
      coord: { q: number; r: number };
    }
  | {
      type: "deployable";
      deployable: DeployableState;
      deployableId: string;
      coord: { q: number; r: number };
    };

export interface TerrainInfo {
  terrainType?: string;
  elevation?: number;
  cover?: "none" | "soft" | "hard";
  isDangerous?: boolean;
  isDifficult?: boolean;
}

export interface MapTooltipProps {
  /** The target being hovered */
  target: HoverTarget | null;
  /** Screen position for the tooltip */
  position: { x: number; y: number } | null;
  /** Delay before showing tooltip (ms) */
  delay?: number;
}

export function MapTooltip({ target, position, delay = 200 }: MapTooltipProps) {
  const [visible, setVisible] = useState(false);
  const [currentTarget, setCurrentTarget] = useState<HoverTarget | null>(null);

  // Handle show/hide with delay
  useEffect(() => {
    if (!target || !position) {
      setVisible(false);
      return;
    }

    // Update target immediately but delay visibility
    setCurrentTarget(target);
    const timer = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(timer);
  }, [target, position, delay]);

  // Don't render if not visible or no target
  if (!visible || !currentTarget || !position) {
    return null;
  }

  // Adjust position to keep tooltip on screen
  const adjustedPosition = { ...position };
  if (typeof window !== "undefined") {
    const tooltipWidth = 200;
    const tooltipHeight = 150;
    if (position.x + tooltipWidth + 20 > window.innerWidth) {
      adjustedPosition.x = position.x - tooltipWidth - 10;
    } else {
      adjustedPosition.x = position.x + 15;
    }
    if (position.y + tooltipHeight > window.innerHeight) {
      adjustedPosition.y = window.innerHeight - tooltipHeight - 10;
    } else {
      adjustedPosition.y = position.y + 10;
    }
  }

  return (
    <div
      className="fixed z-40 bg-popover/95 backdrop-blur-sm border border-border rounded-lg shadow-lg p-3 min-w-[160px] max-w-[220px] pointer-events-none animate-in fade-in-50 duration-100"
      style={{ left: adjustedPosition.x, top: adjustedPosition.y }}
    >
      {currentTarget.type === "combatant" && (
        <CombatantTooltip
          combatant={currentTarget.combatant}
          isEnemy={currentTarget.isEnemy}
        />
      )}
      {currentTarget.type === "deployable" && (
        <DeployableTooltip deployable={currentTarget.deployable} />
      )}
      {currentTarget.type === "empty" && (
        <HexTooltip coord={currentTarget.coord} terrain={currentTarget.terrain} />
      )}
    </div>
  );
}

interface CombatantTooltipProps {
  combatant: CombatantState;
  isEnemy: boolean;
}

function CombatantTooltip({ combatant, isEnemy }: CombatantTooltipProps) {
  // Get HP from stats (max) and resources (current)
  const hpMax = combatant.stats?.hp_max ?? 1;
  const hpCurrent = combatant.resources?.hp_current ?? 0;
  const heatCurrent = combatant.resources?.heat_current ?? 0;
  const heatCap = combatant.resources?.heat_cap ?? 6;

  // Get conditions from status_instances
  const statusInstances = combatant.status_instances ?? [];
  const conditionNames = statusInstances.map((s) => s.status ?? "unknown");

  // For enemies, show limited info (no exact numbers)
  const hpPercent = hpMax > 0 ? Math.round((hpCurrent / hpMax) * 100) : 0;
  const heatPercent = heatCap > 0 ? Math.round((heatCurrent / heatCap) * 100) : 0;

  return (
    <div className="space-y-2">
      {/* Name */}
      <div>
        <div className="font-medium text-sm text-foreground truncate">
          {combatant.name}
        </div>
        <div className="text-xs text-muted-foreground capitalize">
          {combatant.kind}
        </div>
      </div>

      {/* HP Bar */}
      <div className="space-y-1">
        <div className="flex items-center gap-1.5 text-xs">
          <Heart className="w-3 h-3 text-red-400" />
          <span className="text-muted-foreground">HP</span>
          {!isEnemy && (
            <span className="ml-auto font-mono">
              {hpCurrent}/{hpMax}
            </span>
          )}
        </div>
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${getHpBarColor(hpPercent)}`}
            style={{ width: `${hpPercent}%` }}
          />
        </div>
      </div>

      {/* Heat Bar */}
      <div className="space-y-1">
        <div className="flex items-center gap-1.5 text-xs">
          <Flame className="w-3 h-3 text-orange-400" />
          <span className="text-muted-foreground">Heat</span>
          {!isEnemy && (
            <span className="ml-auto font-mono">
              {heatCurrent}/{heatCap}
            </span>
          )}
        </div>
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${getHeatBarColor(heatPercent)}`}
            style={{ width: `${heatPercent}%` }}
          />
        </div>
      </div>

      {/* Conditions/Statuses */}
      {conditionNames.length > 0 && (
        <div className="flex flex-wrap gap-1 pt-1 border-t border-border">
          {conditionNames.slice(0, 4).map((condition, index) => (
            <span
              key={`${condition}-${index}`}
              className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 truncate max-w-[80px]"
            >
              {formatCondition(condition)}
            </span>
          ))}
          {conditionNames.length > 4 && (
            <span className="text-[10px] text-muted-foreground">
              +{conditionNames.length - 4}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

interface DeployableTooltipProps {
  deployable: DeployableState;
}

function DeployableTooltip({ deployable }: DeployableTooltipProps) {
  const hp = deployable.hp ?? 0;
  const maxHp = deployable.max_hp ?? 1;
  const hpPercent = maxHp > 0 ? Math.round((hp / maxHp) * 100) : 0;

  return (
    <div className="space-y-2">
      <div>
        <div className="font-medium text-sm text-foreground truncate">
          {deployable.name ?? "Deployable"}
        </div>
        <div className="text-xs text-muted-foreground capitalize">
          {deployable.kind ?? "deployable"}
        </div>
      </div>

      {deployable.owner_id && (
        <div className="text-xs text-muted-foreground">
          Owner: {deployable.owner_id}
        </div>
      )}

      {maxHp > 0 && (
        <div className="space-y-1">
          <div className="flex items-center gap-1.5 text-xs">
            <Shield className="w-3 h-3 text-blue-400" />
            <span className="text-muted-foreground">HP</span>
            <span className="ml-auto font-mono">
              {hp}/{maxHp}
            </span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${hpPercent}%` }}
            />
          </div>
        </div>
      )}

      {deployable.is_destroyed && (
        <div className="text-xs text-destructive font-medium">Destroyed</div>
      )}
    </div>
  );
}

interface HexTooltipProps {
  coord: { q: number; r: number };
  terrain?: TerrainInfo;
}

function HexTooltip({ coord, terrain }: HexTooltipProps) {
  const TerrainIcon = terrain?.terrainType
    ? TERRAIN_ICONS[terrain.terrainType] ?? Mountain
    : Mountain;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <TerrainIcon className="w-4 h-4 text-muted-foreground" />
        <div>
          <div className="text-sm font-medium text-foreground capitalize">
            {terrain?.terrainType ?? "Open Ground"}
          </div>
          <div className="text-xs text-muted-foreground">
            ({coord.q}, {coord.r})
          </div>
        </div>
      </div>

      {terrain?.elevation !== undefined && terrain.elevation !== 0 && (
        <div className="text-xs text-muted-foreground">
          Elevation: {terrain.elevation > 0 ? "+" : ""}
          {terrain.elevation}
        </div>
      )}

      {terrain?.cover && terrain.cover !== "none" && (
        <div className="flex items-center gap-1.5 text-xs">
          <Shield className="w-3 h-3 text-blue-400" />
          <span className="capitalize">{terrain.cover} Cover</span>
        </div>
      )}

      {(terrain?.isDangerous || terrain?.isDifficult) && (
        <div className="flex flex-wrap gap-1 pt-1 border-t border-border">
          {terrain.isDangerous && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" />
              Dangerous
            </span>
          )}
          {terrain.isDifficult && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">
              Difficult
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function getHpBarColor(percent: number): string {
  if (percent > 50) return "bg-green-500";
  if (percent > 25) return "bg-amber-500";
  return "bg-red-500";
}

function getHeatBarColor(percent: number): string {
  if (percent < 50) return "bg-blue-500";
  if (percent < 75) return "bg-orange-500";
  return "bg-red-500";
}

function formatCondition(condition: string): string {
  return condition
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
