import { useEffect, useCallback } from "react";
import {
  Footprints,
  Zap,
  Swords,
  Crosshair,
  Heart,
  Shield,
  Eye,
  Cpu,
  Radio,
  Scan,
  Lock,
  RotateCcw,
  Skull,
  Target,
  Flame,
  ArrowRight,
  Grip,
  Hand,
  type LucideIcon,
} from "lucide-react";

import { Button } from "../ui";
import type { AvailableActionItem, ActionPreviewResponse } from "../../lib/api/combat";
import type { HexCoord } from "../../lib/types/lancer";

// Icon mapping for actions
const ACTION_ICONS: Record<string, LucideIcon> = {
  move: Footprints,
  boost: Zap,
  skirmish: Swords,
  barrage: Crosshair,
  fight: Swords,
  ram: ArrowRight,
  grapple: Grip,
  quick_tech: Cpu,
  full_tech: Radio,
  scan: Scan,
  lock_on: Lock,
  invade: Skull,
  bolster: Shield,
  stabilize: Heart,
  activate_system: Cpu,
  reload: RotateCcw,
  overwatch: Eye,
  brace: Shield,
  overcharge: Flame,
  dismount: Hand,
  eject: Hand,
  self_destruct: Skull,
  hide: Eye,
  search: Scan,
  prepare: Target,
  disengage: Footprints,
  improvised_attack: Swords,
};

const DEFAULT_ICON = Target;

// Action type colors
const ACTION_TYPE_COLORS = {
  full: "text-blue-400 border-blue-500 bg-blue-500/10",
  quick: "text-green-400 border-green-500 bg-green-500/10",
  free: "text-gray-400 border-gray-500 bg-gray-500/10",
  protocol: "text-purple-400 border-purple-500 bg-purple-500/10",
  reaction: "text-amber-400 border-amber-500 bg-amber-500/10",
};

export interface ActionConfirmationData {
  action: AvailableActionItem;
  targetNames?: string[];
  targetIds?: string[];
  weaponName?: string;
  weaponId?: string;
  systemName?: string;
  systemId?: string;
  movementPath?: HexCoord[];
  pathDistance?: number;
  preview?: ActionPreviewResponse | null;
  isPreviewLoading?: boolean;
}

export interface ActionConfirmationOverlayProps {
  isOpen: boolean;
  data: ActionConfirmationData | null;
  onConfirm: () => void;
  onCancel: () => void;
  isExecuting?: boolean;
}

export function ActionConfirmationOverlay({
  isOpen,
  data,
  onConfirm,
  onCancel,
  isExecuting = false,
}: ActionConfirmationOverlayProps) {
  // Keyboard shortcuts: Enter = Confirm, Escape = Cancel
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isOpen || isExecuting) return;

      if (e.key === "Enter") {
        e.preventDefault();
        onConfirm();
      } else if (e.key === "Escape") {
        e.preventDefault();
        onCancel();
      }
    },
    [isOpen, isExecuting, onConfirm, onCancel]
  );

  useEffect(() => {
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }
  }, [isOpen, handleKeyDown]);

  if (!isOpen || !data) return null;

  const { action, targetNames, weaponName, systemName, movementPath, pathDistance, preview, isPreviewLoading } = data;
  const Icon = ACTION_ICONS[action.action_id] ?? DEFAULT_ICON;
  const typeColors = ACTION_TYPE_COLORS[action.action_type as keyof typeof ACTION_TYPE_COLORS] ?? ACTION_TYPE_COLORS.free;

  // Build action summary
  const getSummary = () => {
    const parts: string[] = [];

    // Movement actions
    if (action.action_id === "move" || action.action_id === "boost") {
      if (movementPath && movementPath.length > 1) {
        const dest = movementPath[movementPath.length - 1];
        parts.push(`Move to (${dest.q}, ${dest.r})`);
        if (pathDistance !== undefined) {
          parts.push(`${pathDistance} hexes`);
        }
      }
    }

    // Target actions
    if (targetNames && targetNames.length > 0) {
      parts.push(`Target: ${targetNames.join(", ")}`);
    }

    // Weapon
    if (weaponName) {
      parts.push(`Using: ${weaponName}`);
    }

    // System
    if (systemName) {
      parts.push(`System: ${systemName}`);
    }

    return parts;
  };

  const summaryParts = getSummary();

  return (
    <div className="absolute inset-0 z-40 flex items-center justify-center pointer-events-auto">
      {/* Semi-transparent backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onCancel}
        role="presentation"
      />

      {/* Confirmation card */}
      <div className={`relative z-10 w-full max-w-md mx-4 rounded-xl border-2 ${typeColors} bg-background/95 backdrop-blur-md shadow-2xl overflow-hidden`}>
        {/* Header */}
        <div className={`px-6 py-4 border-b border-border/50 ${typeColors.replace('text-', 'bg-').replace('/10', '/5')}`}>
          <div className="flex items-center gap-4">
            <div className={`w-14 h-14 rounded-xl flex items-center justify-center ${typeColors.replace('text-', 'bg-')}`}>
              <Icon className="w-8 h-8" />
            </div>
            <div className="flex-1">
              <h2 className="text-xl font-bold text-foreground">
                {action.action_name}
              </h2>
              <p className="text-sm text-muted-foreground capitalize">
                {action.action_type} Action
              </p>
            </div>
          </div>
        </div>

        {/* Details */}
        <div className="px-6 py-4 space-y-3">
          {/* Summary */}
          {summaryParts.length > 0 && (
            <div className="space-y-1">
              {summaryParts.map((part, i) => (
                <div key={i} className="text-sm text-foreground/90">
                  {part}
                </div>
              ))}
            </div>
          )}

          {/* Preview data */}
          {isPreviewLoading && (
            <div className="text-sm text-muted-foreground animate-pulse">
              Calculating outcome...
            </div>
          )}

          {preview && !isPreviewLoading && (
            <div className="bg-muted/50 rounded-lg p-3 space-y-2">
              {/* Hit probability */}
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Hit Chance</span>
                <span className={`font-mono font-bold ${
                  preview.hit_probability >= 0.75 ? "text-green-400" :
                  preview.hit_probability >= 0.5 ? "text-amber-400" :
                  "text-red-400"
                }`}>
                  {(preview.hit_probability * 100).toFixed(0)}%
                </span>
              </div>

              {/* Damage */}
              {preview.damage_average > 0 && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Expected Damage</span>
                  <span className="font-mono font-bold text-foreground">
                    {preview.damage_min}-{preview.damage_max}
                    <span className="text-muted-foreground ml-1">
                      (avg {preview.damage_average.toFixed(0)})
                    </span>
                  </span>
                </div>
              )}

              {/* Damage types */}
              {preview.damage_types.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  Types: {preview.damage_types.join(", ")}
                </div>
              )}

              {/* Predicted effects */}
              {preview.predicted_effects.length > 0 && (
                <div className="text-xs">
                  <span className="text-muted-foreground">Effects: </span>
                  <span className="text-amber-400">
                    {preview.predicted_effects.map(e => e.type).join(", ")}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* No preview available message */}
          {!preview && !isPreviewLoading && (action.requires_target || action.action_id === "skirmish" || action.action_id === "barrage") && (
            <div className="text-xs text-muted-foreground italic">
              Outcome preview not available
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="px-6 py-4 border-t border-border/50 bg-muted/30">
          <div className="flex gap-3">
            <Button
              variant="primary"
              size="lg"
              className="flex-1 h-12 text-base font-semibold"
              onClick={onConfirm}
              disabled={isExecuting}
            >
              {isExecuting ? (
                <span className="flex items-center gap-2">
                  <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  Executing...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  Confirm
                  <kbd className="ml-2 px-1.5 py-0.5 text-xs bg-primary-foreground/20 rounded">
                    Enter
                  </kbd>
                </span>
              )}
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="h-12 px-6"
              onClick={onCancel}
              disabled={isExecuting}
            >
              <span className="flex items-center gap-2">
                Cancel
                <kbd className="ml-1 px-1.5 py-0.5 text-xs bg-muted rounded">
                  Esc
                </kbd>
              </span>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
