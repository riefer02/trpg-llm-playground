import { useEffect, useState } from "react";
import { Heart, Swords, Target, Zap } from "lucide-react";

import type { CombatantState } from "../../lib/types/lancer";
import type { ActionPreviewResponse, AvailableActionItem } from "../../lib/api/combat";

/**
 * Preview panel showing predicted action outcome (damage, hit chance, HP after attack).
 * Appears when hovering over an attack action and a target combatant.
 */

export interface ActionPreviewPanelProps {
  /** The target combatant */
  target: CombatantState | null;
  /** The action being previewed */
  previewAction: AvailableActionItem | null;
  /** Preview response data from API */
  previewResponse: ActionPreviewResponse | null;
  /** Screen position for the panel (near target token) */
  position: { x: number; y: number } | null;
  /** Loading state */
  isLoading?: boolean;
  /** Error state */
  error?: Error | null;
  /** Delay before showing panel (ms) */
  delay?: number;
}

export function ActionPreviewPanel({
  target,
  previewAction,
  previewResponse,
  position,
  isLoading = false,
  error = null,
  delay = 200,
}: ActionPreviewPanelProps) {
  const [visible, setVisible] = useState(false);

  // Handle show/hide with delay
  useEffect(() => {
    if (!target || !previewAction || !position) {
      setVisible(false);
      return;
    }

    // Update visibility with delay
    const timer = setTimeout(() => setVisible(true), delay);
    return () => clearTimeout(timer);
  }, [target, previewAction, position, delay]);

  // Don't render if not visible or no target/action
  if (!visible || !target || !previewAction || !position) {
    return null;
  }

  // Adjust position to keep panel on screen
  const adjustedPosition = { ...position };
  if (typeof window !== "undefined") {
    const panelWidth = 220;
    const panelHeight = 180;
    if (position.x + panelWidth + 20 > window.innerWidth) {
      adjustedPosition.x = position.x - panelWidth - 10;
    } else {
      adjustedPosition.x = position.x + 15;
    }
    if (position.y + panelHeight > window.innerHeight) {
      adjustedPosition.y = window.innerHeight - panelHeight - 10;
    } else {
      adjustedPosition.y = position.y + 10;
    }
  }

  // Get target's current HP
  const hpCurrent = target.resources?.hp_current ?? 0;
  const hpMax = target.stats?.hp_max ?? 1;

  // Compute predicted HP after attack
  const damageAverage = previewResponse?.damage_average ?? 0;
  const damageMin = previewResponse?.damage_min ?? 0;
  const damageMax = previewResponse?.damage_max ?? 0;
  const hitProbability = previewResponse?.hit_probability ?? 0;

  const predictedHp = Math.max(0, hpCurrent - damageAverage);
  const predictedHpPercent = hpMax > 0 ? Math.round((predictedHp / hpMax) * 100) : 0;

  // Determine if attack is likely to hit (color coding)
  const hitChancePercent = Math.round(hitProbability * 100);
  let hitChanceColor = "text-muted-foreground";
  let hitChanceBarColor = "bg-muted-foreground";
  if (hitChancePercent >= 70) {
    hitChanceColor = "text-green-500";
    hitChanceBarColor = "bg-green-500";
  } else if (hitChancePercent >= 40) {
    hitChanceColor = "text-amber-500";
    hitChanceBarColor = "bg-amber-500";
  } else {
    hitChanceColor = "text-red-500";
    hitChanceBarColor = "bg-red-500";
  }

  return (
    <div
      className="fixed z-50 bg-popover/95 backdrop-blur-sm border border-border rounded-lg shadow-xl p-3 min-w-[200px] max-w-[240px] pointer-events-none animate-in fade-in-50 duration-100"
      style={{ left: adjustedPosition.x, top: adjustedPosition.y }}
    >
      {/* Header: Action name */}
      <div className="mb-2 pb-2 border-b border-border">
        <div className="font-medium text-sm text-foreground flex items-center gap-2">
          <Swords className="w-4 h-4 text-primary" />
          {previewAction.action_name}
        </div>
        <div className="text-xs text-muted-foreground">
          vs {target.name}
        </div>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="text-xs text-muted-foreground py-2 text-center">
          Calculating preview...
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="text-xs text-destructive py-2">
          Preview failed: {error.message}
        </div>
      )}

      {/* Preview content */}
      {previewResponse && !isLoading && !error && (
        <div className="space-y-3">
          {/* Hit Chance */}
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Target className="w-3 h-3" />
                Hit Chance
              </div>
              <span className={`font-mono font-medium ${hitChanceColor}`}>
                {hitChancePercent}%
              </span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                 className={`h-full transition-all ${hitChanceBarColor}`}
                style={{ width: `${hitChancePercent}%` }}
              />
            </div>
          </div>

          {/* Damage Range */}
          {damageAverage > 0 && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  <Zap className="w-3 h-3" />
                  Damage
                </div>
                <span className="font-mono font-medium text-foreground">
                  {damageMin}-{damageMax} avg {damageAverage.toFixed(1)}
                </span>
              </div>
              <div className="text-[10px] text-muted-foreground">
                {previewResponse.damage_types?.length > 0 && (
                  <span>Damage types: {previewResponse.damage_types.join(', ')}</span>
                )}
              </div>
            </div>
          )}

          {/* HP Before/After */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Heart className="w-3 h-3 text-red-400" />
                HP After Attack
              </div>
              <div className="font-mono text-foreground">
                {predictedHp.toFixed(0)}/{hpMax}
              </div>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-red-500 transition-all"
                style={{ width: `${predictedHpPercent}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-[10px] text-muted-foreground">
              <span>Current: {hpCurrent}/{hpMax}</span>
              <span>Predicted: {predictedHp.toFixed(0)}/{hpMax}</span>
            </div>
          </div>

          {/* Predicted Effects */}
          {previewResponse.predicted_effects && previewResponse.predicted_effects.length > 0 && (
            <div className="pt-2 border-t border-border">
              <div className="text-xs text-muted-foreground">
                Predicted effects: {previewResponse.predicted_effects.length}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}