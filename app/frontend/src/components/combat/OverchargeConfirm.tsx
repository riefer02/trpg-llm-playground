import { Button } from "../ui";

export interface OverchargeConfirmProps {
  currentLevel: number;  // 0-3
  heatCurrent: number;
  heatCap: number;
  onConfirm: () => void;
  onCancel: () => void;
  isOpen: boolean;
}

// Overcharge costs from Lancer core rules
const OVERCHARGE_COSTS = ["1 heat", "1d3 heat", "1d6 heat", "1d6+4 heat"] as const;

// Average heat generated per level for warning calculations
const OVERCHARGE_HEAT_AVG = [1, 2, 3.5, 7.5] as const;

export function OverchargeConfirm({
  currentLevel,
  heatCurrent,
  heatCap,
  onConfirm,
  onCancel,
  isOpen,
}: OverchargeConfirmProps) {
  if (!isOpen) {
    return null;
  }

  // Clamp to valid level range
  const level = Math.min(Math.max(0, currentLevel), 3);
  const cost = OVERCHARGE_COSTS[level];
  const avgHeat = OVERCHARGE_HEAT_AVG[level];

  // Calculate danger levels
  const projectedHeat = heatCurrent + avgHeat;
  const heatThreshold50 = heatCap * 0.5;
  const wouldOverheat = projectedHeat >= heatCap;
  const mayTriggerCheck = projectedHeat >= heatThreshold50;

  // Determine warning level
  const warningLevel: "none" | "yellow" | "red" = wouldOverheat
    ? "red"
    : mayTriggerCheck
      ? "yellow"
      : "none";

  return (
    <div className="rounded-md border border-border bg-muted/30 p-4 space-y-4">
      <div className="text-sm font-medium text-foreground">
        Confirm Overcharge
      </div>

      <div className="space-y-3">
        {/* Current level indicator */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Level:</span>
          <div className="flex gap-1">
            {[0, 1, 2, 3].map((idx) => (
              <div
                key={idx}
                className={`w-3 h-3 rounded-full border ${
                  idx <= level
                    ? "bg-amber-500 border-amber-600"
                    : "bg-transparent border-muted-foreground/40"
                }`}
              />
            ))}
          </div>
          <span className="text-xs text-muted-foreground ml-1">
            ({level + 1}/4)
          </span>
        </div>

        {/* Cost display */}
        <div className="p-3 rounded bg-amber-500/10 border border-amber-500/30">
          <div className="text-sm font-medium text-amber-500">
            Cost: {cost}
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            Grants +1 Quick Action this turn
          </div>
        </div>

        {/* Heat status */}
        <div className="text-xs text-muted-foreground">
          Current Heat: {heatCurrent}/{heatCap}
        </div>

        {/* Warnings */}
        {warningLevel === "yellow" && (
          <div className="p-2 rounded bg-yellow-500/10 border border-yellow-500/30">
            <div className="text-xs text-yellow-600 font-medium">
              May trigger overheat check
            </div>
            <div className="text-xs text-muted-foreground">
              Projected heat would exceed 50% capacity
            </div>
          </div>
        )}

        {warningLevel === "red" && (
          <div className="p-2 rounded bg-destructive/10 border border-destructive/30">
            <div className="text-xs text-destructive font-medium">
              High meltdown risk
            </div>
            <div className="text-xs text-muted-foreground">
              Projected heat would exceed heat cap - meltdown check required
            </div>
          </div>
        )}
      </div>

      <div className="flex gap-2 pt-2">
        <Button
          variant="primary"
          size="sm"
          onClick={onConfirm}
          className={warningLevel === "red" ? "bg-destructive hover:bg-destructive/90" : ""}
        >
          {warningLevel === "red" ? "Accept Risk" : "Overcharge"}
        </Button>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

/**
 * Get the overcharge cost text for a given level.
 */
export function getOverchargeCost(level: number): string {
  const idx = Math.min(Math.max(0, level), 3);
  return OVERCHARGE_COSTS[idx];
}
