import type { CombatantState } from "../../lib/types/lancer";

/**
 * Prominent display of current actor with large HP/Heat gauges and status conditions.
 * Part of E9-US-006 - Side Panel Information Hierarchy.
 */

export interface CurrentActorPanelProps {
  /** Current actor taking their turn */
  actor: CombatantState | null;
  /** Whether the turn is currently active */
  isTurnActive: boolean;
}

export function CurrentActorPanel({ actor, isTurnActive }: CurrentActorPanelProps) {
  if (!actor) {
    return (
      <div className="rounded-lg border-2 border-dashed border-border bg-muted/20 p-4 text-center">
        <p className="text-sm text-muted-foreground">No active turn</p>
      </div>
    );
  }

  const hpCurrent = actor.resources?.hp_current ?? 0;
  const hpMax = actor.stats?.hp_max ?? 1;
  const hpPercent = Math.max(0, Math.min(100, (hpCurrent / hpMax) * 100));

  const heatCurrent = actor.resources?.heat_current ?? 0;
  const heatCap = actor.resources?.heat_cap ?? 6;
  const heatPercent = Math.max(0, Math.min(100, (heatCurrent / heatCap) * 100));

  const structureCurrent = actor.resources?.structure_current ?? 4;
  const stressCurrent = actor.resources?.stress_current ?? 4;

  const isPlayer = actor.side === "players";
  const conditions = actor.conditions ?? [];
  const statuses = actor.statuses ?? [];

  // Get HP bar color based on percentage
  const getHpColor = (percent: number) => {
    if (percent <= 25) return "bg-red-500";
    if (percent <= 50) return "bg-amber-500";
    return "bg-green-500";
  };

  // Get heat bar color based on percentage
  const getHeatColor = (percent: number) => {
    if (percent >= 75) return "bg-red-500";
    if (percent >= 50) return "bg-orange-500";
    return "bg-cyan-500";
  };

  return (
    <div className={`rounded-lg border-2 p-3 space-y-3 ${
      isTurnActive
        ? isPlayer
          ? "border-primary bg-primary/5 shadow-lg shadow-primary/10"
          : "border-red-500 bg-red-500/5 shadow-lg shadow-red-500/10"
        : "border-border bg-muted/30"
    }`}>
      {/* Actor Name Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${
            isTurnActive
              ? "animate-pulse bg-primary"
              : isPlayer ? "bg-blue-500" : "bg-red-500"
          }`} />
          <h3 className="font-semibold text-base text-foreground truncate">
            {actor.name}
          </h3>
        </div>
        {isTurnActive && (
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            isPlayer
              ? "bg-primary/20 text-primary"
              : "bg-red-500/20 text-red-500"
          }`}>
            {isPlayer ? "Your Turn" : "Enemy Turn"}
          </span>
        )}
      </div>

      {/* HP Gauge - Large and prominent */}
      <div className="space-y-1">
        <div className="flex items-baseline justify-between">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">HP</span>
          <span className="text-lg font-bold tabular-nums">
            {hpCurrent} <span className="text-sm font-normal text-muted-foreground">/ {hpMax}</span>
          </span>
        </div>
        <div className="h-4 rounded-full bg-muted/50 overflow-hidden border border-border/50">
          <div
            className={`h-full transition-all duration-300 ${getHpColor(hpPercent)} rounded-full`}
            style={{ width: `${hpPercent}%` }}
          />
        </div>
      </div>

      {/* Heat Gauge - Large and prominent */}
      <div className="space-y-1">
        <div className="flex items-baseline justify-between">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Heat</span>
          <span className="text-lg font-bold tabular-nums">
            {heatCurrent} <span className="text-sm font-normal text-muted-foreground">/ {heatCap}</span>
          </span>
        </div>
        <div className="h-4 rounded-full bg-muted/50 overflow-hidden border border-border/50">
          <div
            className={`h-full transition-all duration-300 ${getHeatColor(heatPercent)} rounded-full`}
            style={{ width: `${heatPercent}%` }}
          />
        </div>
      </div>

      {/* Structure and Stress (smaller, inline) */}
      <div className="flex gap-4">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium text-muted-foreground uppercase">Structure</span>
          <div className="flex gap-0.5">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className={`w-3 h-3 rounded-sm ${
                  i < structureCurrent ? "bg-blue-500" : "bg-muted/50 border border-border"
                }`}
              />
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium text-muted-foreground uppercase">Stress</span>
          <div className="flex gap-0.5">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className={`w-3 h-3 rounded-sm ${
                  i < stressCurrent ? "bg-orange-500" : "bg-muted/50 border border-border"
                }`}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Conditions and Statuses as badges */}
      {(conditions.length > 0 || statuses.length > 0) && (
        <div className="flex flex-wrap gap-1 pt-1 border-t border-border/50">
          {conditions.map((condition) => (
            <ConditionBadge key={condition} condition={condition} />
          ))}
          {statuses.map((status) => (
            <StatusBadge key={status} status={status} />
          ))}
        </div>
      )}
    </div>
  );
}

/** Badge for condition with tooltip */
function ConditionBadge({ condition }: { condition: string }) {
  const description = getConditionDescription(condition);

  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-amber-500/20 text-amber-600 dark:text-amber-400 cursor-help"
      title={description}
    >
      {formatConditionName(condition)}
    </span>
  );
}

/** Badge for status with tooltip */
function StatusBadge({ status }: { status: string }) {
  const description = getStatusDescription(status);

  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-purple-500/20 text-purple-600 dark:text-purple-400 cursor-help"
      title={description}
    >
      {formatConditionName(status)}
    </span>
  );
}

/** Format condition/status name for display */
function formatConditionName(name: string): string {
  return name
    .split("_")
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/** Get description for a condition (for tooltip) */
function getConditionDescription(condition: string): string {
  const descriptions: Record<string, string> = {
    immobilized: "Cannot move voluntarily. Can still take actions.",
    impaired: "All attacks deal half damage.",
    jammed: "Cannot take actions that require a die roll.",
    lock_on: "Next attack against this target gains +1 accuracy.",
    shredded: "Armor is reduced to 0.",
    slowed: "Speed is halved.",
    stunned: "Cannot take actions or reactions.",
    prone: "Must spend movement to stand. Melee attacks gain +1 accuracy, ranged attacks suffer +1 difficulty.",
    hidden: "Cannot be targeted by hostile actions. Broken by attacking or being attacked.",
    invisible: "All attacks against suffer +2 difficulty.",
    exposed: "Next attack that hits deals double damage.",
    shutdown: "Cannot take any actions. All systems offline.",
    bolstered: "Gains +2 to all saves until end of next turn.",
    braced: "Cannot move. Gains resistance to knockback and immunity to prone.",
  };
  return descriptions[condition] ?? "A combat condition affecting this combatant.";
}

/** Get description for a status (for tooltip) */
function getStatusDescription(status: string): string {
  const descriptions: Record<string, string> = {
    braced: "Cannot move. Gains resistance to knockback.",
    danger_zone: "Heat is at or above half capacity.",
    engaged: "In melee range with an enemy.",
    flying: "Currently airborne.",
    grappled: "Held by another combatant.",
    grappling: "Holding another combatant.",
  };
  return descriptions[status] ?? "A status effect on this combatant.";
}
