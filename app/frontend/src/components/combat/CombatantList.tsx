import type { CombatantState } from "../../lib/types/lancer";

/**
 * Compact combatant list with HP bars visible at a glance.
 * Part of E9-US-006 - Side Panel Information Hierarchy.
 */

export interface CombatantListProps {
  /** All combatants */
  combatants: CombatantState[];
  /** ID of the current actor */
  currentActorId: string | null;
  /** Currently selected target IDs (for targeting highlights) */
  selectedTargetIds?: string[];
  /** Callback when clicking a combatant */
  onCombatantClick?: (id: string) => void;
}

export function CombatantList({
  combatants,
  currentActorId,
  selectedTargetIds = [],
  onCombatantClick,
}: CombatantListProps) {
  // Group combatants by side
  const players = combatants.filter(c => c.side === "players");
  const hostiles = combatants.filter(c => c.side !== "players");

  return (
    <div className="rounded-md border border-border bg-muted/30 overflow-hidden">
      <div className="px-2 py-1.5 border-b border-border">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
          Combatants
        </span>
      </div>

      <div className="divide-y divide-border/50">
        {/* Players section */}
        {players.length > 0 && (
          <div className="py-1">
            <div className="px-2 py-0.5">
              <span className="text-[9px] font-medium text-blue-500 uppercase">Allies</span>
            </div>
            <div className="space-y-0.5">
              {players.map(combatant => (
                <CombatantRow
                  key={combatant.id}
                  combatant={combatant}
                  isCurrent={combatant.id === currentActorId}
                  isSelected={selectedTargetIds.includes(combatant.id)}
                  onClick={onCombatantClick}
                />
              ))}
            </div>
          </div>
        )}

        {/* Hostiles section */}
        {hostiles.length > 0 && (
          <div className="py-1">
            <div className="px-2 py-0.5">
              <span className="text-[9px] font-medium text-red-500 uppercase">Enemies</span>
            </div>
            <div className="space-y-0.5">
              {hostiles.map(combatant => (
                <CombatantRow
                  key={combatant.id}
                  combatant={combatant}
                  isCurrent={combatant.id === currentActorId}
                  isSelected={selectedTargetIds.includes(combatant.id)}
                  onClick={onCombatantClick}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface CombatantRowProps {
  combatant: CombatantState;
  isCurrent: boolean;
  isSelected: boolean;
  onClick?: (id: string) => void;
}

function CombatantRow({ combatant, isCurrent, isSelected, onClick }: CombatantRowProps) {
  const hpCurrent = combatant.resources?.hp_current ?? 0;
  const hpMax = combatant.stats?.hp_max ?? 1;
  const hpPercent = Math.max(0, Math.min(100, (hpCurrent / hpMax) * 100));
  const isPlayer = combatant.side === "players";
  const conditions = combatant.conditions ?? [];

  // HP bar color
  const getHpColor = () => {
    if (hpPercent <= 25) return "bg-red-500";
    if (hpPercent <= 50) return "bg-amber-500";
    return isPlayer ? "bg-blue-500" : "bg-red-500";
  };

  const handleClick = () => {
    if (onClick) {
      onClick(combatant.id);
    }
  };

  return (
    <div
      className={`flex items-center gap-2 px-2 py-1 cursor-pointer hover:bg-muted/50 transition-colors ${
        isCurrent ? "bg-primary/10 border-l-2 border-primary" : ""
      } ${isSelected ? "ring-1 ring-green-500 ring-inset" : ""}`}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleClick();
        }
      }}
    >
      {/* Name */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1">
          <span className="text-xs font-medium truncate">
            {combatant.name}
          </span>
          {isCurrent && (
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          )}
        </div>

        {/* HP Bar - inline */}
        <div className="flex items-center gap-1.5 mt-0.5">
          <div className="flex-1 h-1.5 rounded-full bg-muted/50 overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${getHpColor()} rounded-full`}
              style={{ width: `${hpPercent}%` }}
            />
          </div>
          <span className="text-[9px] tabular-nums text-muted-foreground shrink-0">
            {hpCurrent}/{hpMax}
          </span>
        </div>
      </div>

      {/* Condition count indicator */}
      {conditions.length > 0 && (
        <span
          className="shrink-0 text-[9px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-600 dark:text-amber-400"
          title={conditions.map(c => formatCondition(c)).join(", ")}
        >
          {conditions.length}
        </span>
      )}
    </div>
  );
}

function formatCondition(condition: string): string {
  return condition
    .split("_")
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
