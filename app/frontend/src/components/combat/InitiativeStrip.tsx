import type { CombatantState } from "../../lib/types/lancer";

/**
 * Compact horizontal initiative order strip.
 * Part of E9-US-006 - Side Panel Information Hierarchy.
 */

export interface InitiativeStripProps {
  /** All combatants in turn order */
  combatants: CombatantState[];
  /** ID of the current actor */
  currentActorId: string | null;
  /** Current turn index */
  turnIndex: number;
  /** Current round number */
  roundNumber: number;
}

export function InitiativeStrip({
  combatants,
  currentActorId,
  turnIndex,
  roundNumber,
}: InitiativeStripProps) {
  if (combatants.length === 0) {
    return null;
  }

  return (
    <div className="rounded-md border border-border bg-muted/20 px-2 py-1.5">
      <div className="flex items-center gap-1.5 overflow-x-auto scrollbar-thin scrollbar-thumb-muted">
        {/* Round indicator */}
        <span className="text-[9px] font-medium text-muted-foreground uppercase shrink-0">
          R{roundNumber}
        </span>
        <div className="w-px h-4 bg-border shrink-0" />

        {/* Initiative tokens */}
        {combatants.map((actor, index) => {
          const isCurrent = actor.id === currentActorId;
          const isPast = index < turnIndex;
          const isPlayer = actor.side === "players";

          return (
            <div
              key={actor.id}
              className={`shrink-0 flex items-center gap-1 px-1.5 py-0.5 rounded transition-all ${
                isCurrent
                  ? isPlayer
                    ? "bg-primary text-primary-foreground shadow-sm ring-1 ring-primary"
                    : "bg-red-500 text-white shadow-sm ring-1 ring-red-500"
                  : isPast
                    ? "bg-muted/40 text-muted-foreground/50"
                    : isPlayer
                      ? "bg-blue-500/10 text-blue-500"
                      : "bg-red-500/10 text-red-500"
              }`}
              title={`${actor.name} (${isPlayer ? "Ally" : "Enemy"})${isCurrent ? " - Current" : ""}${isPast ? " - Already acted" : ""}`}
            >
              {/* Numeric position */}
              <span className="text-[8px] font-medium opacity-60">
                {index + 1}
              </span>
              {/* Abbreviated name */}
              <span className="text-[10px] font-medium max-w-[50px] truncate">
                {getAbbreviatedName(actor.name)}
              </span>
              {/* HP indicator dot */}
              <HpDot actor={actor} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Small HP indicator dot */
function HpDot({ actor }: { actor: CombatantState }) {
  const hpCurrent = actor.resources?.hp_current ?? 0;
  const hpMax = actor.stats?.hp_max ?? 1;
  const hpPercent = (hpCurrent / hpMax) * 100;

  const color = hpPercent <= 25
    ? "bg-red-500"
    : hpPercent <= 50
      ? "bg-amber-500"
      : "bg-green-500";

  return (
    <div
      className={`w-1.5 h-1.5 rounded-full ${color}`}
      title={`HP: ${hpCurrent}/${hpMax}`}
    />
  );
}

/** Get abbreviated name (first word or initial + last) */
function getAbbreviatedName(name: string): string {
  // If name is short enough, use it directly
  if (name.length <= 8) return name;

  // Try to get callsign (often in quotes or parentheses)
  const callsignMatch = name.match(/["']([^"']+)["']|\(([^)]+)\)/);
  if (callsignMatch) {
    const callsign = callsignMatch[1] || callsignMatch[2];
    if (callsign && callsign.length <= 8) return callsign;
  }

  // Get first word
  const firstWord = name.split(/[\s-]/)[0];
  if (firstWord.length <= 8) return firstWord;

  // Truncate
  return firstWord.slice(0, 7) + "…";
}
