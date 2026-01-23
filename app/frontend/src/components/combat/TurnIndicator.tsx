import { useEffect, useState } from "react";
import { ChevronDown, ChevronUp, User, Bot } from "lucide-react";

import type { CombatantState } from "../../lib/types/lancer";

/**
 * Prominent turn indicator and initiative order widget.
 * Shows "YOUR TURN" banner and collapsible turn order.
 */

export interface TurnIndicatorProps {
  /** Current actor taking their turn */
  currentActor: CombatantState | null;
  /** All combatants in turn order */
  combatants: CombatantState[];
  /** Current round number */
  roundNumber: number;
  /** Current turn index within the round */
  turnIndex: number;
  /** Whether the turn is active (started but not ended) */
  isTurnActive: boolean;
  /** Whether it's a player-controlled actor's turn */
  isPlayerTurn: boolean;
  /** Callback when clicking to expand/collapse turn order */
  onToggleExpanded?: () => void;
}

export function TurnIndicator({
  currentActor,
  combatants,
  roundNumber,
  turnIndex,
  isTurnActive,
  isPlayerTurn,
}: TurnIndicatorProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showBanner, setShowBanner] = useState(false);

  // Show "YOUR TURN" banner briefly when turn starts
  useEffect(() => {
    if (isTurnActive && isPlayerTurn) {
      setShowBanner(true);
      const timer = setTimeout(() => setShowBanner(false), 3000);
      return () => clearTimeout(timer);
    } else {
      setShowBanner(false);
    }
  }, [isTurnActive, isPlayerTurn, currentActor?.id]);

  // Get upcoming actors (next 2-3 after current)
  const upcomingActors = getUpcomingActors(combatants, turnIndex, 3);

  return (
    <>
      {/* Prominent "YOUR TURN" Banner */}
      {showBanner && (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 z-50 animate-in fade-in slide-in-from-top duration-300">
          <div className="bg-primary text-primary-foreground px-8 py-3 rounded-lg shadow-2xl border-2 border-primary-foreground/20">
            <div className="text-2xl font-bold font-heading tracking-wide text-center">
              YOUR TURN
            </div>
            <div className="text-sm text-center opacity-80">
              {currentActor?.name ?? "Unknown"}
            </div>
          </div>
        </div>
      )}

      {/* Turn Order Widget */}
      <div className="rounded-md border border-border bg-muted/30 overflow-hidden">
        {/* Current Actor Header */}
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full flex items-center justify-between p-3 hover:bg-muted/50 transition-colors"
        >
          <div className="flex items-center gap-3 min-w-0">
            <ActorIcon actor={currentActor} isActive={isTurnActive} />
            <div className="min-w-0 flex-1 text-left">
              <div className="font-medium text-sm text-foreground truncate">
                {currentActor?.name ?? "No active turn"}
              </div>
              <div className="text-xs text-muted-foreground">
                Round {roundNumber}, Turn {turnIndex + 1}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isTurnActive && (
              <span className="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary font-medium">
                Active
              </span>
            )}
            {isExpanded ? (
              <ChevronUp className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            )}
          </div>
        </button>

        {/* Collapsed Preview: Next 2-3 actors */}
        {!isExpanded && upcomingActors.length > 0 && (
          <div className="px-3 pb-2 flex items-center gap-1">
            <span className="text-[10px] text-muted-foreground uppercase mr-1">Next:</span>
            {upcomingActors.map((actor) => (
              <span
                key={actor.id}
                className={`text-xs px-1.5 py-0.5 rounded ${
                  actor.side === "players"
                    ? "bg-blue-500/20 text-blue-400"
                    : "bg-red-500/20 text-red-400"
                }`}
              >
                {truncateName(actor.name, 8)}
              </span>
            ))}
          </div>
        )}

        {/* Expanded: Full Initiative Order */}
        {isExpanded && (
          <div className="border-t border-border max-h-[200px] overflow-y-auto">
            {combatants.map((actor, index) => {
              const isCurrent = actor.id === currentActor?.id;
              const isPast = index < turnIndex;
              return (
                <div
                  key={actor.id}
                  className={`flex items-center gap-2 px-3 py-2 text-sm ${
                    isCurrent
                      ? "bg-primary/10 border-l-2 border-primary"
                      : isPast
                        ? "opacity-50"
                        : ""
                  }`}
                >
                  <span className="w-5 text-xs text-muted-foreground text-right">
                    {index + 1}.
                  </span>
                  <ActorIcon actor={actor} isActive={isCurrent && isTurnActive} size="sm" />
                  <span className="flex-1 truncate">{actor.name}</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded ${
                      actor.side === "players"
                        ? "bg-blue-500/20 text-blue-400"
                        : "bg-red-500/20 text-red-400"
                    }`}
                  >
                    {actor.side === "players" ? "Ally" : "Enemy"}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}

interface ActorIconProps {
  actor: CombatantState | null;
  isActive?: boolean;
  size?: "sm" | "md";
}

function ActorIcon({ actor, isActive, size = "md" }: ActorIconProps) {
  const isPlayer = actor?.side === "players";
  const iconSize = size === "sm" ? "w-4 h-4" : "w-6 h-6";
  const containerSize = size === "sm" ? "w-6 h-6" : "w-8 h-8";

  return (
    <div
      className={`${containerSize} rounded-full flex items-center justify-center ${
        isActive
          ? "bg-primary text-primary-foreground animate-pulse"
          : isPlayer
            ? "bg-blue-500/20 text-blue-400"
            : "bg-red-500/20 text-red-400"
      }`}
    >
      {isPlayer || !actor?.ai_controlled ? (
        <User className={iconSize} />
      ) : (
        <Bot className={iconSize} />
      )}
    </div>
  );
}

function getUpcomingActors(
  combatants: CombatantState[],
  currentIndex: number,
  count: number
): CombatantState[] {
  const upcoming: CombatantState[] = [];
  for (let i = 1; i <= count && currentIndex + i < combatants.length; i++) {
    upcoming.push(combatants[currentIndex + i]);
  }
  return upcoming;
}

function truncateName(name: string, maxLength: number): string {
  if (name.length <= maxLength) return name;
  return name.slice(0, maxLength - 1) + "…";
}
