import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { ActionLog, type SelectedAction } from "./ActionLog";
import type { CombatRound } from "../../lib/types/lancer";

/**
 * Collapsible wrapper for the ActionLog component.
 * Part of E9-US-006 - Side Panel Information Hierarchy.
 */

export interface CollapsibleActionLogProps {
  rounds: CombatRound[];
  currentRound: number;
  currentTurnIndex: number;
  combatantNames: Map<string, string>;
  selectedAction: SelectedAction | null;
  onSelectAction: (roundIdx: number, turnIdx: number, actionIdx: number) => void;
  /** Default collapsed state */
  defaultCollapsed?: boolean;
}

export function CollapsibleActionLog({
  rounds,
  currentRound,
  currentTurnIndex,
  combatantNames,
  selectedAction,
  onSelectAction,
  defaultCollapsed = true,
}: CollapsibleActionLogProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  // Count total actions for badge
  const totalActions = rounds.reduce((acc, round) => {
    return acc + (round.turns ?? []).reduce((turnAcc, turn) => {
      return turnAcc + (turn.actions ?? []).length;
    }, 0);
  }, 0);

  return (
    <div className="rounded-md border border-border bg-muted/30 overflow-hidden">
      {/* Header - always visible */}
      <button
        type="button"
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between px-2 py-1.5 hover:bg-muted/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
            Action Log
          </span>
          {totalActions > 0 && (
            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">
              {totalActions}
            </span>
          )}
        </div>
        {isCollapsed ? (
          <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
        ) : (
          <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" />
        )}
      </button>

      {/* Content - collapsible */}
      {!isCollapsed && (
        <div className="border-t border-border px-2 pb-2">
          <div className="max-h-32 overflow-y-auto">
            <ActionLog
              rounds={rounds}
              currentRound={currentRound}
              currentTurnIndex={currentTurnIndex}
              combatantNames={combatantNames}
              selectedAction={selectedAction}
              onSelectAction={onSelectAction}
            />
          </div>
        </div>
      )}
    </div>
  );
}
