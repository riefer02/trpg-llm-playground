import type { ActionEconomyState } from "../../lib/api/combat";
import { Button } from "../ui";
import { EndTurnConfirmationDialog } from "./EndTurnConfirmationDialog";
import { useState } from "react";

export type TurnState = "not_started" | "active" | "ending";

export interface TurnControlsProps {
  currentActorName: string | null;
  roundNumber: number;
  turnIndex: number;
  turnState: TurnState;
  onStartTurn: () => void;
  onEndTurn: () => void;
  onAutoNpcTurn?: () => void;
  isStarting?: boolean;
  isEnding?: boolean;
  isAutoNpc?: boolean;
  isCurrentActorAI?: boolean;
  /** Action economy state - displayed inline when turn is active */
  economy?: ActionEconomyState | null;
  canOvercharge?: boolean;
  overchargeLevel?: number;
  /** Error message to display prominently */
  error?: string | null;
  /** Whether to show confirmation dialog when ending turn with unused actions */
  confirmEndTurn?: boolean;
}

export function TurnControls({
  currentActorName,
  roundNumber,
  turnIndex,
  turnState,
  onStartTurn,
  onEndTurn,
  onAutoNpcTurn,
  isStarting = false,
  isEnding = false,
  isAutoNpc = false,
  isCurrentActorAI = false,
  economy,
  canOvercharge = false,
  overchargeLevel = 0,
  error,
  confirmEndTurn = true,
}: TurnControlsProps) {
  // Calculate remaining actions
  const fullRemaining = economy ? 1 - economy.full_actions_used : 1;
  const quickTotal = 2 + (economy?.overcharge_used ? 1 : 0);
  const quickRemaining = economy ? quickTotal - economy.quick_actions_used : 2;
  const reactRemaining = economy ? 1 - economy.reactions_used_this_turn : 1;

  // End turn confirmation state
  const [showConfirmation, setShowConfirmation] = useState(false);

  const handleEndTurnClick = () => {
    if (confirmEndTurn && (fullRemaining > 0 || quickRemaining > 0 || reactRemaining > 0)) {
      setShowConfirmation(true);
    } else {
      onEndTurn();
    }
  };

  const handleConfirmEndTurn = () => {
    setShowConfirmation(false);
    onEndTurn();
  };

  const handleCancelEndTurn = () => {
    setShowConfirmation(false);
  };

  return (<>
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
      {/* Header row: Actor name + Turn state badge */}
      <div className="flex items-center justify-between">
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-foreground truncate">
            {currentActorName ? (
              <>
                <span className="text-primary">{currentActorName}</span>
              </>
            ) : (
              <span className="text-muted-foreground">No active turn</span>
            )}
          </div>
          <div className="text-xs text-muted-foreground">
            Round {roundNumber}, Turn {turnIndex + 1}
          </div>
        </div>

        <TurnStateBadge state={turnState} />
      </div>

      {/* Action Economy - inline compact display */}
      {turnState === "active" && economy && (
        <div className="flex items-center gap-4 py-2 px-3 rounded bg-background/50 border border-border/50">
          <EconomyItem
            label="Full"
            remaining={fullRemaining}
            total={1}
            color="primary"
          />
          <EconomyItem
            label="Quick"
            remaining={quickRemaining}
            total={quickTotal}
            color="secondary"
          />
          <EconomyItem
            label="React"
            remaining={reactRemaining}
            total={1}
            color="amber"
          />
          {canOvercharge && !economy.overcharge_used && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-500 font-medium">
              OC
            </span>
          )}
          {economy.overcharge_used && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-destructive/20 text-destructive font-medium">
              OC{overchargeLevel}
            </span>
          )}
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="p-2 rounded bg-destructive/10 border border-destructive/30 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Turn action buttons */}
      <div className="flex gap-2">
        {turnState === "not_started" && (
          <>
            <Button
              variant="primary"
              size="sm"
              onClick={onStartTurn}
              disabled={isStarting || isAutoNpc || !currentActorName}
            >
              {isStarting ? "Starting..." : "Start Turn"}
            </Button>

            {isCurrentActorAI && onAutoNpcTurn && (
              <Button
                variant="secondary"
                size="sm"
                onClick={onAutoNpcTurn}
                disabled={isAutoNpc || isStarting || !currentActorName}
              >
                {isAutoNpc ? "Processing..." : "Auto NPC"}
              </Button>
            )}
          </>
        )}

        {turnState === "active" && (
          <Button
            variant="secondary"
            size="sm"
            onClick={handleEndTurnClick}
            disabled={isEnding}
          >
            {isEnding ? "Ending..." : "End Turn"}
          </Button>
        )}

        {turnState === "ending" && (
          <Button variant="ghost" size="sm" disabled>
            Turn Ending...
          </Button>
        )}
      </div>
    </div>
    <EndTurnConfirmationDialog
      isOpen={showConfirmation}
      fullRemaining={fullRemaining}
      quickRemaining={quickRemaining}
      reactRemaining={reactRemaining}
      canOvercharge={canOvercharge && !economy?.overcharge_used}
      overchargeLevel={overchargeLevel}
      isProcessing={isEnding}
      onConfirm={handleConfirmEndTurn}
      onCancel={handleCancelEndTurn}
    />
  </>);
}

function TurnStateBadge({ state }: { state: TurnState }) {
  const styles = {
    not_started: "bg-muted text-muted-foreground",
    active: "bg-primary/20 text-primary",
    ending: "bg-secondary text-secondary-foreground",
  };

  const labels = {
    not_started: "Waiting",
    active: "Active",
    ending: "Ending",
  };

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${styles[state]}`}>
      {labels[state]}
    </span>
  );
}

interface EconomyItemProps {
  label: string;
  remaining: number;
  total: number;
  color: "primary" | "secondary" | "amber";
}

function EconomyItem({ label, remaining, total, color }: EconomyItemProps) {
  const colorClasses = {
    primary: "text-primary",
    secondary: "text-secondary-foreground",
    amber: "text-amber-500",
  };

  const dotColors = {
    primary: "bg-primary",
    secondary: "bg-secondary",
    amber: "bg-amber-500",
  };

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="flex gap-0.5">
        {Array.from({ length: total }).map((_, i) => (
          <div
            key={i}
            className={`w-2.5 h-2.5 rounded-full border ${
              i < remaining
                ? `${dotColors[color]} border-transparent`
                : "bg-transparent border-muted-foreground/40"
            }`}
          />
        ))}
      </div>
      <span className={`text-xs font-mono font-medium ${colorClasses[color]}`}>
        {remaining}
      </span>
    </div>
  );
}
