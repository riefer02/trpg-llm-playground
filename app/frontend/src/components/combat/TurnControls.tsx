import { Button } from "../ui";

export type TurnState = "not_started" | "active" | "ending";

export interface TurnControlsProps {
  currentActorName: string | null;
  roundNumber: number;
  turnIndex: number;
  turnState: TurnState;
  onStartTurn: () => void;
  onEndTurn: () => void;
  isStarting?: boolean;
  isEnding?: boolean;
}

export function TurnControls({
  currentActorName,
  roundNumber,
  turnIndex,
  turnState,
  onStartTurn,
  onEndTurn,
  isStarting = false,
  isEnding = false,
}: TurnControlsProps) {
  return (
    <div className="rounded-md border border-border bg-muted/30 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium text-foreground">
            {currentActorName ? (
              <>Current Turn: <span className="text-primary">{currentActorName}</span></>
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

      <div className="flex gap-2">
        {turnState === "not_started" && (
          <Button
            variant="primary"
            size="sm"
            onClick={onStartTurn}
            disabled={isStarting || !currentActorName}
          >
            {isStarting ? "Starting..." : "Start Turn"}
          </Button>
        )}

        {turnState === "active" && (
          <Button
            variant="secondary"
            size="sm"
            onClick={onEndTurn}
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
  );
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
