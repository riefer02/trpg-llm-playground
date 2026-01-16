import { useCallback, useMemo, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";

import {
  useCombatSession,
  useStartTurn,
  useEndTurn,
  useExecuteAction,
  useAvailableActions,
  type ActionRequest,
  type ActionEconomyState,
  type AvailableActionItem,
} from "../../lib/api";
import { CombatCanvas, type TargetingMode } from "../../components/combat/CombatCanvas";
import {
  ActionLog,
  type SelectedAction,
} from "../../components/combat/ActionLog";
import { EconomyDisplay } from "../../components/combat/EconomyDisplay";
import { TurnControls, type TurnState } from "../../components/combat/TurnControls";
import { ActionPanel, type TargetMode } from "../../components/combat/ActionPanel";
import {
  adaptCombatScenario,
  type CombatRenderAdapterOutput,
} from "../../lib/combat-render/adapter";
import { createHexLayout } from "../../lib/combat-render/hex";
import type { HexCoord } from "../../lib/types/lancer";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../components/ui";

export const Route = createFileRoute("/combat/$combatId")({
  component: CombatSessionPage,
});

/** Polling interval when combat session is active (5 seconds) */
const ACTIVE_POLLING_INTERVAL = 5000;

function CombatSessionPage() {
  const { combatId } = Route.useParams();
  const { data, isLoading, error } = useCombatSession(combatId, {
    pollingInterval: ACTIVE_POLLING_INTERVAL,
  });

  // Turn management mutations
  const startTurn = useStartTurn(combatId);
  const endTurn = useEndTurn(combatId);
  const executeAction = useExecuteAction(combatId);

  // Turn state tracking
  const [turnActive, setTurnActive] = useState(false);
  const [economy, setEconomy] = useState<ActionEconomyState | null>(null);

  // Available actions query (only when turn is active)
  const { data: availableActions } = useAvailableActions(combatId, {
    enabled: turnActive,
  });

  // Canvas interaction state
  const [hovered, setHovered] = useState<HexCoord | null>(null);
  const [selected, setSelected] = useState<HexCoord | null>(null);
  const [targeted, setTargeted] = useState<HexCoord | null>(null);
  const [selectedAction, setSelectedAction] = useState<SelectedAction | null>(null);

  // Targeting mode state
  const [targetMode, setTargetMode] = useState<TargetMode | null>(null);
  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(null);

  const scenario = data?.scenario;
  const rounds = scenario?.rounds ?? [];
  const currentRound = data?.current_round ?? 1;
  const currentTurnIndex = data?.current_turn_index ?? 0;
  const combatants = scenario?.combatants ?? [];

  // Determine current actor from turn order
  const currentActor = useMemo(() => {
    if (!scenario) return null;
    // Find combatant based on turn order in the current round
    const round = scenario.rounds?.[currentRound - 1];
    const turn = round?.turns?.[currentTurnIndex];
    if (turn?.actor_id) {
      return combatants.find((c) => c.id === turn.actor_id) ?? null;
    }
    // Fall back to first combatant if no turn data
    return combatants[0] ?? null;
  }, [scenario, currentRound, currentTurnIndex, combatants]);

  // Derive turn state
  const turnState: TurnState = useMemo(() => {
    if (startTurn.isPending) return "not_started";
    if (endTurn.isPending) return "ending";
    if (turnActive) return "active";
    return "not_started";
  }, [startTurn.isPending, endTurn.isPending, turnActive]);

  // Handle start turn
  const handleStartTurn = useCallback(() => {
    startTurn.mutate(undefined, {
      onSuccess: (result) => {
        setTurnActive(true);
        setEconomy(result.economy);
      },
    });
  }, [startTurn]);

  // Handle end turn
  const handleEndTurn = useCallback(() => {
    endTurn.mutate(undefined, {
      onSuccess: () => {
        setTurnActive(false);
        setEconomy(null);
        setTargetMode(null);
        setSelectedTargetId(null);
      },
    });
  }, [endTurn]);

  // Handle action selection from ActionPanel
  const handleActionSelect = useCallback((_action: AvailableActionItem) => {
    // Reset target when selecting new action
    setSelectedTargetId(null);
  }, []);

  // Handle action execution
  const handleExecuteAction = useCallback(
    (request: ActionRequest) => {
      executeAction.mutate(request, {
        onSuccess: (result) => {
          if (result.success) {
            setEconomy(result.economy);
            setTargetMode(null);
            setSelectedTargetId(null);
          }
        },
      });
    },
    [executeAction]
  );

  // Handle target mode changes from ActionPanel
  const handleTargetModeChange = useCallback((mode: TargetMode | null) => {
    setTargetMode(mode);
    if (!mode) {
      setSelectedTargetId(null);
    }
  }, []);

  // Handle token click for targeting
  const handleTokenClick = useCallback(
    (tokenId: string) => {
      if (targetMode?.requiresTarget) {
        setSelectedTargetId(tokenId);
      }
    },
    [targetMode]
  );

  // Build targeting mode for canvas
  const canvasTargetingMode: TargetingMode = useMemo(() => {
    if (!targetMode?.requiresTarget) {
      return { active: false };
    }
    // For now, all enemies are valid targets for attacks
    // This could be refined based on action type, range, etc.
    const validTargetIds = combatants
      .filter((c) => c.id !== currentActor?.id)
      .map((c) => c.id);
    return {
      active: true,
      validTargetIds,
    };
  }, [targetMode, combatants, currentActor]);

  // Derive active indices from selectedAction or fall back to current position
  const activeRoundIndex = selectedAction?.roundIdx ?? clampIndex(currentRound - 1, rounds);
  const round = rounds[activeRoundIndex] ?? null;
  const turns = round?.turns ?? [];
  const activeTurnIndex = selectedAction?.turnIdx ?? clampIndex(currentTurnIndex, turns);
  const turn = turns[activeTurnIndex] ?? null;
  const actions = turn?.actions ?? [];
  const activeActionIndex = selectedAction?.actionIdx ?? 0;
  const action = actions[activeActionIndex] ?? null;

  const combatantNameById = useMemo(
    () => new Map((scenario?.combatants ?? []).map((c) => [c.id, c.name])),
    [scenario?.combatants],
  );

  const renderOutput: CombatRenderAdapterOutput | null = useMemo(() => {
    if (!scenario) {
      return null;
    }
    return adaptCombatScenario({
      scenario,
      round,
      turn,
      action,
      hover: hovered,
    });
  }, [action, hovered, round, scenario, turn]);

  if (isLoading) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <div className="text-center py-8 text-muted-foreground">
          Loading combat session...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <p className="text-destructive">
              Error loading combat session: {error.message}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!data || !scenario) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <Card>
          <CardContent className="pt-6 text-center">
            <p className="text-muted-foreground">Combat session not found</p>
            <Link to="/" className="text-primary hover:underline">
              Back to dashboard
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      <section className="dashboard-surface p-6 animate-rise">
        <Link to="/" className="text-primary hover:underline text-sm">
          ← Back to Dashboard
        </Link>
        <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-heading font-semibold text-foreground">
              {data.name}
            </h1>
            <p className="text-muted-foreground">
              Status: {data.status} · Round {data.current_round}
            </p>
          </div>
          <div className="text-xs text-muted-foreground">
            Session ID: {data.id}
          </div>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <Card className="h-full">
          <CardHeader>
            <CardTitle>Combat Canvas</CardTitle>
            <CardDescription>
              {targetMode?.requiresTarget
                ? "Click a combatant to select as target"
                : "Hover for hex highlight, left click to select, right click to target."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border border-border bg-muted/30 p-3">
              <div className="h-[520px] w-full">
                {renderOutput ? (
                  <CombatCanvas
                    width={720}
                    height={520}
                    resizeToParent
                    layout={(size) =>
                      createHexLayout(30, {
                        x: size.width / 2,
                        y: size.height / 2,
                      })
                    }
                    state={renderOutput.state}
                    styles={{
                      grid: { strokeStyle: "rgba(148, 163, 184, 0.5)" },
                      tokens: { strokeStyle: "#0f172a", lineWidth: 2 },
                      overlays: { fillStyle: "rgba(59, 130, 246, 0.12)" },
                      hover: {
                        fillStyle: "rgba(59, 130, 246, 0.2)",
                        strokeStyle: "rgba(59, 130, 246, 0.7)",
                        lineWidth: 2,
                      },
                    }}
                    targetingMode={canvasTargetingMode}
                    onHover={(coord) => setHovered(coord)}
                    onSelect={(coord) => setSelected(coord)}
                    onTarget={(coord) => setTargeted(coord)}
                    onTokenClick={handleTokenClick}
                    className="h-full w-full"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    No scenario data available yet.
                  </div>
                )}
              </div>
              <div className="mt-3 flex flex-wrap gap-4 text-sm text-muted-foreground">
                <div>Hover: {formatCoord(hovered)}</div>
                <div>Selected: {formatCoord(selected)}</div>
                <div>Targeted: {formatCoord(targeted)}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {/* Turn Controls */}
          <TurnControls
            currentActorName={currentActor?.name ?? null}
            roundNumber={currentRound}
            turnIndex={currentTurnIndex}
            turnState={turnState}
            onStartTurn={handleStartTurn}
            onEndTurn={handleEndTurn}
            isStarting={startTurn.isPending}
            isEnding={endTurn.isPending}
          />

          {/* Economy Display (only when turn is active) */}
          {turnActive && economy && (
            <EconomyDisplay
              economy={economy}
              canOvercharge={availableActions?.can_overcharge ?? false}
            />
          )}

          {/* Action Panel (only when turn is active) */}
          {turnActive && (
            <ActionPanel
              availableActions={availableActions ?? null}
              economy={economy}
              onActionSelect={handleActionSelect}
              onExecuteAction={handleExecuteAction}
              onTargetModeChange={handleTargetModeChange}
              isExecuting={executeAction.isPending}
              selectedTargetId={selectedTargetId}
            />
          )}

          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-base">Action Log</CardTitle>
              <CardDescription className="text-xs">
                Click an action to preview area overlays on the canvas.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-0">
              <div className="max-h-48 overflow-y-auto pr-1">
                <ActionLog
                  rounds={rounds}
                  currentRound={currentRound}
                  currentTurnIndex={currentTurnIndex}
                  combatantNames={combatantNameById}
                  selectedAction={selectedAction}
                  onSelectAction={(roundIdx, turnIdx, actionIdx) =>
                    setSelectedAction({ roundIdx, turnIdx, actionIdx })
                  }
                />
              </div>
              {renderOutput?.overlayMetadata.length ? (
                <div className="rounded-md border border-border bg-muted/40 p-3 text-xs text-muted-foreground space-y-2">
                  {renderOutput.overlayMetadata.map((meta) => (
                    <div key={meta.id}>
                      <div className="font-medium text-foreground">
                        {meta.pattern.toUpperCase()} size {meta.size}
                      </div>
                      <div>Origin: {formatPosition(meta.origin)}</div>
                      <div>
                        Direction:{" "}
                        {meta.direction
                          ? `${meta.direction.q},${meta.direction.r}`
                          : "--"}
                      </div>
                      <div>Coords: {meta.coordCount}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  No area overlays available for this action.
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="py-3">
              <CardTitle className="text-base">Combatants</CardTitle>
              <CardDescription className="text-xs">
                Positions mapped from the current scenario snapshot.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm pt-0">
              {combatants.map((combatant) => (
                <div
                  key={combatant.id}
                  className={`flex items-center justify-between rounded-md border px-3 py-2 ${
                    combatant.id === currentActor?.id
                      ? "border-primary bg-primary/10"
                      : "border-border bg-muted/40"
                  } ${
                    selectedTargetId === combatant.id
                      ? "ring-2 ring-green-500"
                      : ""
                  }`}
                >
                  <div>
                    <div className="font-medium">
                      {combatant.name}
                      {combatant.id === currentActor?.id && (
                        <span className="ml-2 text-xs text-primary">(active)</span>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {combatant.side} · {combatant.kind}
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {combatant.position?.coord
                      ? `${combatant.position.coord.q},${combatant.position.coord.r}`
                      : "--"}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function clampIndex<T>(value: number, list: T[]): number {
  if (!list.length) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value >= list.length) {
    return list.length - 1;
  }
  return value;
}

function formatCoord(coord: HexCoord | null): string {
  if (!coord) {
    return "--";
  }
  return `${coord.q},${coord.r}`;
}

function formatPosition(position?: { coord?: HexCoord | null } | null): string {
  if (!position?.coord) {
    return "--";
  }
  return `${position.coord.q},${position.coord.r}`;
}
